from typing import Optional, Callable, Mapping, Type

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from layers.higp_lib.nn.hierarchical.ops import build_Q, src_reduce
from layers.higp_lib.nn.hierarchical.projection_layer import ProjectionLayer
from layers.tsl.engines import Predictor
from layers.tsl.experiment import NeptuneLogger
from layers.tsl.utils import ensure_list


class HierPredictor(Predictor):
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 warm_up: int = -1,
                 reconciliation_start_epoch: int = -1,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 forecast_reconciliation: bool = False,
                 lam: float = 0.,
                 beta: float = 1.,
                 levels: int = 3,
                 fit_base_level_only: bool = False,
                 log_clusters: bool = False,
                 ):
        self.levels = levels
        super(HierPredictor, self).__init__(model=model,
                                            model_class=model_class,
                                            model_kwargs=model_kwargs,
                                            optim_class=optim_class,
                                            optim_kwargs=optim_kwargs,
                                            loss_fn=loss_fn,
                                            scale_target=scale_target,
                                            metrics=metrics,
                                            scheduler_class=scheduler_class,
                                            scheduler_kwargs=scheduler_kwargs)
        self.warm_up = warm_up
        self.reconciliation_start_epoch = reconciliation_start_epoch
        self.lam = lam
        self.beta = beta
        self._C = None
        self.fit_only_base_level = fit_base_level_only
        self.log_clusters = log_clusters
        if forecast_reconciliation:
            self.reconciliation_step = ProjectionLayer()
        else:
            self.register_parameter('reconciliation_step', None)

    def _set_metrics(self, metrics):
        super(HierPredictor, self)._set_metrics(metrics)

        self.hier_train_metrics = nn.ModuleList([MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='train_', postfix=f'_level_{l}') for l in range(self.levels - 1, -1, -1)])

        self.hier_val_metrics = nn.ModuleList([MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='val_', postfix=f'_level_{l}') for l in range(self.levels - 1, -1, -1)])

        self.hier_test_metrics = nn.ModuleList([MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in metrics.items()},
            prefix='test_', postfix=f'_level_{l}') for l in range(self.levels - 1, -1, -1)])

    def update_and_log_hier_metrics(self,
                                    metrics: nn.ModuleList,
                                    ys_hat,
                                    ys,
                                    mask,
                                    **kwargs):
        masks = [None, ] * (self.levels - 1) + [mask, ]
        for l, (m, y_hat, y, msk) in enumerate(zip(metrics, ys_hat, ys, masks)):
            m.update(y_hat, y, mask=msk)
            self.log_metrics(m, **kwargs)

    @property
    def reconcile_forecasts(self):
        if self.training and self.current_epoch < self.reconciliation_start_epoch:
            return False
        return self.reconciliation_step is not None

    @property
    def fit_hier_levels(self):
        return not self.fit_only_base_level and self.current_epoch >= self.warm_up

    def maybe_reconcile(self, y_hat, C):
        if self.reconcile_forecasts:
            Q = build_Q(C)
            y_hat_p = self.reconciliation_step(y_hat, Q)
            if self.training:
                return torch.stack([y_hat, y_hat_p], dim=0)
            return y_hat_p
        return y_hat

    def forward(self, *args, **kwargs):
        y_hat, C, sizes, reg = super(HierPredictor, self).forward(*args, **kwargs)
        y_hat = self.maybe_reconcile(y_hat, C)
        return y_hat, C, sizes, reg

    def predict_batch(self, batch,
                      preprocess: bool = False,
                      postprocess: bool = True,
                      return_target: bool = False,
                      return_hierarchy: bool = False,
                      **forward_kwargs):
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        out = self.forward(**inputs, **forward_kwargs)
        y_hat, C, sizes, reg = out
        if not return_hierarchy:
            y_hat = torch.split(y_hat, sizes, dim=-2)[-1]

        # Rescale outputs
        if postprocess:
            trans = transform.get('y')
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)

        if return_target:
            y = targets.get('y')
            if return_hierarchy:
                # take the average by considering number of nodes
                y_ = src_reduce(y, C.transpose(-2, -1))
                y = torch.cat([y_, y], dim=-2)
                return y, y_hat, C, sizes, mask, reg
            return y, y_hat, mask
        if return_hierarchy:
            return y_hat, C, sizes, reg
        return y_hat

    def shared_step(self, batch, batch_idx):
        """"""
        # Compute predictions and compute loss
        y_loss, y_hat_loss, C, sizes, mask, reg_losses = self.predict_batch(batch,
                                                                            preprocess=False,
                                                                            postprocess=not self.scale_target,
                                                                            return_hierarchy=True,
                                                                            return_target=True)
        y = y_loss
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        return (y, y_loss), (y_hat, y_hat_loss), C, sizes, mask, reg_losses

    def compute_loss(self,
                     pred,
                     target,
                     mask,
                     sizes,
                     batch_size,
                     name):
        preds = torch.split(pred, sizes, dim=-2)
        targets = torch.split(target, sizes, dim=-2)
        loss = 0.
        loss_w = 1.
        if self.fit_hier_levels:
            for i in range(len(sizes) - 1):
                loss_i = self.loss_fn(preds[i], targets[i], None)
                self.log_loss(f'{name}_level_{len(sizes) - 1 - i}',
                              loss_i,
                              batch_size=batch_size)
                w_i = sizes[i] / sum(sizes)
                # w_i = max(sizes[i] / sum(sizes), 0.1)
                loss += w_i * loss_i
                loss_w -= w_i
        loss_base = self.loss_fn(preds[-1], targets[-1], mask)
        self.log_loss(f'{name}_base',
                      loss_base,
                      batch_size=batch_size)
        loss_base *= loss_w
        loss += loss_base
        return loss

    def training_step(self, batch, batch_idx):
        (y, y_loss), (y_hat, y_hat_loss), C, sizes, mask, reg_losses = self.shared_step(batch, batch_idx)

        # Compute loss
        loss = 0.
        if y_hat_loss.dim() == 5:
            for i, pred in enumerate(list(y_hat_loss)):
                loss += self.compute_loss(pred,
                                          y_loss,
                                          mask,
                                          sizes,
                                          batch.batch_size,
                                          name=f'train_loss_{i}')
            loss /= y_hat_loss.size(0)
        else:
            loss += self.compute_loss(y_hat_loss,
                                      y_loss,
                                      mask,
                                      sizes,
                                      batch.batch_size,
                                      name=f'train_loss')

        if self.fit_hier_levels:
            for i, l_reg in enumerate(reg_losses):
                # loss += self.lam * l_reg
                loss += self.beta * l_reg
                self.log_loss(f'reg_loss_{i}',
                              l_reg,
                              batch_size=batch.batch_size)

        if self.fit_hier_levels:
            if not self.reconcile_forecasts:
                Q = build_Q(C)
                if Q.dim() == 3:
                    Q = Q.unsqueeze(1)
                coherence_loss = torch.matmul(Q, y_hat_loss)
                coherence_loss = torch.norm(coherence_loss,
                                            p=2,
                                            dim=-1).mean()
            else:
                assert y_hat_loss.dim() == 5
                assert y_hat.dim() == 5
                coherence_loss = torch.norm(y_hat_loss[0] - y_hat_loss[1],
                                            p=2,
                                            dim=-1).mean()

            self.log_loss('coherence_loss', coherence_loss, batch_size=batch.batch_size)
            loss += self.lam * coherence_loss

        # Logging
        # Compute metrics on base level
        if y_hat.dim() == 5:
            y_hat = y_hat[-1]

        ys = torch.split(y, sizes, dim=-2)
        ys_hat = torch.split(y_hat, sizes, dim=-2)
        # hier
        self.update_and_log_hier_metrics(
            metrics=self.hier_train_metrics,
            ys_hat=ys_hat,
            ys=ys,
            mask=mask,
            batch_size=batch.batch_size
        )

        self.train_metrics.update(ys_hat[-1], ys[-1], mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        ######
        t = self.model.hierarchy_builder.pooling_layers[0]._temp
        self.log('temp', t, on_step=True, on_epoch=False)
        ######
        if self.log_clusters:
            if C.dim() == 3:
                self._C = C.detach().cpu().mean(0).numpy()
            else:
                self._C = C.detach().cpu().numpy()
        return loss

    def validation_step(self, batch, batch_idx):
        (y, y_loss), (y_hat, y_hat_loss), C, sizes, mask, _ = self.shared_step(batch, batch_idx)

        loss = self.compute_loss(y_hat_loss,
                                 y_loss,
                                 mask,
                                 sizes,
                                 batch.batch_size,
                                 name=f'val_loss')
        # Logging
        # Compute metrics on base level
        ys = torch.split(y, sizes, dim=-2)
        ys_hat = torch.split(y_hat, sizes, dim=-2)
        # hier
        self.update_and_log_hier_metrics(
            metrics=self.hier_val_metrics,
            ys_hat=ys_hat,
            ys=ys,
            mask=mask,
            batch_size=batch.batch_size
        )

        self.val_metrics.update(ys_hat[-1], ys[-1], mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', loss, batch_size=batch.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        (y, y_loss), (y_hat, y_hat_loss), C, sizes, mask, reg_losses = self.shared_step(batch, batch_idx)

        loss = self.compute_loss(y_hat_loss,
                                 y_loss,
                                 mask,
                                 sizes,
                                 batch.batch_size,
                                 name=f'test_loss')
        # Logging

        # Compute metrics on base level
        ys = torch.split(y, sizes, dim=-2)
        ys_hat = torch.split(y_hat, sizes, dim=-2)
        # hier
        self.update_and_log_hier_metrics(
            metrics=self.hier_test_metrics,
            ys_hat=ys_hat,
            ys=ys,
            mask=mask,
            batch_size=batch.batch_size
        )

        self.test_metrics.update(ys_hat[-1], ys[-1], mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', loss, batch_size=batch.batch_size)
        return loss

    def on_train_epoch_start(self) -> None:
        # Log learning rate
        optimizers = ensure_list(self.optimizers())
        for i, optimizer in enumerate(optimizers):
            lr = optimizer.optimizer.param_groups[0]['lr']
            self.log(f'lr_{i}', lr, on_step=False, on_epoch=True, logger=True,
                     prog_bar=False, batch_size=1)
        if self.current_epoch == self.reconciliation_start_epoch and self.reconcile_forecasts:
            if hasattr(self.model, 'single_sample'):
                self.model.single_step = False

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self._C is not None:
            if self.current_epoch % 10 == 0:
                self.log_aggregation_matrix(self._C)
            self._C = None

    def log_aggregation_matrix(self, C):
        if isinstance(self.logger, NeptuneLogger):
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(C, vmin=0, vmax=1)
            ax.set_title(f"Aggregation matrix at step: {self.global_step:6d}")
            plt.tight_layout()
            self.logger.log_figure(fig, f'clusters/step{self.global_step}')
            plt.close()

    def log_loss(self, name, loss, on_step=False, **kwargs):
        """"""
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=on_step,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False,
                 **kwargs)
