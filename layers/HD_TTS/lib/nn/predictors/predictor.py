from typing import Callable, Mapping, Optional, Type, Union, List, Tuple

import torch
from torch_geometric.data.storage import recursive_apply
from torchmetrics import Metric
from layers.tsl.engines import Predictor


class MultiScalePredictor(Predictor):

    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 whiten_prob: Optional[Union[float, List[float]]] = None,
                 imputation_loss_fn: Optional[Callable] = None,
                 imputation_loss_weight: float = 1.,
                 imputation_warm_up: Union[int, Tuple[int, int]] = 0,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super().__init__(model=model,
                         model_class=model_class,
                         model_kwargs=model_kwargs,
                         optim_class=optim_class,
                         optim_kwargs=optim_kwargs,
                         loss_fn=loss_fn,
                         scale_target=scale_target,
                         metrics=metrics,
                         scheduler_class=scheduler_class,
                         scheduler_kwargs=scheduler_kwargs)
        self.whiten_prob = whiten_prob
        self.imputation_loss_weight = imputation_loss_weight

        if imputation_loss_fn is not None:
            self.imputation_loss_fn = self._check_metric(imputation_loss_fn,
                                                         on_step=True)
        else:
            self.imputation_loss_fn = None

        if isinstance(imputation_warm_up, int):
            self.imputation_warm_up = (imputation_warm_up, 0)
        else:
            self.imputation_warm_up = tuple(imputation_warm_up)
        if len(self.imputation_warm_up) != 2:
            raise ValueError(
                "'imputation_warm_up' must be an int of time steps to "
                "be cut at the beginning of the sequence or a "
                "pair of int if the sequence must be trimmed in a "
                "bidirectional way.")

    def predict_batch(self, batch,
                      preprocess: bool = False, postprocess: bool = True,
                      return_target: bool = False,
                      **forward_kwargs):
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        out = self.forward(**inputs, **forward_kwargs)
        if isinstance(out, tuple):
            y_hat, x_hat, scores, attn_weights = out
        else:
            y_hat, x_hat, scores, attn_weights = out, None, None, None
        # Rescale outputs
        if postprocess:
            y_trans = transform.get('y')
            if y_trans is not None:
                y_hat = y_trans.inverse_transform(y_hat)
            x_trans = transform.get('x')
            if x_trans is not None:
                x_hat = x_trans.inverse_transform(x_hat)
        if return_target:
            y = targets.get('y')
            return y, y_hat, mask
        return y_hat, x_hat, scores, attn_weights

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat, _, _, _ = self.predict_batch(batch, preprocess=False,
                                            postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    # Imputation data hooks ###################################################

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        r"""For every training batch, randomly mask out value with probability
        :obj:`p = self.whiten_prob`. Then, whiten missing values in
        :obj:`batch.input.x`."""
        batch.original_mask = batch.input_mask
        if self.whiten_prob is not None:
            # randomly mask out value with probability p = whiten_prob
            mask = batch.input_mask
            p = self.whiten_prob
            whiten_mask = torch.rand(mask.size(), device=mask.device) > p
            batch.input_mask = batch.input_mask & whiten_mask
            # whiten missing values
            if 'x' in batch.input:
                injected_missing = batch.original_mask ^ batch.input_mask
                batch.input.x = torch.where(injected_missing,
                                            torch.zeros_like(batch.input.x),
                                            batch.input.x)

    def trim_warm_up(self, *args):
        """Trim all tensors in :obj:`args` removing a number of first and last
        steps equals to :obj:`(self.warm_up_steps[0], self.warm_up_steps[1])`,
        respectively."""
        left, right = self.imputation_warm_up
        # assume time in second dimension (after batch dim)
        trim = lambda s: s[:, left:s.size(1) - right]  # noqa
        args = recursive_apply(args, trim)
        if len(args) == 1:
            return args[0]
        return args

    def compute_imputation_loss(self, x_hat, batch, scale_target,
                                step: str):
        assert step == 'train'

        mask = batch.original_mask
        x, x_hat, mask = self.trim_warm_up(batch.x, x_hat, mask)

        # Scale target and output, eventually
        if scale_target:
            x = batch.transform['x'].transform(x)

        if isinstance(x_hat, (list, tuple)):
            loss = sum(self.imputation_loss_fn(x_p, x, mask) for x_p in x_hat)
        else:
            loss = self.imputation_loss_fn(x_hat, x, mask)

        self.log_loss(f'{step}_imputation', loss, batch_size=batch.batch_size)
        return loss

    def compute_prediction_loss(self, y_hat, batch, scale_target,
                                step: str):
        # assert step in ['train', 'val', 'test']
        y_hat_loss = y_hat
        y_hat = y_hat.detach()
        y_loss = y = batch.y
        mask = batch.mask

        # Scale target and output, eventually
        if scale_target:
            y_loss = batch.transform['y'].transform(y_loss)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        metrics = getattr(self, f'{step}_metrics')
        metrics.update(y_hat, y, mask)
        self.log_metrics(metrics, batch_size=batch.batch_size)
        self.log_loss(step, loss, batch_size=batch.batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        """"""
        # Compute predictions and compute loss
        y_hat, x_hat, _, _ = self.predict_batch(
            batch, preprocess=False, postprocess=not self.scale_target)

        loss = self.compute_prediction_loss(
            y_hat, batch, scale_target=self.scale_target, step='train'
        )
        if self.imputation_loss_fn is not None and x_hat is not None:
            loss += self.imputation_loss_weight * self.compute_imputation_loss(
                x_hat, batch, scale_target=self.scale_target, step='train'
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """"""
        # Compute predictions and compute loss
        y_hat, x_hat, scores, attn_weights = self.predict_batch(
            batch, preprocess=False, postprocess=not self.scale_target)

        loss = self.compute_prediction_loss(
            y_hat, batch, scale_target=self.scale_target, step='val'
        )

        return loss

    def test_step(self, batch, batch_idx):
        """"""
        # Compute outputs and rescale
        y_hat, _, scores, attn_weights = self.predict_batch(batch,
                                                            preprocess=False,
                                                            postprocess=True)

        y, mask = batch.y, batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss
