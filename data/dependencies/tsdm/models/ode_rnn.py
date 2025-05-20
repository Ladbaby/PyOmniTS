r"""ODR-RNN Model Import."""

__all__ = [
    # Classes
    "ODE_RNN",
]


import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from data.dependencies.tsdm.models._models import BaseModel
from data.dependencies.tsdm.utils import deep_dict_update


@contextmanager
def add_to_path(p: Path) -> Iterator:
    r"""Append path to environment variable PATH.

    Parameters
    ----------
    p: Path

    References
    ----------
    - https://stackoverflow.com/a/41904558/9318372
    """
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path = old_path


def path_import(module_path: Path, module_name: str = None) -> ModuleType:
    r"""Return python module imported from path.

    References
    ----------
    - https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    - https://stackoverflow.com/a/41904558/9318372

    Parameters
    ----------
    module_path: Path
        Path to the folder where the module is located
    module_name: str, optional

    Returns
    -------
    ModuleType
    """
    module_name = module_name or module_path.parts[-1]
    module_init = module_path.joinpath("__init__.py")
    assert module_init.exists(), f"Module {module_path} has no __init__ file !!!"

    with add_to_path(module_path):
        spec = spec_from_file_location(module_name, str(module_init))
        the_module = module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(the_module)  # type: ignore[union-attr]
        return the_module


class ODE_RNN(BaseModel, nn.Module):
    r"""TODO: add docstring.

    Parameters
    ----------
    batch-size: int, default 50
        Batch size
    classif_per_tp: bool, default False
    concat_mask: bool, default True
    device: torch.device, default 'cpu'
    input_dim: int
        dimensionality of input
    lr: float, default 1e-2
        Learn-rate
    nonlinear: Callable, default nn.Tanh,
        Nonlinearity used
    n_gru_units: int, default 100
        Number of units per layer in each of GRU update networks
    n_labels: int, default 1
        Number of outputs
    n_layers: int
        Number of layers in ODE func in recognition ODE
    n_ode_gru_dims: int
        Size of the latent state
    n_units: int, default 100
        Number of units per layer in ODE func
    obsrv_std: float, default 0.01
        Measurement error
    odeint_rtol: float, default 1e-3
        Relative tolerance of ODE solver
    odeint_atol: float, default 1e-4
        Absolute tolerance of ODE solver
    use_binary_classif: bool, default False
        train_classif_w_reconstr: bool, default False

    Keyword Args
    ------------
    Net_cfg: dict, default {}
        Configuration parameters for the Net
    ODEFunc_cfg: dict, default {}
        Configuration parameters for the ODEFunc
    DiffeqSolver_cfg: dict, default {}
        Configuration parameters for the DiffeqSolver
    ODE_RNN_cfg: dict, default {}
        Configuration parameters for the ODE-RNN
    """

    model_path: Path
    url: str = r"https://github.com/YuliaRubanova/latent_ode.git"

    HP: dict = {
        # Size of the latent state
        "n_ode_gru_dims": 6,
        # Number of layers in ODE func in recognition ODE
        "n_layers": 1,
        # Number of units per layer in ODE func
        "n_units": 100,
        # nonlinearity used
        "nonlinear": nn.Tanh,
        #
        "concat_mask": True,
        # dimensionality of input
        "input_dim": None,
        # device: 'cpu' or 'cuda'
        "device": torch.device("cpu"),
        # Number of units per layer in each of GRU update networks
        "n_gru_units": 100,
        # measurement error
        "obsrv_std": 0.01,
        #
        "use_binary_classif": False,
        #
        "train_classif_w_reconstr": False,
        #
        "classif_per_tp": False,
        # number of outputs
        "n_labels": 1,
        # relative tolerance of ODE solver
        "odeint_rtol": 1e-3,
        # absolute tolerance of ODE solver
        "odeint_atol": 1e-4,
        # batch_size
        "batch-size": 50,
        # learn-rate
        "lr": 1e-2,
        "ODEFunc_cfg": {},
        "DiffeqSolver_cfg": {},
        "ODE_RNN_cfg": {
            "input_dim": None,
            "latent_dim": None,
            "device": None,
        },
    }

    def __new__(cls, *args, **kwargs):
        r"""TODO: add docstring."""
        return super(ODE_RNN, cls).__new__(*args, **kwargs)

    def __init__(self, **HP):
        r"""Initialize the internal ODE-RNN model."""
        super().__init__()
        # TODO: Use tsdm.home_path or something
        module = path_import(Path.home() / ".tsdm/models/ODE-RNN")
        create_net = module.lib.utils.create_net
        ODEFunc = module.lib.ode_func.ODEFunc
        DiffeqSolver = module.lib.diffeq_solver.DiffeqSolver
        _ODE_RNN = module.lib.ode_rnn.ODE_RNN

        self.HP = HP = deep_dict_update(self.HP, HP)

        self.ode_func_net = create_net(
            n_inputs=HP["n_ode_gru_dims"],
            n_outputs=HP["n_ode_gru_dims"],
            n_layers=HP["n_layers"],
            n_units=HP["n_units"],
            nonlinear=HP["nonlinear"],
        )

        self.rec_ode_func = ODEFunc(
            ode_func_net=self.ode_func_net,
            input_dim=HP["input_dim"],
            latent_dim=HP["n_ode_gru_dims"],
            device=HP["device"],
        )

        self.z0_diffeq_solver = DiffeqSolver(
            input_dim=HP["input_dim"],
            ode_func=self.rec_ode_func,
            method="euler",
            latents=HP["n_ode_gru_dims"],
            odeint_rtol=HP["odeint_rtol"],
            odeint_atol=HP["odeint_atol"],
            device=HP["device"],
        )

        self.model = _ODE_RNN(
            input_dim=HP["input_dim"],
            latent_dim=HP["n_ode_gru_dims"],
            device=HP["device"],
            z0_diffeq_solver=self.z0_diffeq_solver,
            n_gru_units=HP["n_gru_units"],
            concat_mask=HP["concat_mask"],
            obsrv_std=HP["obsrv_std"],
            use_binary_classif=HP["use_binary_classif"],
            classif_per_tp=HP["classif_per_tp"],
            n_labels=HP["n_labels"],
            train_classif_w_reconstr=HP["train_classif_w_reconstr"],
        )

    def forward(self, T, X):
        r"""TODO: add docstring."""
        (pred,) = self.model.get_reconstruction(
            # Note: n_traj_samples and mode have no effect -> omitted!
            time_steps_to_predict=T,
            data=X,
            truth_time_steps=T,
            mask=torch.isnan(X),
        )

        return pred
