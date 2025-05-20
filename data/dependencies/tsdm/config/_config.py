r"""TSDM Configuration.

# TODO: There must be a better way to handle global config
"""

__all__ = [
    # CONSTANTS
    "CONFIG",
    "DATASETS",
    "MODELS",
    "HASHES",
    "HOMEDIR",
    "BASEDIR",
    "LOGDIR",
    "MODELDIR",
    "DATASETDIR",
    "RAWDATADIR",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "conf",
    # Classes
    "Config",
]

import logging
import os
from importlib import resources
from pathlib import Path

import torch
import yaml

from data.dependencies.tsdm.config import config_files

__logger__ = logging.getLogger(__name__)

os.environ["TSDM_AUTOJIT"] = "True"
r"""Default value."""

NAME = __name__
r"""Name of the module."""
FILE = __file__
r"""Path to the current file."""


class Config:
    r"""Configuration Interface."""

    # TODO: Should be initialized by a init/toml file.
    _autojit: bool = True
    __name__ = NAME
    __file__ = FILE

    @property
    def autojit(self) -> bool:
        r"""Whether to automatically jit-compile the models."""
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ["TSDM_AUTOJIT"] = str(value)


conf: Config = Config()  # = Config(__name__, __doc__)
r"""The unique `~tsdm.config.Config` instance used to configure `tsdm`."""

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r"""The default `torch` device to use."""
DEFAULT_DTYPE = torch.float32
r"""The default `torch` datatype to use."""


with resources.open_text(config_files, "config.yaml") as file:
    # with open(file, "r", encoding="utf8") as fname:
    CONFIG = yaml.safe_load(file)
    r"""Dictionary containing basic configuration of TSDM."""

with resources.open_text(config_files, "models.yaml") as file:
    MODELS = yaml.safe_load(file)
    r"""Dictionary containing sources of the available models."""

with resources.open_text(config_files, "datasets.yaml") as file:
    DATASETS = yaml.safe_load(file)
    r"""Dictionary containing sources of the available dataset."""

with resources.open_text(config_files, "hashes.yaml") as file:
    HASHES = yaml.safe_load(file)
    r"""Dictionary containing hash values for both models and dataset."""

HOMEDIR = Path.home()
r"""The users home directory."""

BASEDIR = HOMEDIR.joinpath(CONFIG["basedir"])
r"""Root directory for tsdm storage."""

LOGDIR = BASEDIR.joinpath(CONFIG["logdir"])
r"""Path where logfiles are stored."""

MODELDIR = BASEDIR.joinpath(CONFIG["modeldir"])
r"""Path where imported models are stored."""

DATASETDIR = BASEDIR.joinpath(CONFIG["datasetdir"])
r"""Path where preprocessed dataset are stored."""

RAWDATADIR = BASEDIR.joinpath(CONFIG["rawdatadir"])
r"""Path where raw imported dataset are stored."""

LOGDIR.mkdir(parents=True, exist_ok=True)
# logging.basicConfig(
#     filename=str(LOGDIR.joinpath("example.log")),
#     filemode="w",
#     format="[%(asctime)s] [%(levelname)-s]\t[%(name)s]\t%(message)s, (%(filename)s:%(lineno)s)",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=logging.INFO)

__logger__.info("Available Models: %s", set(MODELS))
__logger__.info("Available Datasets: %s", set(DATASETS))


def generate_folders(d: dict, current_path: Path) -> None:
    r"""Create nested folder structure based on nested dictionary index.

    Parameters
    ----------
    current_path: Path
    d: dict

    Returns
    -------
    None

    References
    ----------
    `StackOverflow <https://stackoverflow.com/a/22058144/9318372>`_
    """
    for directory in d:
        path = current_path.joinpath(directory)
        if d[directory] is None:
            __logger__.debug("creating folder %s", path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[directory], path)


__logger__.debug("Initializing folder structure")
generate_folders(CONFIG["folders"], BASEDIR)
__logger__.debug("Created folder structure in %s", BASEDIR)
