r"""Configuration Options.

Content:
  - config.yaml
  - dataset.yaml
  - models.yaml
  - hashes.yaml
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

from data.dependencies.tsdm.config._config import (
    BASEDIR,
    CONFIG,
    DATASETDIR,
    DATASETS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    HASHES,
    HOMEDIR,
    LOGDIR,
    MODELDIR,
    MODELS,
    RAWDATADIR,
    Config,
    conf,
)
