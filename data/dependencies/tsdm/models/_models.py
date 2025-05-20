r"""Base Model that all other models must subclass."""

__all__ = [
    # Classes
    "BaseModel",
]

import logging
import os
import subprocess
from abc import ABC, ABCMeta, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Optional, Protocol
from urllib.parse import urlparse

from data.dependencies.tsdm.config import MODELDIR


class BaseModelMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if os.environ.get("GENERATING_DOCS", False):
            cls.MODEL_DIR = Path(f"~/.tsdm/models/{cls.__name__}/")
        else:
            cls.MODEL_DIR = MODELDIR / cls.__name__

        super().__init__(*args, **kwargs)


class BaseModel(ABC):
    r"""BaseModel that all models should subclass."""

    LOGGER: logging.Logger
    r"""Logger for the model."""
    SOURCE_URL: Optional[str] = None
    r"""HTTP address from where the model can be downloaded."""
    INFO_URL: Optional[str] = None
    r"""HTTP address containing additional information about the dataset."""
    MODEL_DIR: Path
    r"""Location where the model is stored."""

    @cached_property
    def model_path(self) -> Path:
        r"""Return the path to the model."""
        return MODELDIR / self.__class__.__name__

    def download(self, *, url: Optional[str | Path] = None) -> None:
        r"""Download model (e.g. via git clone)."""
        target_url: str = str(self.SOURCE_URL) if url is None else str(url)
        parsed_url = urlparse(target_url)

        self.LOGGER.info("Obtaining model from %s", self.SOURCE_URL)

        if parsed_url.netloc == "github.com":

            if "tree/main" in target_url:
                export_url = target_url.replace("tree/main", "trunk")
            elif "tree/master" in target_url:
                export_url = target_url.replace("tree/master", "trunk")
            else:
                raise ValueError(f"Unrecognized URL: {target_url}")

            subprocess.run(
                f"svn export --force {export_url} {self.model_path}",
                shell=True,
                check=True,
            )
        elif "google-research" in parsed_url.path:
            subprocess.run(
                f"svn export {self.SOURCE_URL} {self.model_path}",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"grep -qxF '{self.model_path}' .gitignore || echo '{self.model_path}' >> .gitignore",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"git clone {self.SOURCE_URL} {self.model_path}", shell=True, check=True
            )
            # subprocess.run(F"git -C {model_path} pull", shell=True)

        self.LOGGER.info("Finished importing model from %s", self.SOURCE_URL)

    @abstractmethod
    def forward(self, *args):
        r"""Synonym for forward and __call__."""


class RemoteModelProtocol(Protocol):
    r"""Thin wrapper for models that are provided by an external GIT repository."""


class RemotePreTrainedModel(Protocol):
    r"""Thin wrapper for pretrained models that are provided by an external GIT repository."""


class PreTrainedModel(ABC):
    r"""Base class for all pretrained models."""

    @classmethod
    @abstractmethod
    def from_hyperparameters(cls) -> None:
        r"""Create a model from hyperparameters."""
        raise NotImplementedError

    @abstractmethod
    def download(self) -> None:
        r"""Download the model."""

    @abstractmethod
    def forward(self):
        r"""Give the model output given encoded data."""

    def predict(self):
        r"""Wrap the forward with encode and decode."""
