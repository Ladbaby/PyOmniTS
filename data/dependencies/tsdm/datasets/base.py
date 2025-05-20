r"""Base Classes for dataset."""

from __future__ import annotations

__all__ = [
    # Classes
    "BaseDataset",
    "BaseDatasetMetaClass",
    "SingleFrameDataset",
    "MultiFrameDataset",
    # Types
    "DATASET_OBJECT",
]
import inspect
import logging
import os
import subprocess
import warnings
import webbrowser
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, MutableMapping, Sequence
from functools import cached_property, partial
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, TypeAlias, overload
from urllib.parse import urlparse

import pandas
from pandas import DataFrame, Series

from data.dependencies.tsdm.config import DATASETDIR, RAWDATADIR
from data.dependencies.tsdm.utils import flatten_nested, paths_exists, prepend_path
from data.dependencies.tsdm.utils.remote import download
from data.dependencies.tsdm.utils.types import KeyVar, Nested, PathType

DATASET_OBJECT: TypeAlias = Series | DataFrame
r"""Type hint for pandas objects."""


class BaseDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # signature: type.__init__(name, bases, attributes)
        if len(args) == 1:
            attributes = {}
        elif len(args) == 3:
            _, _, attributes = args
        else:
            raise ValueError("BaseDatasetMetaClass must be used with 1 or 3 arguments.")

        if "LOGGER" not in attributes:
            cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if "RAWDATA_DIR" not in attributes:
            if os.environ.get("GENERATING_DOCS", False):
                cls.RAWDATA_DIR = Path(f"~/.tsdm/rawdata/{cls.__name__}/")
            else:
                cls.RAWDATA_DIR = RAWDATADIR / cls.__name__

        if "DATASET_DIR" not in attributes:
            if os.environ.get("GENERATING_DOCS", False):
                cls.DATASET_DIR = Path(f"~/.tsdm/datasets/{cls.__name__}/")
            else:
                cls.DATASET_DIR = DATASETDIR / cls.__name__

        # print(f"Setting Attribute {cls}.RAWDATA_DIR = {cls.RAWDATA_DIR}")
        # print(f"{cls=}\n\n{args=}\n\n{kwargs.keys()=}\n\n")

    # def __getitem__(cls, parent: type[BaseDataset]) -> type[BaseDataset]:
    #     # if inspect.isabstract(cls):
    #     cls.RAWDATA_DIR = parent.RAWDATA_DIR
    #     print(f"Setting {cls}.RAWDATA_DIR = {parent.RAWDATA_DIR=}")
    #     return cls
    # return super().__getitem__(parent)


class BaseDataset(ABC, metaclass=BaseDatasetMetaClass):
    r"""Abstract base class that all dataset must subclass.

    Implements methods that are available for all dataset classes.
    """

    BASE_URL: ClassVar[Optional[str]] = None
    r"""HTTP address from where the dataset can be downloaded."""
    INFO_URL: ClassVar[Optional[str]] = None
    r"""HTTP address containing additional information about the dataset."""
    RAWDATA_DIR: ClassVar[Path]
    r"""Location where the raw data is stored."""
    DATASET_DIR: ClassVar[Path]
    r"""Location where the pre-processed data is stored."""
    LOGGER: ClassVar[logging.Logger]
    r"""Logger for the dataset."""

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        r"""Initialize the dataset."""
        if not inspect.isabstract(self):
            self.RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
            self.DATASET_DIR.mkdir(parents=True, exist_ok=True)

        if reset:
            self.clean()
        if initialize:
            self.load()

    def __len__(self) -> int:
        r"""Return the number of samples in the dataset."""
        return self.dataset.__len__()

    def __getitem__(self, idx):
        r"""Return the sample at index `idx`."""
        return self.dataset.__getitem__(idx)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        r"""Return the sample at index `idx`."""
        self.dataset[key] = value

    def __delitem__(self, key: Hashable, /) -> None:
        r"""Return the sample at index `idx`."""
        self.dataset.__delitem__(key)

    def __iter__(self) -> Iterator:
        r"""Return an iterator over the dataset."""
        return self.dataset.__iter__()

    def __repr__(self):
        r"""Return a string representation of the dataset."""
        return f"{self.__class__.__name__}\n{self.dataset}"

    @classmethod
    def info(cls):
        r"""Open dataset information in browser."""
        if cls.INFO_URL is None:
            print(cls.__doc__)
        else:
            webbrowser.open_new_tab(cls.INFO_URL)

    @cached_property
    @abstractmethod
    def dataset(self) -> Any | MutableMapping:
        r"""Store cached version of dataset."""
        return self.load()

    @property
    @abstractmethod
    def dataset_files(self) -> Nested[Optional[PathType]]:
        r"""Relative paths to the cleaned dataset file(s)."""

    @property
    @abstractmethod
    def rawdata_files(self) -> Nested[Optional[PathType]]:
        r"""Relative paths to the raw dataset file(s)."""

    @cached_property
    def rawdata_paths(self) -> Nested[Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.rawdata_files, parent=self.RAWDATA_DIR)

    @cached_property
    def dataset_paths(self) -> Nested[Path]:
        r"""Absolute paths to the raw dataset file(s)."""
        return prepend_path(self.dataset_files, parent=self.RAWDATA_DIR)

    def rawdata_files_exist(self) -> bool:
        r"""Check if raw data files exist."""
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self) -> bool:
        r"""Check if dataset files exist."""
        return paths_exists(self.dataset_paths)

    @abstractmethod
    def clean(self):
        r"""Clean an already downloaded raw dataset and stores it in self.data_dir.

        Preferably, use the '.feather' data format.
        """

    @abstractmethod
    def load(self):
        r"""Load the pre-processed dataset."""

    @abstractmethod
    def download(self) -> None:
        r"""Download the raw data."""

    @classmethod
    def download_from_url(cls, url: str) -> None:
        r"""Download files from a URL."""
        cls.LOGGER.info("Downloading from %s", url)
        parsed_url = urlparse(url)

        if parsed_url.netloc == "www.kaggle.com":
            kaggle_name = Path(parsed_url.path).name
            subprocess.run(
                f"kaggle competitions download -p {cls.RAWDATA_DIR} -c {kaggle_name}",
                shell=True,
                check=True,
            )
        elif parsed_url.netloc == "github.com":
            subprocess.run(
                f"svn export --force {url.replace('tree/main', 'trunk')} {cls.RAWDATA_DIR}",
                shell=True,
                check=True,
            )
        else:  # default parsing, including for UCI dataset
            fname = url.split("/")[-1]
            download(url, cls.RAWDATA_DIR / fname)


class FrameDataset(BaseDataset, ABC):
    r"""Base class for datasets that are stored as pandas.DataFrame."""

    DEFAULT_FILE_FORMAT: str = "parquet"
    r"""Default format for the dataset."""

    @staticmethod
    def serialize(frame: DATASET_OBJECT, path: Path, /, **kwargs: Any) -> None:
        r"""Serialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]

        if isinstance(frame, Series):
            frame = frame.to_frame()

        if hasattr(frame, f"to_{file_type}"):
            pandas_writer = getattr(frame, f"to_{file_type}")
            pandas_writer(path, **kwargs)
            return

        raise NotImplementedError(f"No loader for {file_type=}")

    @staticmethod
    def deserialize(path: Path, /, *, squeeze: bool = True) -> DATASET_OBJECT:
        r"""Deserialize the dataset."""
        file_type = path.suffix
        assert file_type.startswith("."), "File must have a suffix!"
        file_type = file_type[1:]

        if hasattr(pandas, f"read_{file_type}"):
            pandas_loader = getattr(pandas, f"read_{file_type}")
            pandas_object = pandas_loader(path)
            return pandas_object.squeeze() if squeeze else pandas_object

        raise NotImplementedError(f"No loader for {file_type=}")

    def validate(
        self,
        filespec: Nested[str | Path],
        /,
        *,
        reference: Optional[str | Mapping[str, str]] = None,
    ) -> None:
        r"""Validate the file hash."""
        self.LOGGER.debug("Starting to validate dataset")

        if isinstance(filespec, Mapping):
            for value in filespec.values():
                self.validate(value, reference=reference)
            return
        if isinstance(filespec, Sequence) and not isinstance(filespec, (str, Path)):
            for value in filespec:
                self.validate(value, reference=reference)
            return

        assert isinstance(filespec, (str, Path)), f"{filespec=} wrong type!"
        file = Path(filespec)

        if not file.exists():
            raise FileNotFoundError(f"File '{file.name}' does not exist!")

        filehash = sha256(file.read_bytes()).hexdigest()

        if reference is None:
            warnings.warn(
                f"File '{file.name}' cannot be validated as no hash is stored in {self.__class__}."
                f"The filehash is '{filehash}'."
            )

        elif isinstance(reference, str):
            if filehash != reference:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            self.LOGGER.info(
                f"File '{file.name}' validated successfully '{filehash=}'."
            )

        elif isinstance(reference, Mapping):
            if not (file.name in reference) ^ (file.stem in reference):
                warnings.warn(
                    f"File '{file.name}' cannot be validated as it is not contained in {reference}."
                    f"The filehash is '{filehash}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.name in reference and filehash != reference[file.name]:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference[file.name]}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            elif file.stem in reference and filehash != reference[file.stem]:
                warnings.warn(
                    f"File '{file.name}' failed to validate!"
                    f"File hash '{filehash}' does not match reference '{reference[file.stem]}'."
                    f"𝗜𝗴𝗻𝗼𝗿𝗲 𝘁𝗵𝗶𝘀 𝘄𝗮𝗿𝗻𝗶𝗻𝗴 𝗶𝗳 𝘁𝗵𝗲 𝗳𝗶𝗹𝗲 𝗳𝗼𝗿𝗺𝗮𝘁 𝗶𝘀 𝗽𝗮𝗿𝗾𝘂𝗲𝘁."
                )
            else:
                self.LOGGER.info(
                    f"File '{file.name}' validated successfully '{filehash=}'."
                )
        else:
            raise TypeError(f"Unsupported type for {reference=}.")

        self.LOGGER.debug("Finished validating file.")


class SingleFrameDataset(FrameDataset):
    r"""Dataset class that consists of a singular DataFrame."""

    RAWDATA_SHA256: Optional[str | Mapping[str, str]] = None
    r"""SHA256 hash value of the raw data file(s)."""
    RAWDATA_SHAPE: Optional[tuple[int, ...] | Mapping[str, tuple[int, ...]]] = None
    r"""Reference shape of the raw data file(s)."""
    DATASET_SHA256: Optional[str] = None
    r"""SHA256 hash value of the dataset file(s)."""
    DATASET_SHAPE: Optional[tuple[int, ...]] = None
    r"""Reference shape of the dataset file(s)."""

    def _repr_html_(self):
        if hasattr(self.dataset, "_repr_html_"):
            header = f"<h3>{self.__class__.__name__}</h3>"
            # noinspection PyProtectedMember
            html_repr = self.dataset._repr_html_()  # pylint: disable=protected-access
            return header + html_repr
        raise NotImplementedError

    @cached_property
    def dataset(self) -> DATASET_OBJECT:
        r"""Store cached version of dataset."""
        return self.load()

    @cached_property
    def dataset_files(self) -> PathType:
        r"""Return the dataset files."""
        return self.__class__.__name__ + f".{self.DEFAULT_FILE_FORMAT}"

    @cached_property
    def dataset_paths(self) -> Path:
        r"""Path to raw data."""
        return self.DATASET_DIR / (self.dataset_files or "")

    @abstractmethod
    def _clean(self) -> DATASET_OBJECT | None:
        r"""Clean the dataset."""

    def _load(self) -> DATASET_OBJECT:
        r"""Load the dataset."""
        return self.deserialize(self.dataset_paths)

    def _download(self) -> None:
        r"""Download the dataset."""
        assert self.BASE_URL is not None, "base_url is not set!"

        nested_files: Nested[Path] = prepend_path(
            self.rawdata_files, Path(), keep_none=False
        )
        files: set[Path] = flatten_nested(nested_files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

    def load(self, *, force: bool = True, validate: bool = True) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        if not self.dataset_files_exist():
            self.clean(force=force, validate=validate)
        else:
            self.LOGGER.debug("Dataset files already exist!")

        if validate:
            self.validate(self.dataset_paths, reference=self.DATASET_SHA256)

        self.LOGGER.debug("Starting to load dataset.")
        ds = self._load()
        self.LOGGER.debug("Finished loading dataset.")
        return ds

    def clean(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Clean the selected DATASET_OBJECT."""
        if self.dataset_files_exist() and not force:
            self.LOGGER.debug("Dataset files already exist, skipping.")
            return

        if not self.rawdata_files_exist():
            self.download(force=force, validate=validate)

        if validate:
            self.validate(self.rawdata_paths, reference=self.RAWDATA_SHA256)

        self.LOGGER.debug("Starting to clean dataset.")
        df = self._clean()
        if df is not None:
            self.LOGGER.info("Serializing dataset.")
            self.serialize(df, self.dataset_paths)
        self.LOGGER.debug("Finished cleaning dataset.")

        if validate:
            self.validate(self.dataset_paths, reference=self.DATASET_SHA256)

    def download(self, *, force: bool = True, validate: bool = True) -> None:
        r"""Download the dataset."""
        if self.rawdata_files_exist() and not force:
            self.LOGGER.info("Dataset already exists. Skipping download.")
            return

        if self.BASE_URL is None:
            self.LOGGER.info("Dataset provides no url. Assumed offline")
            return

        self.LOGGER.debug("Starting to download dataset.")
        self._download()
        self.LOGGER.debug("Starting downloading dataset.")

        if validate:
            self.validate(self.rawdata_paths, reference=self.RAWDATA_SHA256)


class MultiFrameDataset(FrameDataset, Mapping, Generic[KeyVar]):
    r"""Dataset class that consists of a multiple DataFrames.

    The Datasets are accessed by their index.
    We subclass `Mapping` to provide the mapping interface.
    """

    RAWDATA_SHA256: Optional[str | Mapping[str, str]] = None
    r"""SHA256 hash value of the raw data file(s)."""
    RAWDATA_SHAPE: Optional[tuple[int, ...] | Mapping[str, tuple[int, ...]]] = None
    r"""Reference shape of the raw data file(s)."""
    DATASET_SHA256: Optional[Mapping[str, str]] = None
    r"""SHA256 hash value of the dataset file(s)."""
    DATASET_SHAPE: Optional[Mapping[str, tuple[int, ...]]] = None
    r"""Reference shape of the dataset file(s)."""

    def __init__(self, *, initialize: bool = True, reset: bool = False):
        r"""Initialize the Dataset."""
        self.LOGGER.info("Adding keys as attributes.")
        if initialize:
            for key in self.index:
                if isinstance(key, str) and not hasattr(self, key):
                    _get_dataset = partial(self.__class__.load, key=key)
                    _get_dataset.__doc__ = f"Load dataset for {key=}."
                    setattr(self.__class__, key, property(_get_dataset))

        super().__init__(initialize=initialize, reset=reset)

    def __repr__(self):
        r"""Pretty Print."""
        if len(self.index) > 6:
            indices = list(self.index)
            selection = [str(indices[k]) for k in [0, 1, 2, -2, -1]]
            selection[2] = "..."
            index_str = ", ".join(selection)
        else:
            index_str = repr(self.index)
        return f"{self.__class__.__name__}{index_str}"

    @property
    @abstractmethod
    def index(self) -> Sequence[KeyVar]:
        r"""Return the index of the dataset."""

    @cached_property
    def dataset(self) -> MutableMapping[KeyVar, DATASET_OBJECT]:
        r"""Store cached version of dataset."""
        return {key: None for key in self.index}

    @cached_property
    def dataset_files(self) -> Mapping[KeyVar, str]:
        r"""Relative paths to the dataset files for each key."""
        return {key: f"{key}.{self.DEFAULT_FILE_FORMAT}" for key in self.index}

    @cached_property
    def dataset_paths(self) -> Mapping[KeyVar, Path]:
        r"""Absolute paths to the dataset files for each key."""
        return {
            key: self.DATASET_DIR / file for key, file in self.dataset_files.items()
        }

    @abstractmethod
    def _clean(self, key: KeyVar) -> DATASET_OBJECT | None:
        r"""Clean the selected DATASET_OBJECT."""

    def _load(self, key: KeyVar) -> DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT."""
        return self.deserialize(self.dataset_paths[key])

    def rawdata_files_exist(self, key: Optional[KeyVar] = None) -> bool:
        r"""Check if raw data files exist."""
        if key is None:
            return paths_exists(self.rawdata_paths)
        if isinstance(self.rawdata_paths, Mapping):
            return paths_exists(self.rawdata_paths[key])
        return paths_exists(self.rawdata_paths)

    def dataset_files_exist(self, key: Optional[KeyVar] = None) -> bool:
        r"""Check if dataset files exist."""
        if key is None:
            return paths_exists(self.dataset_paths)
        return paths_exists(self.dataset_paths[key])

    def clean(
        self,
        key: Optional[KeyVar] = None,
        *,
        force: bool = False,
        validate: bool = True,
    ) -> None:
        r"""Clean the selected DATASET_OBJECT.

        Parameters
        ----------
        key: Optional[KeyType] = None
            The key of the dataset to clean. If None, clean all dataset.
        force: bool = False
            Force cleaning of dataset.
        validate: bool = True
            Validate the dataset after cleaning.
        """
        # TODO: Do we need this code block?
        if not self.rawdata_files_exist(key=key):
            self.LOGGER.debug("Raw files missing, fetching it now! <%s>", key)
            self.download(key=key, force=force, validate=validate)

        if (
            key in self.dataset_files
            and self.dataset_files_exist(key=key)
            and not force
        ):
            self.LOGGER.debug("Clean files already exists, skipping <%s>", key)
            return

        if key is None:
            if validate:
                self.validate(self.rawdata_paths, reference=self.RAWDATA_SHA256)

            self.LOGGER.debug("Starting to clean dataset.")
            for key_ in self.index:
                self.clean(key=key_, force=force, validate=validate)
            self.LOGGER.debug("Finished cleaning dataset.")

            if validate:
                self.validate(self.dataset_paths, reference=self.DATASET_SHA256)
            return

        self.LOGGER.debug("Starting to clean dataset <%s>", key)
        df = self._clean(key=key)
        if df is not None:
            self.LOGGER.info("Serializing dataset <%s>", key)
            self.serialize(df, self.dataset_paths[key])
        self.LOGGER.debug("Finished cleaning dataset <%s>", key)

    @overload
    def load(
        self,
        *,
        key: None = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[KeyVar, Any]:
        ...

    @overload
    def load(
        self,
        *,
        key: KeyVar = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Any:
        ...

    def load(
        self,
        *,
        key: Optional[KeyVar] = None,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> Mapping[KeyVar, DATASET_OBJECT] | DATASET_OBJECT:
        r"""Load the selected DATASET_OBJECT.

        Parameters
        ----------
        key: Optional[KeyType] = None
        force: bool = False
            Reload the dataset if it already exists.
        validate: bool = True
            Validate the dataset file hash.

        Returns
        -------
        DATASET_OBJECT | Mapping[KeyType, DATASET_OBJECT]
        """
        if not self.dataset_files_exist(key=key):
            self.clean(key=key, force=force)

        if key is None:
            # Download full dataset
            self.LOGGER.debug("Starting to load  dataset.")
            ds = {
                k: self.load(key=k, force=force, validate=validate, **kwargs)
                for k in self.index
            }
            self.LOGGER.debug("Finished loading  dataset.")
            return ds

        # download specific key
        if key in self.dataset and self.dataset[key] is not None and not force:
            self.LOGGER.debug("Dataset already exists, skipping! <%s>", key)
            return self.dataset[key]

        if validate:
            self.validate(self.dataset_paths[key], reference=self.DATASET_SHA256)

        self.LOGGER.debug("Starting to load  dataset <%s>", key)
        self.dataset[key] = self._load(key=key)
        self.LOGGER.debug("Finished loading  dataset <%s>", key)
        return self.dataset[key]

    def _download(self, key: KeyVar = None) -> None:
        r"""Download the selected DATASET_OBJECT."""
        assert self.BASE_URL is not None, "base_url is not set!"

        rawdata_files: Nested[Optional[PathType]]
        if isinstance(self.rawdata_files, Mapping):
            rawdata_files = self.rawdata_files[key]
        else:
            rawdata_files = self.rawdata_files

        nested_files: Nested[Path] = prepend_path(
            rawdata_files, Path(), keep_none=False
        )
        files: set[Path] = flatten_nested(nested_files, kind=Path)

        for file in files:
            self.download_from_url(self.BASE_URL + file.name)

    def download(
        self,
        key: Optional[KeyVar] = None,
        *,
        force: bool = False,
        validate: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Download the dataset.

        Parameters
        ----------
        key: Optional[KeyType] = None
        validate: bool = True
            Validate the downloaded files.
        force: bool = False
            Force re-downloading of dataset.
        """
        if self.BASE_URL is None:
            self.LOGGER.debug("Dataset provides no base_url. Assumed offline")
            return

        if self.rawdata_files is None:
            self.LOGGER.debug("Dataset needs no raw data files. Skipping.")
            return

        if not force and self.rawdata_files_exist(key=key):
            self.LOGGER.debug("Rawdata files already exist, skipping. <%s>", str(key))
            return

        if key is None:
            # Download full dataset
            self.LOGGER.debug("Starting to download dataset.")
            if isinstance(self.rawdata_files, Mapping):
                for key_ in self.rawdata_files:
                    self.download(key=key_, force=force, validate=validate, **kwargs)
            else:
                self._download()
            self.LOGGER.debug("Finished downloading dataset.")

            if validate:
                self.validate(self.rawdata_paths, reference=self.RAWDATA_SHA256)

            return

        # Download specific key
        self.LOGGER.debug("Starting to download dataset <%s>", key)
        self._download(key=key)
        self.LOGGER.debug("Finished downloading dataset <%s>", key)
