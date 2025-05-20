# TOTOs

## Multi-Datasets

A single dataset can have multiple processed versions originating from the same raw data.

- For example, a dataset can have a version with and without the `NaN` values removed.
- A dataset might be available in multiple versions (e.g. MIMIC-IV v1.0 and v2.0).

We want to allow subclasses of a `BaseDataset`, which work as follows:

- They share the same `RAWDATADIR`.
  - This stores the raw data, potentially multiple versions like `mimic-iv-v1.0.zip` and `mimic-iv-v2.0.zip`
  - The `DATSETDIR` contains a base folder for the dataset, e.g. `MIMIC-IV` with potentially multiple subfolders.
- The `BaseDataset` class provides
  - Downloading functionality.
  - Light preprocessing, in particular conversion of the data to binary parquet format with appropriate column types.
  - "Subclasses" can use this data to build more complex preprocessing pipelines.

Questions:

- Do "subclasses" really make sense here?
  - It seems sufficient that the dataset would instantiate the "base class".
- The download functionality etc. should be maintained.

Implementation:

Create 3 subclass of BaseDataset:

1. `MIMIC_IV` the base_class
2. A `DerivedDataset` abstract base class that is parametrized by a `BaseDataset`
3. The subclass / instance of the `DerivedDataset` class.

```python

class MIMIC_IV(BaseDataset):
    ...

class DerivedDataset(ABCMeta):
    ...

class MIMIC_IV_DeBrouwer(DerivedDataset[MIMIC_IV["1.0"]]):   # <-- too magic for type checkers / linters
  RAWDATADIR = MIMIC_IV.RAWDATADIR
  DATASETDIR = MIMIC_IV.DATASETDIR / MIMIC_IV_DeBrouwer
  RAWDATAFILES = MIMIC_IV.RAWDATAFILES
  RAWDATAPATHS = MIMIC_IV.RAWDATAPATHS
  download = MIMIC_IV.download


@extends(MIMIC_IV)
class MIMIC_IV_DeBrouwer(FrameDataset, ...):
  ...
```

Question: How to combine with Versioning? We want to allow multiple versions of the same dataset. Options:

1. MIMIC_IV @ "1.0"
2. MIMIC_IV["1.0"] => returns a class `"MIMIC-IV@1.0"` that can then be instantiated.
3. MIMIC_IV(version="1.0")

Should different versions be different classes, or different instances of the same class?
At least in the case of MIMIC-IV, the data, e.g. the columns in the table may change between versions.
Thus, it seems versions should be different classes.

=> Create `MetaClass` called `VersionedDataset`?

```python
class DerivedDatasetMetaClass(ABCMeta):
    r"""Metaclass for BaseDataset."""

    def __init__(cls, *args, **kwargs):
        cls.LOGGER = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if os.environ.get("GENERATING_DOCS", False):
            cls.RAWDATA_DIR = Path(f"~/.tsdm/rawdata/{cls.__name__}/")
            cls.DATASET_DIR = Path(f"~/.tsdm/datasets/{cls.__name__}/")
        else:
            cls.RAWDATA_DIR = RAWDATADIR / cls.__name__
            cls.DATASET_DIR = DATASETDIR / cls.__name__

        super().__init__(*args, **kwargs)

    def __getitem__(cls, klass: type[BaseDataset]) -> type[BaseDataset]:
        r"""Get the dataset class."""
        return klass
```
