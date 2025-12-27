# Code from: https://github.com/Ladbaby/PyOmniTS
import importlib

from torch.utils.data import DataLoader, Dataset

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


def data_provider(configs: ExpConfigs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:
    '''
    - flag: "train", "val", "test", "test_all"
    - shuffle_flag (DEPRECATED): In rare cases, it can be manually overwrite.
    - drop_last (DEPRECATED): In rare cases, it can be manually overwrite.
    '''
    # backward compatibility
    assert not (shuffle_flag or drop_last), "Please use --train_val_loader_shuffle 0/1 and --train_val_loader_drop_last 0/1 to set shuffle_flag and drop_last for train/val dataloader instead."
    # dynamically import the desired dataset class
    dataset_module = importlib.import_module(f"data.data_provider.datasets.{configs.dataset_name}")
    Data = dataset_module.Data

    # try to load custom collate_fn for the dataset, if present
    collate_fn = None
    if configs.collate_fn is not None:
        try:
            collate_fn = getattr(dataset_module, configs.collate_fn)
        except:
            logger.warning(f"--collate_fn {configs.collate_fn} not implemented for dataset {configs.dataset_name}. Fall back to None.")
            logger.warning(f"Hint: You can implement the {configs.collate_fn}() function in data/data_provider/datasets/{configs.dataset_name}.py")

    if flag in ["test", "test_all"]:
        shuffle_flag = False
        drop_last = False
        batch_size = configs.batch_size
    else:
        shuffle_flag = bool(configs.train_val_loader_shuffle)
        drop_last = bool(configs.train_val_loader_drop_last)
        batch_size = configs.batch_size

    logger.debug(f"Getting {flag} set of {configs.dataset_name}")
    data_set: Dataset = Data(
        configs=configs,
        flag=flag,
        # DEBUG: temporal change
        # **configs._asdict()
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=configs.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return data_set, data_loader
