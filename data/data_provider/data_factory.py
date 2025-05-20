import importlib

from torch.utils.data import Dataset, DataLoader
from utils.ExpConfigs import ExpConfigs

def data_provider(configs: ExpConfigs, flag: str, shuffle_flag: bool = None, drop_last: bool = None) -> tuple[Dataset, DataLoader]:
    '''
    - flag: "train", "val", "test", "test_all"
    - shuffle_flag: In rare cases, it can be manually overwrite.
    - drop_last: In rare cases, it can be manually overwrite.
    '''
    # dynamically import the desired dataset class
    dataset_module = importlib.import_module(f"data.data_provider.datasets.{configs.dataset_name}")
    Data = dataset_module.Data

    # try to load custom collate_fn for the dataset, if present
    try:
        collate_fn = getattr(dataset_module, configs.collate_fn)
    except:
        collate_fn = None

    if flag in ["test", "test_all"]:
        shuffle_flag = shuffle_flag or False
        drop_last = drop_last or False
        batch_size = configs.batch_size
    else:
        shuffle_flag = shuffle_flag or True
        drop_last = drop_last or True
        batch_size = configs.batch_size

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
