import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from data.dependencies.MTS_Dataset.utils.timefeatures import time_features
from data.dependencies.MTS_Dataset.utils.augmentation import run_augmentation_single

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Data(Dataset):
    def __init__(
        self, 
        configs: ExpConfigs,
        flag='train', 
        **kwargs
    ):
        self.configs = configs
        self.dataset = Dataset_Custom(
            configs=configs,
            root_path=configs.dataset_root_path,
            flag=flag,
            size=(configs.seq_len, configs.label_len, configs.pred_len),
            features=configs.features,
            data_path=configs.dataset_file_name,
            target="OT" if configs.target_variable_name is None else configs.target_variable_name,
            scale=True, 
            timeenc=0 if configs.embed != "timeF" else 1, 
            freq=configs.freq
        )

    def __getitem__(self, index):
        # Get an item from the underlying dataset
        return self.dataset[index]

    def __len__(self):
        # Return the length of the underlying dataset
        return len(self.dataset)

class Dataset_Custom(Dataset):
    def __init__(
        self, 
        configs: ExpConfigs, 
        root_path, 
        flag='train', 
        size=None,
        features='S', 
        data_path='ETTh1.csv',
        target='OT', 
        scale=True, 
        timeenc=0, 
        freq='h'
    ):
        # size [seq_len, label_len, pred_len]
        self.configs = configs
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'test_all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'test_all': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0, 
            num_train - self.seq_len, 
            len(df_raw) - num_test - self.seq_len,
            0
        ]
        border2s = [
            num_train, 
            num_train + num_vali, 
            len(df_raw),
            len(df_raw)
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.configs.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.configs)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return {
            'x': seq_x.astype(np.float32),
            'y': seq_y.astype(np.float32),
            "x_mark": seq_x_mark.astype(np.float32),
            "y_mark": seq_y_mark.astype(np.float32)
        }
        # return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)