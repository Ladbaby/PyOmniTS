from typing import Union, List

import pandas as pd
from layers.tsl.datasets import PvUS as _PvUS


class PvUS(_PvUS):
    # Courtesy of Stefano Imoscopi
    #
    # UTC shifts for USA timezones (Standard, non DST)
    # see https://en.wikipedia.org/wiki/Time_in_the_United_States
    tz_mapper = {
        'Alabama': 'Etc/GMT+6',  # 'UTC-06:00',
        'Arkansas': 'Etc/GMT+6',  # 'UTC-06:00',
        'Connecticut': 'Etc/GMT+5',  # 'UTC-05:00',
        'Delaware': 'Etc/GMT+5',  # 'UTC-05:00',
        'Florida': 'Etc/GMT+5',  # 'UTC−05:00',
        'Georgia': 'Etc/GMT+5',  # 'UTC−05:00',
        'Illinois': 'Etc/GMT+6',  # 'UTC−06:00',
        'Indiana': 'Etc/GMT+6',  # 'UTC−06:00',
        'Iowa': 'Etc/GMT+6',  # 'UTC−06:00',
        'Kansas': 'Etc/GMT+6',  # 'UTC−06:00',
        'Kentucky': 'Etc/GMT+6',  # 'UTC−06:00',
        'Louisiana': 'Etc/GMT+6',  # 'UTC−06:00',
        'Maine': 'Etc/GMT+5',  # 'UTC−05:00',
        'Maryland': 'Etc/GMT+5',  # 'UTC−05:00',
        'Massachusetts': 'Etc/GMT+5',  # 'UTC−05:00',
        'Michigan': 'Etc/GMT+5',  # 'UTC−05:00',
        'Minnesota': 'Etc/GMT+6',  # 'UTC−06:00',
        'Mississippi': 'Etc/GMT+6',  # 'UTC−06:00',
        'Missouri': 'Etc/GMT+6',  # 'UTC−06:00',
        'Montana East': 'Etc/GMT+7',  # 'UTC−07:00',
        'Nebraska': 'Etc/GMT+7',  # 'UTC−07:00',
        'New Hampshire': 'Etc/GMT+5',  # 'UTC−05:00',
        'New Jersey': 'Etc/GMT+5',  # 'UTC−05:00',
        'New Mexico East': 'Etc/GMT+7',  # 'UTC−07:00',
        'New York': 'Etc/GMT+5',  # 'UTC−05:00',
        'North Carolina': 'Etc/GMT+5',  # 'UTC−05:00',
        'Ohio': 'Etc/GMT+5',  # 'UTC−05:00',
        'Oklahoma': 'Etc/GMT+6',  # 'UTC−06:00',
        'Pennsylvania': 'Etc/GMT+5',  # 'UTC−05:00',
        'Rhode Island': 'Etc/GMT+5',  # 'UTC−05:00',
        'South Carolina': 'Etc/GMT+5',  # 'UTC−05:00',
        'South Dakota East': 'Etc/GMT+6',  # 'UTC−06:00',
        'Tennessee': 'Etc/GMT+5',  # 'UTC−05:00',
        'Texas East': 'Etc/GMT+6',  # 'UTC−06:00',
        'Vermont': 'Etc/GMT+5',  # 'UTC−05:00',
        'Virginia': 'Etc/GMT+5',  # 'UTC−05:00',
        'West Virginia': 'Etc/GMT+5',  # 'UTC−05:00',
        'Wisconsin': 'Etc/GMT+6',  # 'UTC−06:00',
        'Arizona': 'Etc/GMT+7',  # 'UTC−07:00',
        'California': 'Etc/GMT+8',  # 'UTC−08:00',
        'Colorado': 'Etc/GMT+7',  # 'UTC−07:00',
        'Idaho': 'Etc/GMT+7',  # 'UTC−07:00',
        'Montana': 'Etc/GMT+7',  # 'UTC−07:00',
        'Nevada': 'Etc/GMT+7',  # 'UTC−07:00',
        'New Mexico': 'Etc/GMT+7',  # 'UTC−07:00',
        'Oregon': 'Etc/GMT+7',  # 'UTC−07:00',
        'South Dakota': 'Etc/GMT+7',  # 'UTC−07:00',
        'Texas': 'Etc/GMT+7',  # 'UTC−07:00',
        'Utah': 'Etc/GMT+7',  # 'UTC−07:00',
        'Washington': 'Etc/GMT+8',  # 'UTC−08:00',
        'Wyoming': 'Etc/GMT+7'  # 'UTC−07:00'
    }

    def __init__(self,
                 zones: Union[str, List] = None,
                 mask_zeros: bool = False,
                 convert_to_timezone: Union[str, None] = 'infer',
                 root: str = None,
                 freq: str = None):
        self.convert_to_timezone = convert_to_timezone
        super().__init__(zones=zones,
                         mask_zeros=mask_zeros,
                         root=root,
                         freq=freq)

    def load_raw(self):
        actual, metadata = super().load_raw()
        if self.convert_to_timezone is None:
            return actual, metadata

        tz_df = pd.DataFrame.from_dict(self.tz_mapper, orient='index',
                                       columns=['tz'])
        tz_nodes = tz_df.loc[metadata.loc[:, 'state']]

        if self.convert_to_timezone == 'infer':
            # infer timezone from states
            target_tz = tz_nodes.mode().values[0, 0]
        else:
            target_tz = self.convert_to_timezone

        # localize tz and convert to UTC, so that
        # farms in different tz can be aligned together
        tzs = tz_nodes['tz'].unique()
        dfs = []
        for tz in tzs:
            nodes = metadata.index[tz_nodes.values.ravel() == tz]
            df_tz = actual.loc[:, nodes]
            df_tz = df_tz.tz_localize(tz, ambiguous='raise')
            dfs.append(df_tz.tz_convert(target_tz))
        # Concatenate and fill missing values (due to TZ, all night hours)
        actual = pd.concat(dfs, axis=1).fillna(0.)
        return actual, metadata
