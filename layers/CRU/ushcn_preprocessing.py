# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from GRU-ODE-Bayes (https://github.com/edebrouwer/gru_ode_bayes)
# Copyright (c) 2022 Edward De Brouwer
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from concurrent.futures import process
import pandas as pd
from itertools import chain
import numpy as np
import os
import gzip
from torchvision.datasets.utils import download_url

# new code component
def download_ushcn(data_path):
    os.makedirs(data_path, exist_ok=True)
    url = 'https://cdiac.ess-dive.lbl.gov/ftp/ushcn_daily/'
    daily_state_files = [
            'state01_AL.txt.gz',	 
            'state02_AZ.txt.gz',	 
            'state03_AR.txt.gz',	 
            'state04_CA.txt.gz',	 
            'state05_CO.txt.gz',	 
            'state06_CT.txt.gz',	 
            'state07_DE.txt.gz',	 
            'state08_FL.txt.gz',	 
            'state09_GA.txt.gz',	 
            'state10_ID.txt.gz',	 
            'state11_IL.txt.gz',	 
            'state12_IN.txt.gz',	 
            'state13_IA.txt.gz',	 
            'state14_KS.txt.gz',	 
            'state15_KY.txt.gz',	 
            'state16_LA.txt.gz',	 
            'state17_ME.txt.gz',	 
            'state18_MD.txt.gz',	 
            'state19_MA.txt.gz',	 
            'state20_MI.txt.gz',	 
            'state21_MN.txt.gz',	 
            'state22_MS.txt.gz',	 
            'state23_MO.txt.gz',	 
            'state24_MT.txt.gz',	 
            'state25_NE.txt.gz',	 
            'state26_NV.txt.gz',	 
            'state27_NH.txt.gz',	 
            'state28_NJ.txt.gz',	 
            'state29_NM.txt.gz',	 
            'state30_NY.txt.gz',	 
            'state31_NC.txt.gz',	 
            'state32_ND.txt.gz',	 
            'state33_OH.txt.gz',	 
            'state34_OK.txt.gz',	 
            'state35_OR.txt.gz',	 
            'state36_PA.txt.gz',	 
            'state37_RI.txt.gz',	 
            'state38_SC.txt.gz',	 
            'state39_SD.txt.gz',	 
            'state40_TN.txt.gz',	 
            'state41_TX.txt.gz',	 
            'state42_UT.txt.gz',	 
            'state43_VT.txt.gz',	 
            'state44_VA.txt.gz',	 
            'state45_WA.txt.gz',	 
            'state46_WV.txt.gz',	 
            'state47_WI.txt.gz',	 
            'state48_WY.txt.gz'
            ]
    for state_file in daily_state_files:
        state_url = url + state_file
        download_url(state_url, data_path, state_file, None)


# taken from https://github.com/edebrouwer/gru_ode_bayes and modified
def to_pandas(state_dir, target_dir):
    name = state_dir[:-4]+'.csv'
    output_file = os.path.join(target_dir, name)

    if not os.path.exists(output_file):
        sep_list = [0, 6, 10, 12, 16]
        for day in range(31):
            sep_list = sep_list + [21+day*8, 22+day*8, 23+day*8, 24+day*8]

        columns = ['COOP_ID', 'YEAR', 'MONTH', 'ELEMENT']
        values_list = list(chain.from_iterable(("VALUE-"+str(i+1), "MFLAG-" +
                        str(i+1), "QFLAG-"+str(i+1), "SFLAG-"+str(i+1)) for i in range(31)))
        columns += values_list

        df_list = []
        with gzip.open(os.path.join(target_dir, state_dir), 'rt') as f:
            for line in f:
                line = line.strip()
                nl = [line[sep_list[i]:sep_list[i+1]]
                    for i in range(len(sep_list)-1)]
                df_list.append(nl)

        df = pd.DataFrame(df_list, columns=columns)
        val_cols = [s for s in columns if "VALUE" in s]

        df[val_cols] = df[val_cols].astype(np.float32)

        df.replace(r'\s+', np.nan, regex=True, inplace=True)
        df.replace(-9999, np.nan, inplace=True)

        df_m = df.melt(id_vars=["COOP_ID", "YEAR", "MONTH", "ELEMENT"])
        df_m[["TYPE", "DAY"]] = df_m.variable.str.split(pat="-", expand=True)

        df_n = df_m[["COOP_ID", "YEAR", "MONTH",
                    "DAY", "ELEMENT", "TYPE", "value"]].copy()

        df_p = df_n.pivot_table(values='value', index=[
                                "COOP_ID", "YEAR", "MONTH", "DAY", "ELEMENT"], columns="TYPE", aggfunc="first")
        df_p.reset_index(inplace=True)

        df_q = df_p[["COOP_ID", "YEAR", "MONTH", "DAY",
                    "ELEMENT", "MFLAG", "QFLAG", "SFLAG", "VALUE"]]

        df_q.to_csv(output_file, index=False)
        # Number of non missing
        #meas_tot = df.shape[0]*31
        #na_meas = df[val_cols].isna().sum().sum()


# taken from https://github.com/edebrouwer/gru_ode_bayes and not modified
def convert_all_to_pandas(input_dir, output_dir):
    list_dir = os.listdir(input_dir)
    txt_list_dir = [s for s in list_dir if ".txt" in s]
    state_list_dir = [s for s in txt_list_dir if "state" in s]

    for state_dir in state_list_dir:
        print(f'Computing State : {state_dir}')
        to_pandas(state_dir, output_dir)


# taken from https://github.com/edebrouwer/gru_ode_bayes and modified
def merge_dfs(input_dir, output_dir, keyword):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "daily_merged.csv")
    if not os.path.exists(output_file):
        df_list = os.listdir(input_dir)
        csv_list = [s for s in df_list if '.csv' in s]
        keyword_csv_list = [s for s in csv_list if keyword in s]

        df_list = []
        for keyword_csv in keyword_csv_list:
            print(f"Loading dataframe for keyword : {keyword_csv[:-4]}")
            df_temp = pd.read_csv(os.path.join(input_dir, keyword_csv), low_memory=False)
            #df_temp.insert(0, "UNIQUE_ID", keyword_csv[-7:-4])
            df_list.append(df_temp)
        print("All dataframes are loaded")
        # Merge all datasets:
        print("Concat all ...")
        df = pd.concat(df_list)
        df.to_csv(output_file)


# taken from https://github.com/edebrouwer/gru_ode_bayes and modified
def clean(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "cleaned_df.csv")
    if not os.path.exists(output_file):
        df = pd.read_csv(os.path.join(input_dir, "daily_merged.csv"), low_memory=False)
        print(f"Loaded df. Number of observations : {df.shape[0]}")
        # Remove NaNs
        df.drop(df.loc[df.VALUE.isna()].index, inplace=True)

        # Remove values with bad quality flag.
        qflags = ["D", "G", "I", "K", "L", "M",
                "N", "O", "R", "S", "T", "W", "X", "Z"]
        df.drop(df.loc[df.QFLAG.isin(qflags)].index, inplace=True)
        print(f"Removed bad quality flags. Number of observations {df.shape[0]}")

        # Drop centers which observations end before 1994
        gp_id_year = df.groupby("COOP_ID")["YEAR"]
        print(f"Drop centers which observations end before 1994")
        coop_list = list(gp_id_year.max().loc[gp_id_year.max() >= 1994].index)
        df.drop(df.loc[~df.COOP_ID.isin(coop_list)].index, inplace=True)
        # Drop center which observations start after 1990
        gp_id_year = df.groupby("COOP_ID")["YEAR"]
        crop_list = list(gp_id_year.min().loc[gp_id_year.min() <= 1990].index)
        df.drop(df.loc[~df.COOP_ID.isin(crop_list)].index, inplace=True)

        # Crop the observations before 1950 and after 2001.
        df = df.loc[df.YEAR >= 1950].copy()
        df = df.loc[df.YEAR <= 2000].copy()

        print(f"Number of kept centers : {df.COOP_ID.nunique()}")
        print(
            f"Number of observations / center : {df.shape[0]/df.COOP_ID.nunique()}")
        print(f"Number of days : {50*365}")

        # Create a unique_index
        unique_map = dict(zip(list(df.COOP_ID.unique()),
                        np.arange(df.COOP_ID.nunique())))
        label_map = dict(zip(list(df.ELEMENT.unique()),
                        np.arange(df.ELEMENT.nunique())))

        df.insert(0, "UNIQUE_ID", df.COOP_ID.map(unique_map))
        df.insert(1, "LABEL", df.ELEMENT.map(label_map))

        # Create a time_index.
        import datetime
        df["DATE"] = pd.to_datetime(
            (df.YEAR*10000+df.MONTH*100+df.DAY).apply(str), format='%Y%m%d')
        df["DAYS_FROM_1950"] = (df.DATE-datetime.datetime(1950, 1, 1)).dt.days

        # GENERATE TIME STAMP !!
        df["TIME_STAMP"] = df.DAYS_FROM_1950
        df.to_csv(output_file, index=False)

        # Save dict.
        np.save(os.path.join(output_dir, "centers_id_mapping_cleaned.npy"), unique_map)
        np.save(os.path.join(output_dir, "label_id_mapping_cleaned.npy"), label_map)


# new code component
def train_test_valid_split(input_dir, output_dir, test=0.2, valid=0.2, seed=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed+1)
    df = pd.read_csv(os.path.join(input_dir, 'cleaned_df.csv'))
    idx = df.UNIQUE_ID.unique()

    test_idx = rng1.choice(idx, int(len(idx) * test), replace=False)
    train_valid_idx = [i for i in idx if i not in test_idx]
    valid_idx = rng2.choice(train_valid_idx, int(len(idx)*valid), replace=False)
    train_idx = [i for i in train_valid_idx if i not in valid_idx]

    test_set = df.loc[df.UNIQUE_ID.isin(test_idx), :]
    train_valid_set = df.loc[df.UNIQUE_ID.isin(train_valid_idx), :]
    valid_set = df.loc[df.UNIQUE_ID.isin(valid_idx), :]
    train_set = df.loc[df.UNIQUE_ID.isin(train_idx), :]
    print(f'test: {test_set.UNIQUE_ID.nunique()} \nvalid: {valid_set.UNIQUE_ID.nunique()} \ntrain: {train_set.UNIQUE_ID.nunique()}')

    test_set.to_csv(os.path.join(output_dir, 'cleaned_test.csv'))
    train_valid_set.to_csv(os.path.join(
        output_dir, 'cleaned_train_valid.csv'))
    valid_set.to_csv(os.path.join(output_dir, 'cleaned_valid.csv'))
    train_set.to_csv(os.path.join(output_dir, 'cleaned_train.csv'))


# new code component
def select_time_period(input_dir, input_name, output_dir, output_name, start_year, end_year):
    data = pd.read_csv(os.path.join(input_dir, input_name))
    data = data[(data['YEAR'] >= start_year) & (data['YEAR'] <= end_year)]
    data.to_csv(os.path.join(
        output_dir, f'{output_name}_{start_year}_{end_year}.csv'), index=False)


# new code component
def cleaning_after_split(input_dir, name, output_dir, outlier_thr=4, scaling='standardize', min_time_points_per_center=None):
    df = pd.read_csv(os.path.join(input_dir, name))
    df = df[['LABEL', 'COOP_ID', 'YEAR', 'VALUE', 'ELEMENT', 'TIME_STAMP']]

    for label in ["SNOW", "SNWD", "PRCP", "TMAX", "TMIN"]:
        # remove outliers
        avg = df.loc[df.ELEMENT == label, "VALUE"].mean()
        s_dev = df.loc[df.ELEMENT == label, "VALUE"].std()
        print(f'{label} before outliers are removed: mean {avg} std {s_dev}')
        df.loc[df.ELEMENT == label, "VALUE"] = np.where(abs(
            df.loc[df.ELEMENT == label, "VALUE"] - avg) > outlier_thr*s_dev, np.nan, df.loc[df.ELEMENT == label, "VALUE"])
        print(f'outliers removed for label {label}:', df[[
              'VALUE']].isna().sum().item(), avg)
        df.dropna(axis=0, subset=['VALUE'], inplace=True)

        # scale data
        if scaling == 'standardize':
            avg = df.loc[df.ELEMENT == label, "VALUE"].mean()
            s_dev = df.loc[df.ELEMENT == label, "VALUE"].std()
            df.loc[df.ELEMENT == label, "VALUE"] -= avg
            df.loc[df.ELEMENT == label, "VALUE"] /= s_dev
            print(
                f'{label} after standardization: mean {df.loc[df.ELEMENT==label,"VALUE"].mean()} std {df.loc[df.ELEMENT==label,"VALUE"].std()}')

        elif scaling == 'normalize':
            label_min = df.loc[df.ELEMENT == label, "VALUE"].min()
            label_max = df.loc[df.ELEMENT == label, "VALUE"].max()
            label_max = 1 if label_max == 0 else label_max
            df.loc[df.ELEMENT == label, "VALUE"] -= label_min
            df.loc[df.ELEMENT == label, "VALUE"] /= (label_max - label_min)
            print(f'min value {label_min}, max value {label_max}')
            print(
                f'{label} after normalization: mean {df.loc[df.ELEMENT==label,"VALUE"].mean()} std {df.loc[df.ELEMENT==label,"VALUE"].std()}')

        else:
            raise Exception('Scaling method unknown')

    # adjust data format to pivot table
    pivot = df.pivot(index=['COOP_ID', 'TIME_STAMP'],
                     columns='LABEL', values='VALUE').reset_index()
    pivot = pivot.sort_values(['COOP_ID', 'TIME_STAMP'])

    # drop centers that have too little observations (just a handfull)
    if min_time_points_per_center is not None:
        time_points_per_center = pivot.groupby(['COOP_ID'])['COOP_ID'].count()
        centers_to_drop = list(
            time_points_per_center.loc[time_points_per_center < min_time_points_per_center].index)
        pivot.drop(pivot.loc[pivot.COOP_ID.isin(
            centers_to_drop)].index, inplace=True)

    # create new id
    print('mapping to unique id...')
    unique_map = dict(zip(list(pivot.COOP_ID.unique()),
                      np.arange(pivot.COOP_ID.nunique())))
    pivot.insert(0, "UNIQUE_ID", pivot.COOP_ID.map(unique_map))
    pivot.drop('COOP_ID', axis=1, inplace=True)

    print('writing to file...')
    np.save(os.path.join(
        output_dir, f'center_id_map_{name[:-14]}.npy'), unique_map)
    pivot.to_csv(os.path.join(
        output_dir, f'pivot_{name[:-4]}_thr{outlier_thr}_{scaling}.csv'), index=False)


# new code component
def download_and_process_ushcn(file_path):
    raw_path = os.path.join(file_path, 'raw')
    processed_path = os.path.join(file_path, 'processed')
    #split_path = os.path.join(file_path, 'train_test_val_splits')
    sets = ['train', 'train_valid', 'test', 'valid']
    start_year = 1990
    end_year = 1993

    # downloads daily state files from ushcn webpage
    download_ushcn(raw_path)

    # converts .txt files to .csv
    convert_all_to_pandas(input_dir=raw_path, output_dir=raw_path)

    # reads the individual state .csv files and merges it to a single file named "daily_merged.csv"
    merge_dfs(input_dir=raw_path, output_dir=processed_path, keyword='state')

    # cleans daily_merged.csv
    clean(input_dir=processed_path, output_dir=processed_path)

    # splits data in disjoint train, valid and test sets and saves the union of train and valid set as train_valid
    train_test_valid_split(
        input_dir=processed_path, output_dir=processed_path)

    for set in sets:
        select_time_period(input_dir=processed_path, input_name=f'cleaned_{set}.csv',
                        output_dir=processed_path, output_name=set, start_year=start_year, end_year=end_year)
        cleaning_after_split(input_dir=processed_path, name=f'{set}_{start_year}_{end_year}.csv', 
                        output_dir=file_path, scaling='normalize', min_time_points_per_center=730)

