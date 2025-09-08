import sys
import os

import pandas as pd
import numpy as np

file_path = sys.argv[1]
fn = 'core/admissions.csv.gz'

adm = pd.read_csv(file_path + fn, compression='gzip')

#keep only patients present in patients data
patients_df=pd.read_csv(file_path + 'core/patients.csv.gz')
adm_dob=pd.merge(patients_df[["subject_id","anchor_age"]],adm,on="subject_id")

df=adm.groupby("subject_id")["hadm_id"].nunique()
subj_ids=list(df[df==1].index)
adm_1=adm_dob.loc[adm_dob["subject_id"].isin(subj_ids)]
print(f"Patients after subject id filter: {len(adm_1.index)}")


# time of stay in ICU
adm_1=adm_1.copy()
adm_1['admittime']=pd.to_datetime(adm_1["admittime"], format='%Y-%m-%d %H:%M:%S')
adm_1['dischtime']=pd.to_datetime(adm_1["dischtime"], format='%Y-%m-%d %H:%M:%S')

adm_1["elapsed_time"]=adm_1["dischtime"]-adm_1["admittime"]
adm_1["elapsed_days"]=adm_1["elapsed_time"].dt.days 

adm_2=adm_1.loc[(adm_1["elapsed_days"]<30) & (adm_1["elapsed_days"]>2)]
print(f"Patients after time of stay filter: {len(adm_2.index)}")

# only patients older than 15
adm_2_15=adm_2.loc[adm_2["anchor_age"]>15].copy()
print(f"Patients after age filter: {len(adm_2_15.index)}")

# workaround:
ids = np.array([])
for chunk in pd.read_csv(file_path + 'icu/chartevents.csv.gz', chunksize=1000000):
    ids = np.append(ids, chunk['hadm_id'].unique())
    ids = np.unique(ids)


adm_2_15_chart=adm_2_15.loc[adm_2_15["hadm_id"].isin(ids)].copy()
print(f"Patients after admission id filter: {len(adm_2_15_chart.index)}")

os.makedirs(f"{file_path}/processed", exist_ok=True)
adm_2_15_chart.to_csv(f"{file_path}processed/Admissions_processed.csv")
print(f'{file_path}processed/Admissions_processed.csv saved')