from datetime import timedelta
import sys

import pandas as pd

file_path = sys.argv[1]
outfile_path="~/.tsdm/rawdata/MIMIC_IV_Bilos2021/"

lab_df=pd.read_csv(f"{file_path}processed/LAB_processed.csv")[["subject_id","hadm_id","charttime","valuenum","label"]]
inputs_df=pd.read_csv(f"{file_path}processed/INPUTS_processed.csv")[["subject_id","hadm_id","charttime","amount","label"]]
outputs_df=pd.read_csv(f"{file_path}processed/OUTPUTS_processed.csv")[["subject_id","hadm_id","charttime","value","label"]]
presc_df=pd.read_csv(f"{file_path}processed/PRESCRIPTIONS_processed.csv")[["subject_id","hadm_id","charttime","dose_val_rx","drug"]]


#Change the name of amount. Valuenum for every table
inputs_df["valuenum"]=inputs_df["amount"]
inputs_df=inputs_df.drop(columns=["amount"]).copy()

outputs_df["valuenum"]=outputs_df["value"]
outputs_df=outputs_df.drop(columns=["value"]).copy()

presc_df["valuenum"]=presc_df["dose_val_rx"]
presc_df=presc_df.drop(columns=["dose_val_rx"]).copy()
presc_df["label"]=presc_df["drug"]
presc_df=presc_df.drop(columns=["drug"]).copy()
presc_df = presc_df.drop((presc_df['valuenum']=='3-10').index)

#Tag to distinguish between lab and inputs events
inputs_df["Origin"]="Inputs"
lab_df["Origin"]="Lab"
outputs_df["Origin"]="Outputs"
presc_df["Origin"]="Prescriptions"

#merge both dfs.
merged_df1=(inputs_df.append(lab_df)).reset_index()
merged_df2=(merged_df1.append(outputs_df)).reset_index()
merged_df2.drop(columns="level_0",inplace=True)
merged_df=(merged_df2.append(presc_df)).reset_index()

#Check that all labels have different names.
assert(merged_df["label"].nunique()==(inputs_df["label"].nunique()+lab_df["label"].nunique()+outputs_df["label"].nunique()+presc_df["label"].nunique()))


# set the timestamp as the time delta between the first chart time for each admission
merged_df['charttime']=pd.to_datetime(merged_df["charttime"], format='%Y-%m-%d %H:%M:%S')
ref_time=merged_df.groupby("hadm_id")["charttime"].min()
merged_df_1=pd.merge(ref_time.to_frame(name="ref_time"),merged_df,left_index=True,right_on="hadm_id")
merged_df_1["time_stamp"]=merged_df_1["charttime"]-merged_df_1["ref_time"]
assert(len(merged_df_1.loc[merged_df_1["time_stamp"]<timedelta(hours=0)].index)==0)

# Create a label code (int) for the labels.
label_dict=dict(zip(list(merged_df_1["label"].unique()),range(len(list(merged_df_1["label"].unique())))))
merged_df_1["label_code"]=merged_df_1["label"].map(label_dict)

merged_df_short=merged_df_1[["hadm_id","valuenum","time_stamp","label_code","Origin"]]

label_dict_df=pd.Series(merged_df_1["label"].unique()).reset_index()
label_dict_df.columns=["index","label"]
label_dict_df["label_code"]=label_dict_df["label"].map(label_dict)
label_dict_df.drop(columns=["index"],inplace=True)
label_dict_df.to_csv(f"{file_path}processed/label_dict.csv")

merged_df_short["valuenum"] = merged_df_short["valuenum"].astype(float)

# select only values within first 48 hours
merged_df_short=merged_df_short.loc[(merged_df_short["time_stamp"]<timedelta(hours=48))]
merged_df_short["time_stamp"] = merged_df_short["time_stamp"].dt.total_seconds().div(60).astype(int)
print(f"Number of patients considered: {merged_df_short['hadm_id'].nunique()}")
assert(len(merged_df_short.loc[merged_df_short["time_stamp"]>2880].index)==0)

# drop columns that are not needed for final dataset
merged_df_short.drop(["Origin"], axis=1, inplace=True)
complete_df = merged_df_short

# create value- and mask- columns and fill with data
labels = complete_df["label_code"].unique()
value_columns = []
mask_columns  = []
for num in labels:
    name = "Value_label_" + str(num)
    name2 = "Mask_label_" + str(num)
    value_columns.append(name)
    mask_columns.append(name2)
    complete_df[name] = 0
    complete_df[name2] = 0
    complete_df[name] = complete_df[name].astype(float)

complete_df.dropna(inplace=True)
for index, row in complete_df.iterrows():
    name = "Value_label_" + str(row["label_code"].astype(int))
    name2 = "Mask_label_" + str(row["label_code"].astype(int))
    complete_df.at[index, name] = row["valuenum"]
    complete_df.at[index, name2] = 1

# drop all unneccesary columns and do sanity check
complete_df.drop(["valuenum", "label_code"], axis=1, inplace=True)
complete_df = complete_df.groupby(["hadm_id", "time_stamp"], as_index=False).max()
for x in mask_columns:
    assert(len(complete_df.loc[complete_df[x]>1])==0)

complete_df.to_csv(f"{outfile_path}full_dataset.csv", index=False)
print(f'{outfile_path}full_dataset.csv saved')