from datetime import timedelta
import sys

import pandas as pd
import numpy as np

file_path = sys.argv[1]
outfile_path="~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/"

lab_df=pd.read_csv(file_path+"LAB_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","VALUENUM","LABEL"]]
inputs_df=pd.read_csv(file_path+"INPUTS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","AMOUNT","LABEL"]]
outputs_df=pd.read_csv(file_path+"OUTPUTS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","VALUE","LABEL"]]
presc_df=pd.read_csv(file_path+"PRESCRIPTIONS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","DOSE_VAL_RX","DRUG"]]

#Process names of columns to have the same everywhere.

#Change the name of amount. Valuenum for every table
inputs_df["VALUENUM"]=inputs_df["AMOUNT"]
inputs_df.head()
inputs_df=inputs_df.drop(columns=["AMOUNT"]).copy()

#Change the name of amount. Valuenum for every table
outputs_df["VALUENUM"]=outputs_df["VALUE"]
outputs_df=outputs_df.drop(columns=["VALUE"]).copy()

#Change the name of amount. Valuenum for every table
presc_df["VALUENUM"]=presc_df["DOSE_VAL_RX"]
presc_df=presc_df.drop(columns=["DOSE_VAL_RX"]).copy()
presc_df["LABEL"]=presc_df["DRUG"]
presc_df=presc_df.drop(columns=["DRUG"]).copy()


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
assert(merged_df["LABEL"].nunique()==(inputs_df["LABEL"].nunique()+lab_df["LABEL"].nunique()+outputs_df["LABEL"].nunique()+presc_df["LABEL"].nunique()))

#Set the reference time as the lowest chart time for each admission.
merged_df['CHARTTIME']=pd.to_datetime(merged_df["CHARTTIME"], format='%Y-%m-%d %H:%M:%S')
ref_time=merged_df.groupby("HADM_ID")["CHARTTIME"].min()

merged_df_1=pd.merge(ref_time.to_frame(name="REF_TIME"),merged_df,left_index=True,right_on="HADM_ID")
merged_df_1["TIME_STAMP"]=merged_df_1["CHARTTIME"]-merged_df_1["REF_TIME"]
assert(len(merged_df_1.loc[merged_df_1["TIME_STAMP"]<timedelta(hours=0)].index)==0)

#Create a label code (int) for the labels.
label_dict=dict(zip(list(merged_df_1["LABEL"].unique()),range(len(list(merged_df_1["LABEL"].unique())))))
merged_df_1["LABEL_CODE"]=merged_df_1["LABEL"].map(label_dict)

merged_df_short=merged_df_1[["HADM_ID","VALUENUM","TIME_STAMP","LABEL_CODE","Origin"]]

label_dict_df=pd.Series(merged_df_1["LABEL"].unique()).reset_index()
label_dict_df.columns=["index","LABEL"]
label_dict_df["LABEL_CODE"]=label_dict_df["LABEL"].map(label_dict)
label_dict_df.drop(columns=["index"],inplace=True)
# label_dict_df.to_csv(outfile_path+"label_dict.csv")


# #### Time binning of the data
# First we select the data up to a certain time limit (48 hours)

#Now only select values within 48 hours.
merged_df_short=merged_df_short.loc[(merged_df_short["TIME_STAMP"]<timedelta(hours=48))]
print(f"Number of patients considered: {merged_df_short['HADM_ID'].nunique()}")

#Plot the number of "hits" based on the binning. That is, the number of measurements falling into the same bin in function of the number of bins
bins_num=range(1,60)
merged_df_short_binned=merged_df_short.copy()
hits_vec=[]
for bin_k in bins_num:
    time_stamp_str="TIME_STAMP_Bin_"+str(bin_k)
    merged_df_short_binned[time_stamp_str]=round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36)).astype(int)
    hits_prop=merged_df_short_binned.duplicated(subset=["HADM_ID","LABEL_CODE",time_stamp_str]).sum()/len(merged_df_short_binned.index)
    hits_vec+=[hits_prop]

#We choose 2 bins per hour. We now need to aggregate the data in different ways.
bin_k=2
merged_df_short["TIME"]=round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36)).astype(int)

#For lab, we have to average the duplicates.
lab_subset=merged_df_short.loc[merged_df_short["Origin"]=="Lab",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
lab_subset["KEY_ID"]=lab_subset["HADM_ID"].astype(str)+"/"+lab_subset["TIME"].astype(str)+"/"+lab_subset["LABEL_CODE"].astype(str)
lab_subset["VALUENUM"]=lab_subset["VALUENUM"].astype(float)

lab_subset_s=lab_subset.groupby("KEY_ID")["VALUENUM"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
lab_s=pd.merge(lab_subset,lab_subset_s,on="KEY_ID")
assert(not lab_s.isnull().values.any())

#For inputs, we have to sum the duplicates.
input_subset=merged_df_short.loc[merged_df_short["Origin"]=="Inputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
input_subset["KEY_ID"]=input_subset["HADM_ID"].astype(str)+"/"+input_subset["TIME"].astype(str)+"/"+input_subset["LABEL_CODE"].astype(str)
input_subset["VALUENUM"]=input_subset["VALUENUM"].astype(float)

input_subset_s=input_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

input_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
input_s=pd.merge(input_subset,input_subset_s,on="KEY_ID")
assert(not input_s.isnull().values.any())

#For outpus, we have to sum the duplicates as well.
output_subset=merged_df_short.loc[merged_df_short["Origin"]=="Outputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
output_subset["KEY_ID"]=output_subset["HADM_ID"].astype(str)+"/"+output_subset["TIME"].astype(str)+"/"+output_subset["LABEL_CODE"].astype(str)
output_subset["VALUENUM"]=output_subset["VALUENUM"].astype(float)

output_subset_s=output_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

output_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
output_s=pd.merge(output_subset,output_subset_s,on="KEY_ID")
assert(not output_s.isnull().values.any())

#For prescriptions, we have to sum the duplicates as well.
presc_subset=merged_df_short.loc[merged_df_short["Origin"]=="Prescriptions",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
presc_subset["KEY_ID"]=presc_subset["HADM_ID"].astype(str)+"/"+presc_subset["TIME"].astype(str)+"/"+presc_subset["LABEL_CODE"].astype(str)
presc_subset["VALUENUM"]=presc_subset["VALUENUM"].astype(float)

presc_subset_s=presc_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
presc_s=pd.merge(presc_subset,presc_subset_s,on="KEY_ID")
assert(not presc_s.isnull().values.any())

#Now remove the duplicates/
lab_s=(lab_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
input_s=(input_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
output_s=(output_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
presc_s=(presc_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()

#We append both subsets together to form the complete dataframe
complete_df1=lab_s.append(input_s)
complete_df2=complete_df1.append(output_s)
complete_df=complete_df2.append(presc_s)


assert(sum(complete_df.duplicated(subset=["HADM_ID","LABEL_CODE","TIME"])==True)==0) #Check if no duplicates anymore.

# We remove patients with less than 50 observations.
id_counts=complete_df.groupby("HADM_ID").count()
id_list=list(id_counts.loc[id_counts["TIME"]<50].index)
complete_df=complete_df.drop(complete_df.loc[complete_df["HADM_ID"].isin(id_list)].index).copy()

# # Dataframe creation for Tensor Decomposition
# 
# Creation of a unique index for the admissions id.

#Creation of a unique index
unique_ids=np.arange(complete_df["HADM_ID"].nunique())
# np.random.shuffle(unique_ids) # this shuffle is removed to ensure the sha256 checksum of complete_tensor.csv is consistent
d=dict(zip(complete_df["HADM_ID"].unique(),unique_ids))

admissions=pd.read_csv(file_path+"Admissions_processed.csv")
death_tags_s=admissions.groupby("HADM_ID")["DEATHTAG"].unique().astype(int).to_frame().reset_index()
death_tags_df=death_tags_s.loc[death_tags_s["HADM_ID"].isin(complete_df["HADM_ID"])].copy()
death_tags_df["UNIQUE_ID"]=death_tags_df["HADM_ID"].map(d)
death_tags_df.sort_values(by="UNIQUE_ID",inplace=True)
death_tags_df.rename(columns={"DEATHTAG":"Value"},inplace=True)
# death_tags_df.to_csv(outfile_path+"complete_death_tags.csv")

complete_df["UNIQUE_ID"] = complete_df["HADM_ID"].map(d)


#ICD9 codes
try:
    ICD_diag=pd.read_csv(file_path+"DIAGNOSES_ICD.csv")
except:
    ICD_diag=pd.read_csv(file_path+"DIAGNOSES_ICD.csv.gz", compression="gzip")

main_diag=ICD_diag.loc[(ICD_diag["SEQ_NUM"]==1)]
complete_tensor=pd.merge(complete_df,main_diag[["HADM_ID","ICD9_CODE"]],on="HADM_ID")

#Only select the first 3 digits of each ICD9 code.
complete_tensor["ICD9_short"]=complete_tensor["ICD9_CODE"].astype(str).str[:3]
#Check that all codes are 3 digits long.
str_len=complete_tensor["ICD9_short"].str.len()
assert(str_len.loc[str_len!=3].count()==0)

#Finer encoding (3 digits)
hot_encodings=pd.get_dummies(complete_tensor["ICD9_short"])
complete_tensor[hot_encodings.columns]=hot_encodings
complete_tensor_nocov=complete_tensor[["UNIQUE_ID","LABEL_CODE","TIME"]+["VALUENUM"]].copy()

complete_tensor_nocov.rename(columns={"TIME":"TIME_STAMP"},inplace=True)


# ### Normalization of the data (N(0,1))

#Add a column with the mean and std of each different measurement type and then normalize them.
d_mean=dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].mean())
complete_tensor_nocov["MEAN"]=complete_tensor_nocov["LABEL_CODE"].map(d_mean)
d_std=dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].std())
complete_tensor_nocov["STD"]=complete_tensor_nocov["LABEL_CODE"].map(d_std)
complete_tensor_nocov["VALUENORM"]=(complete_tensor_nocov["VALUENUM"]-complete_tensor_nocov["MEAN"])/complete_tensor_nocov["STD"]

#Save locally.
complete_tensor_nocov.to_csv(f"{outfile_path}complete_tensor.csv") #Full data
print(f'{outfile_path}complete_tensor.csv saved')