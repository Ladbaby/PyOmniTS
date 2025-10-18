# Code from: https://github.com/Ladbaby/PyOmniTS
import sys

import pandas as pd

file_path = sys.argv[1]

adm=pd.read_csv(file_path+"Admissions_processed.csv")

# We now consider the prescriptions dataset. We select only the patients present in the cleaned admission file
try:
    presc=pd.read_csv(file_path+"PRESCRIPTIONS.csv")
except:
    presc=pd.read_csv(file_path+"PRESCRIPTIONS.csv.gz", compression="gzip")

#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm["HADM_ID"])
presc=presc.loc[presc["HADM_ID"].isin(adm_ids)]

print(f"Patients after admission id filter: {presc['SUBJECT_ID'].nunique()}")

#This part is for selecting the x most frequent prescriptions. Instead we use the list of prescriptions as in the paper.

n_best=10
#For each item, evaluate the number of patients who have been given this item.
pat_for_item=presc.groupby("DRUG")["SUBJECT_ID"].nunique()
#Order by occurence and take the 20 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]

#Select only the time series with high occurence.
presc2=presc.loc[presc["DRUG"].isin(list(frequent_labels.index))].copy()

print(f"Patients after high occurence filter: {presc2['SUBJECT_ID'].nunique()}")

#Select entries whose drug name is in the list from the paper.
drugs_list=["Aspirin","Bisacodyl","Docusate Sodium","D5W","Humulin-R Insulin","Potassium Chloride","Magnesium Sulfate","Metoprolol Tartrate","Sodium Chloride 0.9%  Flush","Pantoprazole"]
presc2=presc.loc[presc["DRUG"].isin(drugs_list)]

print(f"Patients after drug name filter: {presc2['SUBJECT_ID'].nunique()}")

# ### Units Cleaning
# 
# #### 1) In amounts

#Verification that all input labels have the same amounts units.

#Units correction
presc2=presc2.drop(presc2.loc[presc2["DOSE_UNIT_RX"].isnull()].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Acetaminophen")&(presc2["DOSE_UNIT_RX"]!="mg")].index).copy()
presc2.loc[(presc2["DRUG"]=="D5W")&(presc2["DOSE_UNIT_RX"]=="ml"),"DOSE_UNIT_RX"]="mL"
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="D5W")&(presc2["DOSE_UNIT_RX"]!="mL")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Heparin")&(presc2["DOSE_UNIT_RX"]!="UNIT")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Insulin")&(presc2["DOSE_UNIT_RX"]!="UNIT")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Magnesium Sulfate")&(presc2["DOSE_UNIT_RX"]!="gm")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Potassium Chloride")&(presc2["DOSE_UNIT_RX"]!="mEq")].index).copy()
presc2.loc[(presc2["DRUG"]=="Sodium Chloride 0.9%  Flush")&(presc2["DOSE_UNIT_RX"]=="ml"),"DOSE_UNIT_RX"]="mL"
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Bisacodyl")&(presc2["DOSE_UNIT_RX"]!="mg")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Humulin-R Insulin")&(presc2["DOSE_UNIT_RX"]!="UNIT")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["DRUG"]=="Pantoprazole")&(presc2["DOSE_UNIT_RX"]!="mg")].index).copy()

# ### Check for outliers
# 
# #### 1) In amounts

#We need to transform the value columns in float type.
original_num_entries=len(presc2.index)
#First transform the ranges (xx-yy) as the mean of the ranges.
range_df=presc2.loc[presc2["DOSE_VAL_RX"].str.contains("-")].copy()
range_df["First_digit"]=range_df["DOSE_VAL_RX"].str.split("-").str[0].astype(float)
range_df["Second_digit"]=range_df["DOSE_VAL_RX"].str.split("-").str[1]
range_df.loc[range_df["Second_digit"]=="",'Second_digit']=range_df.loc[range_df["Second_digit"]=="",'First_digit']
range_df["Second_digit"]=range_df["Second_digit"].astype(float)
range_df.head()
range_df["mean"]=(range_df["First_digit"]+range_df["Second_digit"])/2
range_df["DOSE_VAL_RX"]=range_df["mean"]
range_df.drop(columns=["First_digit","Second_digit","mean"],inplace=True)


#Now remove the entries with the - from the original df and force conversion to float.
presc3=presc2.drop(presc2.loc[presc2["DOSE_VAL_RX"].str.contains("-")].index).copy()
presc3["DOSE_VAL_RX"]=pd.to_numeric(presc2["DOSE_VAL_RX"], errors="coerce")
presc3.dropna(subset=["DOSE_VAL_RX"],inplace=True)

presc2=presc3.append(range_df)

print(f"Dropped entries: {original_num_entries-len(presc2.index)}")

#Remove all entries whose rate is more than 4 std away from the mean.
presc_desc=presc2.groupby("DRUG")["DOSE_VAL_RX"].describe()
name_list=list(presc_desc.loc[presc_desc["count"]!=0].index)
for label in name_list:
    presc2=presc2.drop(presc2.loc[(presc2["DRUG"]==label)&(presc2["DOSE_VAL_RX"]>(presc_desc.loc[label,"mean"]+4*presc_desc.loc[label,"std"]))].index).copy()

print(f"Patients after outlier filter: {presc2['SUBJECT_ID'].nunique()}")
print(f"Datapoints after outlier filter: {len(presc2.index)}")

presc2['CHARTTIME']=pd.to_datetime(presc2["STARTDATE"], format='%Y-%m-%d %H:%M:%S')

#To avoid confounding labels with labels from other tables, we add "drug" to the name
presc2["DRUG"]=presc2["DRUG"]+" Drug"


presc2.to_csv(f"{file_path}PRESCRIPTIONS_processed.csv")
print(f'{file_path}PRESCRIPTIONS_processed.csv saved')