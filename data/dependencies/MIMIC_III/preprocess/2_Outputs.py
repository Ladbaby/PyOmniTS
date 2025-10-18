# Code from: https://github.com/Ladbaby/PyOmniTS
import sys

import pandas as pd

file_path = sys.argv[1]

adm=pd.read_csv(file_path+"Admissions_processed.csv")

# We now consider the outputevents dataset. We select only the patients with the same criteria as above.
try:
    outputs=pd.read_csv(file_path+"OUTPUTEVENTS.csv")
except:
    outputs=pd.read_csv(file_path+"OUTPUTEVENTS.csv.gz", compression="gzip")

#Some checks
assert(len(outputs.loc[outputs["ISERROR"].notnull()].index)==0) #No entry with iserror==TRUE

#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm["HADM_ID"])
outputs=outputs.loc[outputs["HADM_ID"].isin(adm_ids)]

print(f"Patients after admission id filter: {outputs['SUBJECT_ID'].nunique()}")

# We load the D_ITEMS dataframe which contains the name of the ITEMID. And we merge both tables together.

#item_id 
try:
    item_id=pd.read_csv(file_path+"D_ITEMS.csv")
except:
    item_id=pd.read_csv(file_path+"D_ITEMS.csv.gz", compression="gzip")
item_id_1=item_id[["ITEMID","LABEL"]]

#We merge the name of the item administrated.
outputs_2=pd.merge(outputs,item_id_1,on="ITEMID")
print(f"Patients after name merging: {outputs_2['SUBJECT_ID'].nunique()}")

# We compute the number of patients that have the specific outputs labels and we select only the features that are the most present over the whole data set. For this, we rank the features by number of patients and select the n_best.

n_best=15
#For each item, evaluate the number of patients who have been given this item.
pat_for_item=outputs_2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 20 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]

#Select only the time series with high occurence.
outputs_3=outputs_2.loc[outputs_2["LABEL"].isin(list(frequent_labels.index))].copy()

print(f"Patients after high occurence filter: {outputs_3['SUBJECT_ID'].nunique()}")
print(f"Datapoints after high occurence filter: {len(outputs_3.index)}")

# #### Eventually, we select the same labels of the paper

outputs_label_list=['Gastric Gastric Tube','Stool Out Stool','Urine Out Incontinent','Ultrafiltrate Ultrafiltrate','Foley', 'Void','Condom Cath','Fecal Bag','Ostomy (output)','Chest Tube #1','Chest Tube #2','Jackson Pratt #1','OR EBL','Pre-Admission','TF Residual']
outputs_bis=outputs_2.loc[outputs_2["LABEL"].isin(outputs_label_list)].copy()

print(f"Patients after label filter: {outputs_bis['SUBJECT_ID'].nunique()}")
print(f"Datapoints after label filter: {len(outputs_bis.index)}")

outputs_3=outputs_bis.copy()

#Remove all entries whose rate is more than 4 std away from the mean.
out_desc=outputs_3.groupby("LABEL")["VALUE"].describe()
name_list=list(out_desc.loc[out_desc["count"]!=0].index)
for label in name_list:
    outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]==label)&(outputs_3["VALUE"]>(out_desc.loc[label,"mean"]+4*out_desc.loc[label,"std"]))].index).copy()

print(f"Patients after outlier filter: {outputs_3['SUBJECT_ID'].nunique()}")
print(f"Datapoints after outlier filter: {len(outputs_3.index)}")

#Clean Foley, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Foley") & (outputs_3["VALUE"]>5500)].index).copy()
#Clean Expected Blood Loss, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR EBL") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Out Expected Blood Loss, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR Out EBL") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean OR Urine, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="OR Urine") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Pre-Admission, remove too large and negative values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]<0)].index).copy()
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Pre-Admission output, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Pre-Admission Output Pre-Admission Output") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Urine Out Foley output, remove too large values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Urine Out Foley") & (outputs_3["VALUE"]>5000)].index).copy()
#Clean Void, remove negative values
outputs_3=outputs_3.drop(outputs_3.loc[(outputs_3["LABEL"]=="Void") & (outputs_3["VALUE"]<0)].index).copy()

outputs_3.dropna(subset=["VALUE"],inplace=True)

print(f"Patients after final filter: {outputs_3['SUBJECT_ID'].nunique()}")
print(f"Datapoints after final filter: {len(outputs_3.index)}")

# As data is already in timestamp format, we don't neeed to consider rates

outputs_3.to_csv(f"{file_path}OUTPUTS_processed.csv")
print(f'{file_path}OUTPUTS_processed.csv saved')