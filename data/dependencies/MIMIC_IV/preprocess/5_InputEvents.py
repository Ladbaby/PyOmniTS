from datetime import timedelta
import sys

import pandas as pd
import numpy as np

file_path = sys.argv[1]
adm_3 = pd.read_csv(f"{file_path}processed/Admissions_processed.csv")

# only choose previously selected admission ids
inputs=pd.read_csv(f"{file_path}icu/inputevents.csv.gz")
adm_ids=list(adm_3["hadm_id"])
inputs=inputs.loc[inputs["hadm_id"].isin(adm_ids)]

# only keep columns of interest
inputs_small=inputs[["subject_id","hadm_id","starttime","endtime","itemid","amount","amountuom","rate","rateuom","patientweight","ordercategorydescription"]]
print(f'Patients after admission id filter: {inputs_small["subject_id"].nunique()}')

# get item ids for inputs 
item_id=pd.read_csv(f"{file_path}icu/d_items.csv.gz")
item_id_1=item_id[["itemid","label"]]

inputs_small_2=pd.merge(inputs_small,item_id_1,on="itemid")
print(f'Patients after merging: {inputs_small_2["subject_id"].nunique()}')

#For each item, evaluate the number of patients who have been given this item
# Select only the inputs with highest occurence
pat_for_item=inputs_small_2.groupby("label")["subject_id"].nunique()
frequent_labels=pat_for_item.sort_values(ascending=False)[:50]
inputs_small_3=inputs_small_2.loc[inputs_small_2["label"].isin(list(frequent_labels.index))].copy()

print(f'Patients after occurence filter: {inputs_small_3["subject_id"].nunique()}')

##### Cleaning the Cefazolin (remove the ones that are not in dose unit)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["itemid"]==225850) & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Cefepime (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Cefepime") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Ceftriaxone (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Ceftriaxone") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Ciprofloxacin (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Ciprofloxacin") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Famotidine (Pepcid) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Famotidine (Pepcid)") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Fentanyl (Concentrate) (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Fentanyl (Concentrate)") & (inputs_small_3["amountuom"]!="mg")].index).copy()
inputs_small_3.loc[(inputs_small_3["label"]=="Fentanyl (Concentrate)") & (inputs_small_3["amountuom"]=="mg"),"amount"]*=1000
inputs_small_3.loc[(inputs_small_3["label"]=="Fentanyl (Concentrate)") & (inputs_small_3["amountuom"]=="mg"),"amountuom"]="mcg"
#Cleaning the Heparin Sodium (Prophylaxis) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Heparin Sodium (Prophylaxis)") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Hydromorphone (Dilaudid) (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Hydromorphone (Dilaudid)") & (inputs_small_3["amountuom"]!="mg")].index).copy()
#Cleaning the Magnesium Sulfate (remove the non grams)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Magnesium Sulfate") & (inputs_small_3["amountuom"]!="grams")].index).copy()
#Cleaning the Propofol (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Propofol") & (inputs_small_3["amountuom"]!="mg")].index).copy()
#Cleaning the Metoprolol (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Metoprolol") & (inputs_small_3["amountuom"]!="mg")].index).copy()
#Cleaning the Piperacillin/Tazobactam (Zosyn) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Piperacillin/Tazobactam (Zosyn)") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Metronidazole (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Metronidazole") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Ranitidine (Prophylaxis)(remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Ranitidine (Prophylaxis)") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Vancomycin (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Vancomycin") & (inputs_small_3["amountuom"]!="dose")].index).copy()
#Cleaning the Fentanyl. Put the mg to mcg 
inputs_small_3.loc[(inputs_small_3["itemid"]==221744) & (inputs_small_3["amountuom"]=="mg"),"amount"]*=1000
inputs_small_3.loc[(inputs_small_3["itemid"]==221744) & (inputs_small_3["amountuom"]=="mg"),"amountuom"]="mcg"
#Cleaning of the Pantoprazole (Protonix)
    #divide in two (drug shot or continuous treatment and create a new item id for the continuous version)
inputs_small_3.loc[(inputs_small_3["itemid"]==225910) & (inputs_small_3["ordercategorydescription"]=="Continuous Med"),"label"]="Pantoprazole (Protonix) Continuous"
inputs_small_3.loc[(inputs_small_3["itemid"]==225910) & (inputs_small_3["ordercategorydescription"]=="Continuous Med"),"itemid"]=2217441
#remove the non dose from the drug shot version
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Pantoprazole (Protonix)") & (inputs_small_3["amountuom"]!="dose")].index).copy()

# Additional Preprocessing for MIMIC 4 items
#Cleaning the Acetaminophen-IV (keep mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Acetaminophen-IV") & (inputs_small_3["amountuom"]!="mg")].index).copy()

#Cleaning the D5 1/2NS (keep ml)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="D5 1/2NS") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning the Dexmedetomidine (Precedex) (cast all to mg)
inputs_small_3.loc[(inputs_small_3["label"]=="Dexmedetomidine (Precedex)") & (inputs_small_3["amountuom"]=="mcg"),"amount"]/=1000
inputs_small_3.loc[(inputs_small_3["label"]=="Dexmedetomidine (Precedex)") & (inputs_small_3["amountuom"]=="mcg"),"amountuom"]="mg"

#Cleaning the LR
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="LR") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning the NaCl 0.9%
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="NaCl 0.9%") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning the OR Crystalloid Intake 
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="OR Crystalloid Intake") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning the PO Intake
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="PO Intake") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning the Pre-Admission/Non-ICU Intake 
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Pre-Admission/Non-ICU Intake") & (inputs_small_3["amountuom"]!="ml")].index).copy()

#Cleaning of Dextrose 5%  (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Dextrose 5%") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Magnesium Sulfate (Bolus)  (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Magnesium Sulfate (Bolus)") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()
#Cleaning of NaCl 0.9% (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="NaCl 0.9%") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Piggyback (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Piggyback") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()
#Cleaning of Packed Red Bllod Cells (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Packed Red Blood Cells") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()

# additional cleaning for mimic4
#Cleaning of Acetaminophen-IV
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Acetaminophen-IV") & (inputs_small_3["rateuom"]!="mg/min")].index).copy()

#Cleaning of Fentanyl (Concentrate)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Fentanyl (Concentrate)") & (inputs_small_3["rateuom"]!="mcg/hour")].index).copy()

#Cleaning of Phenylephrine
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Phenylephrine") & (inputs_small_3["rateuom"]!="mcg/kg/min")].index).copy()

#Cleaning of Sterile Water 
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["label"]=="Sterile Water") & (inputs_small_3["rateuom"]!="mL/hour")].index).copy()

# We now split the entries which are spread in time. We chose the duration window for the sampling. here we choose 30 minutes. 
# So every entry which has a rate and with duration larger than 1 hour, we split it into fixed times injections.

#First check the /hours units
df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mcg/kg/hour"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/3600)*df_temp["patientweight"]

assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-1000*df_temp["amount"])>0.01)].index)==0) #OK

df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mL/hour"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/3600)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-df_temp["amount"])>0.01)].index)==0) #OK

df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mg/hour"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/3600)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-df_temp["amount"])>0.01)].index)==0) #OK

df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mcg/hour"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/3600)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-df_temp["amount"])>0.01)].index)==0) #OK

df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("units/hour"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/3600)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-df_temp["amount"])>0.01)].index)==0) #OK

df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mg/min"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/60)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]-df_temp["amount"])>0.01)].index)==0) #OK

#Third check the kg/min units
df_temp=inputs_small_3.loc[(inputs_small_3["rate"].notnull()) & (inputs_small_3["rateuom"].str.contains("mcg/kg/min"))].copy()
df_temp["computed_amount"]=df_temp["rate"]*((pd.to_datetime(df_temp["endtime"])-pd.to_datetime(df_temp["starttime"])).dt.total_seconds()/60)*df_temp["patientweight"]

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["computed_amount"]/1000-df_temp["amount"])>0.01)].index)==0) #OK

duration_split_hours=0.5
to_sec_fact=3600*duration_split_hours

#split data set in four.

#The first dataframe contains the entries with no rate but with extended duration inputs (over 0.5 hour)
df_temp1=inputs_small_3.loc[((pd.to_datetime(inputs_small_3["endtime"])-pd.to_datetime(inputs_small_3["starttime"]))>timedelta(hours=duration_split_hours)) & (inputs_small_3["rate"].isnull())].copy().reset_index(drop=True)
#The second dataframe contains the entries with no rate and low duration entries (<0.5hour)
df_temp2=inputs_small_3.loc[((pd.to_datetime(inputs_small_3["endtime"])-pd.to_datetime(inputs_small_3["starttime"]))<=timedelta(hours=duration_split_hours)) & (inputs_small_3["rate"].isnull())].copy().reset_index(drop=True)
#The third dataframe contains the entries with a rate and extended duration inputs (over 0.5 hour)
df_temp3=inputs_small_3.loc[((pd.to_datetime(inputs_small_3["endtime"])-pd.to_datetime(inputs_small_3["starttime"]))>timedelta(hours=duration_split_hours)) & (inputs_small_3["rate"].notnull())].copy().reset_index(drop=True)
#The forth dataframe contains the entries with a rate and low duration entries (< 0.5 hour)
df_temp4=inputs_small_3.loc[((pd.to_datetime(inputs_small_3["endtime"])-pd.to_datetime(inputs_small_3["starttime"]))<=timedelta(hours=duration_split_hours)) & (inputs_small_3["rate"].notnull())].copy().reset_index(drop=True)

#Check if split is complete
assert(len(df_temp1.index)+len(df_temp2.index)+len(df_temp3.index)+len(df_temp4.index)==len(inputs_small_3.index))

#We then process all of these dfs.
#In the first one, we need to duplicate the entries according to their duration and then divide each entry by the number of duplicates

#We duplicate the rows with the number bins for each injection
df_temp1["Repeat"]=np.ceil((pd.to_datetime(df_temp1["endtime"])-pd.to_datetime(df_temp1["starttime"])).dt.total_seconds()/to_sec_fact).astype(int)
df_new1=df_temp1.reindex(df_temp1.index.repeat(df_temp1["Repeat"]))

#We then create the admninistration time as a shifted version of the STARTTIME.
df_new1["charttime"]=df_new1.groupby(level=0)['starttime'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new1["amount"]=df_new1["amount"]/df_new1["Repeat"]

# In the third one, we do the same
#We duplicate the rows with the number bins for each injection
df_temp3["Repeat"]=np.ceil((pd.to_datetime(df_temp3["endtime"])-pd.to_datetime(df_temp3["starttime"])).dt.total_seconds()/to_sec_fact).astype(int)
df_new3=df_temp3.reindex(df_temp3.index.repeat(df_temp3["Repeat"]))
#We then create the admninistration time as a shifted version of the STARTTIME.

df_new3["charttime"]=df_new3.groupby(level=0)['starttime'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new3["amount"]=df_new3["amount"]/df_new3["Repeat"]

df_temp2["charttime"]=df_temp2["starttime"]
df_temp4["charttime"]=df_temp4["starttime"]

#Eventually, we merge all 4splits into one.
inputs_small_4=df_new1.append([df_temp2,df_new3,df_temp4])
#The result is a dataset with discrete inputs for each treatment.

inputs_small_4.to_csv(f"{file_path}processed/INPUTS_processed.csv")
print(f'{file_path}processed/INPUTS_processed.csv saved')