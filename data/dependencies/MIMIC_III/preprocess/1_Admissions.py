'''
- python: 3.7.13
- numpy: 1.21.6
- pandas: 1.3.5
'''

from datetime import timedelta
import sys

import pandas as pd
import numpy as np

file_path = sys.argv[1]

try:
    adm=pd.read_csv(file_path+"ADMISSIONS.csv")
except:
    adm=pd.read_csv(file_path+"ADMISSIONS.csv.gz", compression="gzip")

#Load the patients data base and add the Date of birth to the admission dataset.
try:
    patients_df=pd.read_csv(file_path+"PATIENTS.csv")
except:
    patients_df=pd.read_csv(file_path+"PATIENTS.csv.gz", compression="gzip")
patients_df["DOBTIME"]=pd.to_datetime(patients_df["DOB"], format='%Y-%m-%d')
patients_df[["SUBJECT_ID","DOBTIME"]].head()
adm_dob=pd.merge(patients_df[["SUBJECT_ID","DOBTIME"]],adm,on="SUBJECT_ID")

#Number of admissions by patient
df=adm.groupby("SUBJECT_ID")["HADM_ID"].nunique()
# print("Number of patients with specific number of admissions : \n",df.value_counts())

# As the majortity of patients only present a single admission, we filter out all the patients with more than 1 admission

subj_ids=list(df[df==1].index) #index of patients with only one visit.
adm_1=adm_dob.loc[adm_dob["SUBJECT_ID"].isin(subj_ids)] #filter out the patients with more than one visit
print(f"Patients after single admission filter: {len(adm_1.index)}")

#We now add a new column with the duration of each stay.
adm_1=adm_1.copy()
adm_1['ADMITTIME']=pd.to_datetime(adm_1["ADMITTIME"], format='%Y-%m-%d %H:%M:%S')
adm_1['DISCHTIME']=pd.to_datetime(adm_1["DISCHTIME"], format='%Y-%m-%d %H:%M:%S')

adm_1["ELAPSED_TIME"]=adm_1["DISCHTIME"]-adm_1["ADMITTIME"]
adm_1.head()
adm_1["ELAPSED_DAYS"]=adm_1["ELAPSED_TIME"].dt.days #Elapsed time in days in ICU

#Let's now report the death rate in function of the duration stay in ICU.
adm_1["DEATHTAG"]=0
adm_1.loc[adm_1.DEATHTIME.notnull(),"DEATHTAG"]=1

df_deaths_per_duration=adm_1.groupby("ELAPSED_DAYS")["DEATHTAG"].sum()
df_patients_per_duration=adm_1.groupby("ELAPSED_DAYS")["SUBJECT_ID"].nunique()
df_death_ratio_per_duration=df_deaths_per_duration/df_patients_per_duration

# Given the results above, we select patients with a least 48 hours in the ICU and with less than 30 days stay.

adm_2=adm_1.loc[(adm_1["ELAPSED_DAYS"]<30) & (adm_1["ELAPSED_DAYS"]>2)]
print(f"Patients after stay duration filter: {len(adm_2.index)}")

# We remove the patients who are younger than 15 at admission time
adm_2_15=adm_2.loc[(adm_2["ADMITTIME"].values-adm_2["DOBTIME"].values).astype(np.float64)/8.64e13//365 >= 15].copy()
print(f"Patients after age filter: {len(adm_2_15.index)}")

# We remove the admissions with no chart events data.

adm_2_15_chart=adm_2_15.loc[adm_2_15["HAS_CHARTEVENTS_DATA"]==1].copy()
print(f"Patients after chart events filter: {len(adm_2_15_chart.index)}")

#We now investigate the admission_type
df_type=adm_2_15_chart.groupby("ADMISSION_TYPE")["SUBJECT_ID"].count()
df_type


# We remove the newborns as they are specific
adm_3=adm_2_15_chart.loc[adm_2_15_chart["ADMISSION_TYPE"]!="NEWBORN"]
print(f"Patients after newborns filter: {adm_3['SUBJECT_ID'].nunique()}")

adm_3.to_csv(f"{file_path}Admissions_processed.csv")
print(f'{file_path}Admissions_processed.csv saved')


# ## INPUTS EVENTS DATA
# 
# We now consider the inputevents dataset. We select only the patients in the metavision system and with the same criteria as above.
try:
    inputs=pd.read_csv(file_path+"INPUTEVENTS_MV.csv")
except:
    inputs=pd.read_csv(file_path+"INPUTEVENTS_MV.csv.gz", compression="gzip")
#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm_3["HADM_ID"])
inputs=inputs.loc[inputs["HADM_ID"].isin(adm_ids)]

#Inputs_small only contains the columns of interest.
inputs_small=inputs[["SUBJECT_ID","HADM_ID","STARTTIME","ENDTIME","ITEMID","AMOUNT","AMOUNTUOM","RATE","RATEUOM","PATIENTWEIGHT","ORDERCATEGORYDESCRIPTION"]]

# We load the D_ITEMS dataframe which contains the name of the ITEMID. And we merge both tables together.

#item_id 
try:
    item_id=pd.read_csv(file_path+"D_ITEMS.csv")
except:
    item_id=pd.read_csv(file_path+"D_ITEMS.csv.gz", compression='gzip')
item_id_1=item_id[["ITEMID","LABEL"]]

#We merge the name of the item administrated.
inputs_small_2=pd.merge(inputs_small,item_id_1,on="ITEMID")

#For each item, evaluate the number of patients who have been given this item.
pat_for_item=inputs_small_2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 33 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:50]

#Select only the time series with high occurence.
inputs_small_3=inputs_small_2.loc[inputs_small_2["LABEL"].isin(list(frequent_labels.index))].copy()
print(f"Patients after item filter: {inputs_small_3['SUBJECT_ID'].nunique()}")

#Only select specific labels for the inputs.
#list of retained inputs :
retained_list=["Albumin 5%","Dextrose 5%","Lorazepam (Ativan)","Calcium Gluconate","Midazolam (Versed)","Phenylephrine","Furosemide (Lasix)","Hydralazine","Norepinephrine","Magnesium Sulfate","Nitroglycerin","Insulin - Glargine","Insulin - Humalog","Insulin - Regular","Heparin Sodium","Morphine Sulfate","Potassium Chloride","Packed Red Blood Cells","Gastric Meds","D5 1/2NS","LR","K Phos","Solution","Sterile Water","Metoprolol","Piggyback","OR Crystalloid Intake","OR Cell Saver Intake","PO Intake","GT Flush","KCL (Bolus)","Magnesium Sulfate (Bolus)"]
#missing :Fresh Frozen Plasma
inputs_small_3=inputs_small_3.loc[inputs_small_3["LABEL"].isin(retained_list)].copy()


# # Cleaning of the input data
# 
# ### Units Cleaning
# 
# #### 1) Amounts

##### Cleaning the Cefazolin (remove the ones that are not in dose unit)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["ITEMID"]==225850) & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Cefepime (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefepime") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Ceftriaxone (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Ceftriaxone") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Ciprofloxacin (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Ciprofloxacin") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Famotidine (Pepcid) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Famotidine (Pepcid)") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Fentanyl (Concentrate) (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNTUOM"]!="mg")].index).copy()
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNTUOM"]=="mg"),"AMOUNT"]*=1000
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNTUOM"]=="mg"),"AMOUNTUOM"]="mcg"
#Cleaning the Heparin Sodium (Prophylaxis) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Heparin Sodium (Prophylaxis)") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Hydromorphone (Dilaudid) (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Hydromorphone (Dilaudid)") & (inputs_small_3["AMOUNTUOM"]!="mg")].index).copy()
#Cleaning the Magnesium Sulfate (remove the non grams)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Magnesium Sulfate") & (inputs_small_3["AMOUNTUOM"]!="grams")].index).copy()
#Cleaning the Propofol (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Propofol") & (inputs_small_3["AMOUNTUOM"]!="mg")].index).copy()
#Cleaning the Metoprolol (remove the non mg)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Metoprolol") & (inputs_small_3["AMOUNTUOM"]!="mg")].index).copy()
#Cleaning the Piperacillin/Tazobactam (Zosyn) (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Piperacillin/Tazobactam (Zosyn)") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Metronidazole (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Metronidazole") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Ranitidine (Prophylaxis)(remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Ranitidine (Prophylaxis)") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Vancomycin (remove the non dose)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Vancomycin") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()
#Cleaning the Fentanyl. Put the mg to mcg 
inputs_small_3.loc[(inputs_small_3["ITEMID"]==221744) & (inputs_small_3["AMOUNTUOM"]=="mg"),"AMOUNT"]*=1000
inputs_small_3.loc[(inputs_small_3["ITEMID"]==221744) & (inputs_small_3["AMOUNTUOM"]=="mg"),"AMOUNTUOM"]="mcg"
#Cleaning of the Pantoprazole (Protonix)
    #divide in two (drug shot or continuous treatment and create a new item id for the continuous version)
inputs_small_3.loc[(inputs_small_3["ITEMID"]==225910) & (inputs_small_3["ORDERCATEGORYDESCRIPTION"]=="Continuous Med"),"LABEL"]="Pantoprazole (Protonix) Continuous"
inputs_small_3.loc[(inputs_small_3["ITEMID"]==225910) & (inputs_small_3["ORDERCATEGORYDESCRIPTION"]=="Continuous Med"),"ITEMID"]=2217441
#remove the non dose from the drug shot version
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Pantoprazole (Protonix)") & (inputs_small_3["AMOUNTUOM"]!="dose")].index).copy()


#Verification that all input labels have the same units.
inputs_small_3.groupby("LABEL")["AMOUNTUOM"].value_counts()


# #### 2) Rates

#Cleaning of Dextrose 5%  (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["RATEUOM"]!="mL/hour")].index).copy()
#Cleaning of Magnesium Sulfate (Bolus)  (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Magnesium Sulfate (Bolus)") & (inputs_small_3["RATEUOM"]!="mL/hour")].index).copy()
#Cleaning of NaCl 0.9% (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="NaCl 0.9%") & (inputs_small_3["RATEUOM"]!="mL/hour")].index).copy()
#Cleaning of Piggyback (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Piggyback") & (inputs_small_3["RATEUOM"]!="mL/hour")].index).copy()
#Cleaning of Packed Red Bllod Cells (remove the non mL/hour)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Packed Red Blood Cells") & (inputs_small_3["RATEUOM"]!="mL/hour")].index).copy()

# ### Check for outliers
# 
# #### 1) In amounts

#Clean Albumin 5%
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Albumin 5%") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Albumin 5%") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Albumin 5%") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Albumin 5%") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Albumin 5%") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1

#Clean Calcium Gluconate
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1
#Remove entries with more 10 grams.
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Calcium Gluconate") & (inputs_small_3["AMOUNT"]>10)].index).copy()


#Clean Cefazolin
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1
#Remove entries with more than 2 doses amount.
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefazolin") & (inputs_small_3["AMOUNT"]>2)].index).copy()

#Clean Cefepime
#Remove the negative entries (they are anyway too large in the positive range as well.)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Cefepime") & (inputs_small_3["AMOUNT"]<0)].index).copy()

#Clean Ceftriaxone
#Remove the negative entries (they are anyway too large in the positive range as well.)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Ceftriaxone") & (inputs_small_3["AMOUNT"]<0)].index).copy()

#Clean Ciprofloxacin
#Remove the negative entries (they are anyway too large in the positive range as well.)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Ciprofloxacin") & (inputs_small_3["AMOUNT"]<0)].index).copy()

#Clean D5 1/2NS
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1

#Clean Dextrose 5%
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Dextrose 5%") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1

#Clean Famotidine (Pepcid)
#Remove the negative entries (they are anyway too large in the positive range as well.)
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Famotidine (Pepcid)") & (inputs_small_3["AMOUNT"]<0)].index).copy()
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Famotidine (Pepcid)") & (inputs_small_3["AMOUNT"]>1)].index).copy()

#Clean Fentanyl
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1

#Clean Fentanyl (Concentrate)
#Invert start time and end time
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNT"]<0),"STARTTIME"]=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]
a=inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNT"]<0),"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[(inputs_small_3["LABEL"]=="Fentanyl (Concentrate)") & (inputs_small_3["AMOUNT"]<0),"AMOUNT"]*=-1


#Check if all remaining negative values are linked to the swapping the start and end times.
inputs_small_3['STARTTIME']=pd.to_datetime(inputs_small_3["STARTTIME"], format='%Y-%m-%d %H:%M:%S')
inputs_small_3['ENDTIME']=pd.to_datetime(inputs_small_3["ENDTIME"], format='%Y-%m-%d %H:%M:%S')
inputs_small_3["DURATION"]=inputs_small_3['ENDTIME']-inputs_small_3['STARTTIME']

#Revert all the remaining negative values to the positive range.
a=inputs_small_3.loc[inputs_small_3["AMOUNT"]<0,"STARTTIME"]
inputs_small_3.loc[inputs_small_3["AMOUNT"]<0,"STARTTIME"]=inputs_small_3.loc[inputs_small_3["AMOUNT"]<0,"ENDTIME"]
a=inputs_small_3.loc[inputs_small_3["AMOUNT"]<0,"ENDTIME"]=a
#Positive rate
inputs_small_3.loc[inputs_small_3["AMOUNT"]<0,"AMOUNT"]*=-1
#Recompute the durations with the correct time stamps for start and end.
inputs_small_3["DURATION"]=inputs_small_3['ENDTIME']-inputs_small_3['STARTTIME']

#Clean Gastric Meds, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Gastric Meds") & (inputs_small_3["AMOUNT"]>5000)].index).copy()

#Clean Heparin Sodium, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Heparin Sodium") & (inputs_small_3["AMOUNT"]>50000)].index).copy()

#Clean Heparin Sodium (Prophylaxis), remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Heparin Sodium (Prophylaxis)") & (inputs_small_3["AMOUNT"]>2)].index).copy()

#Clean Hydralazine, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Hydralazine") & (inputs_small_3["AMOUNT"]>200)].index).copy()

#Clean Hydromorphone (Dilaudid), remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Hydromorphone (Dilaudid)") & (inputs_small_3["AMOUNT"]>500)].index).copy()

#Clean Insulin - Humalog, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Insulin - Humalog") & (inputs_small_3["AMOUNT"]>100)].index).copy()

#Clean Insulin - Regular, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Insulin - Regular") & (inputs_small_3["AMOUNT"]>1000)].index).copy()

#Clean Magnesium Sulfate, remove too large values
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="Magnesium Sulfate") & (inputs_small_3["AMOUNT"]>51)].index).copy()

# #### 2) In rates
#Clean D5 1/2NS Remove too large rates
inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]=="D5 1/2NS") & (inputs_small_3["RATE"]>1000)].index).copy()

#Remove all entries whose rate is more than 4 std away from the mean.
rate_desc=inputs_small_3.groupby("LABEL")["RATE"].describe()
name_list=list(rate_desc.loc[rate_desc["count"]!=0].index)
for label in name_list:
    inputs_small_3=inputs_small_3.drop(inputs_small_3.loc[(inputs_small_3["LABEL"]==label)&(inputs_small_3["RATE"]>(rate_desc.loc[label,"mean"]+4*rate_desc.loc[label,"std"]))].index).copy()


# ## We now split the entries which are spread in time.
# 
# We chose the duration window for the sampling. here we choose 30 minutes. So every entry which has a rate and with duration larger than 1 hour, we split it into fixed times injections.
# 
# 

#First we check if when there is a duration, the amount is matching.

#First check the /hours units
df_temp=inputs_small_3.loc[(inputs_small_3["RATE"].notnull()) & (inputs_small_3["RATEUOM"].str.contains("hour"))].copy()
df_temp["COMPUTED_AMOUNT"]=df_temp["RATE"]*(df_temp["DURATION"].dt.total_seconds()/3600)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["COMPUTED_AMOUNT"]-df_temp["AMOUNT"])>0.01)].index)==0) #OK

#Second check the /min units
df_temp=inputs_small_3.loc[(inputs_small_3["RATE"].notnull()) & (inputs_small_3["RATEUOM"].str.contains("mL/min"))].copy()
df_temp["COMPUTED_AMOUNT"]=df_temp["RATE"]*(df_temp["DURATION"].dt.total_seconds()/60)

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["COMPUTED_AMOUNT"]-df_temp["AMOUNT"])>0.01)].index)==0) #OK

#Third check the kg/min units
df_temp=inputs_small_3.loc[(inputs_small_3["RATE"].notnull()) & (inputs_small_3["RATEUOM"].str.contains("kg/min"))].copy()
df_temp["COMPUTED_AMOUNT"]=df_temp["RATE"]*(df_temp["DURATION"].dt.total_seconds()/60)*(df_temp["PATIENTWEIGHT"])

#Check with a 0.01 tolerance
assert(len(df_temp.loc[(abs(df_temp["COMPUTED_AMOUNT"]-1000*df_temp["AMOUNT"])>0.01)].index)==0) #OK


duration_split_hours=0.5
to_sec_fact=3600*duration_split_hours

#split data set in four.

#The first dataframe contains the entries with no rate but with extended duration inputs (over 0.5 hour)
df_temp1=inputs_small_3.loc[(inputs_small_3["DURATION"]>timedelta(hours=duration_split_hours)) & (inputs_small_3["RATE"].isnull())].copy().reset_index(drop=True)
#The second dataframe contains the entries with no rate and low duration entries (<0.5hour)
df_temp2=inputs_small_3.loc[(inputs_small_3["DURATION"]<=timedelta(hours=duration_split_hours)) & (inputs_small_3["RATE"].isnull())].copy().reset_index(drop=True)
#The third dataframe contains the entries with a rate and extended duration inputs (over 0.5 hour)
df_temp3=inputs_small_3.loc[(inputs_small_3["DURATION"]>timedelta(hours=duration_split_hours)) & (inputs_small_3["RATE"].notnull())].copy().reset_index(drop=True)
#The forth dataframe contains the entries with a rate and low duration entries (< 0.5 hour)
df_temp4=inputs_small_3.loc[(inputs_small_3["DURATION"]<=timedelta(hours=duration_split_hours)) & (inputs_small_3["RATE"].notnull())].copy().reset_index(drop=True)

#Check if split is complete
assert(len(df_temp1.index)+len(df_temp2.index)+len(df_temp3.index)+len(df_temp4.index)==len(inputs_small_3.index))


#We then process all of these dfs.
#In the first one, we need to duplicate the entries according to their duration and then divide each entry by the number of duplicates

#We duplicate the rows with the number bins for each injection
df_temp1["Repeat"]=np.ceil(df_temp1["DURATION"].dt.total_seconds()/to_sec_fact).astype(int)
df_new1=df_temp1.reindex(df_temp1.index.repeat(df_temp1["Repeat"]))
#We then create the admninistration time as a shifted version of the STARTTIME.
df_new1["CHARTTIME"]=df_new1.groupby(level=0)['STARTTIME'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new1["AMOUNT"]=df_new1["AMOUNT"]/df_new1["Repeat"]


# In the third one, we do the same
#We duplicate the rows with the number bins for each injection
df_temp3["Repeat"]=np.ceil(df_temp3["DURATION"].dt.total_seconds()/to_sec_fact).astype(int)
df_new3=df_temp3.reindex(df_temp3.index.repeat(df_temp3["Repeat"]))
#We then create the admninistration time as a shifted version of the STARTTIME.
df_new3["CHARTTIME"]=df_new3.groupby(level=0)['STARTTIME'].transform(lambda x: pd.date_range(start=x.iat[0],freq=str(60*duration_split_hours)+'min',periods=len(x)))
#We divide each entry by the number of repeats
df_new3["AMOUNT"]=df_new3["AMOUNT"]/df_new3["Repeat"]

df_temp2["CHARTTIME"]=df_temp2["STARTTIME"]
df_temp4["CHARTTIME"]=df_temp4["STARTTIME"]


#Eventually, we merge all 4splits into one.
inputs_small_4=df_new1.append([df_temp2,df_new3,df_temp4])
#The result is a dataset with discrete inputs for each treatment.

#Again, we remove all the observations that are more than 5std away from the mean.

amount_desc=inputs_small_4.groupby("LABEL")["AMOUNT"].describe()
name_list=list(amount_desc.loc[amount_desc["count"]!=0].index)
for label in name_list:
    inputs_small_4=inputs_small_4.drop(inputs_small_4.loc[(inputs_small_4["LABEL"]==label)&(inputs_small_4["AMOUNT"]>(amount_desc.loc[label,"mean"]+5*amount_desc.loc[label,"std"]))].index).copy()

inputs_small_4.to_csv(f"{file_path}INPUTS_processed.csv")
print(f'{file_path}INPUTS_processed.csv saved')