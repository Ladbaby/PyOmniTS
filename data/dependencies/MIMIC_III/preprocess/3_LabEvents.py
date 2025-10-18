import sys

import pandas as pd

file_path = sys.argv[1]

adm=pd.read_csv(file_path+"Admissions_processed.csv")


# We now consider the labevents dataset. We select only the patients with the same criteria as above.
try:
    lab=pd.read_csv(file_path+"LABEVENTS.csv")
except:
    lab=pd.read_csv(file_path+"LABEVENTS.csv.gz", compression="gzip")

#Restrict the dataset to the previously selected admission ids only.
adm_ids=list(adm["HADM_ID"])
lab=lab.loc[lab["HADM_ID"].isin(adm_ids)]

print(f"Patients after admission id filter: {lab['SUBJECT_ID'].nunique()}")

# We load the D_ITEMS dataframe which contains the name of the ITEMID. And we merge both tables together.

#item_id
try:
    item_id=pd.read_csv(file_path+"D_LABITEMS.csv")
except:
    item_id=pd.read_csv(file_path+"D_LABITEMS.csv.gz", compression="gzip")
item_id_1=item_id[["ITEMID","LABEL"]]

#We merge the name of the item administrated.
lab2=pd.merge(lab,item_id_1,on="ITEMID")
print(f"Patients after item name merging: {lab2['SUBJECT_ID'].nunique()}")

n_best=150
#For each item, evaluate the number of patients who have been given this item.
pat_for_item=lab2.groupby("LABEL")["SUBJECT_ID"].nunique()
#Order by occurence and take the 20 best (the ones with the most patients)
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]

#Select only the time series with high occurence.
lab3=lab2.loc[lab2["LABEL"].isin(list(frequent_labels.index))].copy()

print(f"Patients after high occurence filter: {lab3['SUBJECT_ID'].nunique()}")

# ### Units Cleaning
# 
# #### 1) In amounts

#Correct the units
lab3.loc[lab3["LABEL"]=="Calculated Total CO2","VALUEUOM"]="mEq/L"
lab3.loc[lab3["LABEL"]=="PT","VALUEUOM"]="sec"
lab3.loc[lab3["LABEL"]=="pCO2","VALUEUOM"]="mm Hg"
lab3.loc[lab3["LABEL"]=="pH","VALUEUOM"]="units"
lab3.loc[lab3["LABEL"]=="pO2","VALUEUOM"]="mm Hg"


#Only select the subset that was used in the paper (only missing is INR(PT))
subset=["Albumin","Alanine Aminotransferase (ALT)","Alkaline Phosphatase","Anion Gap","Asparate Aminotransferase (AST)","Base Excess","Basophils","Bicarbonate","Bilirubin, Total","Calcium, Total","Calculated Total CO2","Chloride","Creatinine","Eosinophils","Glucose","Hematocrit","Hemoglobin",
"Lactate","Lymphocytes","MCH","MCHC","MCV","Magnesium","Monocytes","Neutrophils","PT","PTT","Phosphate","Platelet Count","Potassium","RDW","Red Blood Cells","Sodium","Specific Gravity","Urea Nitrogen","White Blood Cells","pCO2","pH","pO2"]

lab3=lab3.loc[lab3["LABEL"].isin(subset)].copy()


# ### Check for outliers
# 
# #### 1) In amounts
#Glucose : mettre -1 aux résultats négatifs et supprimer les autres entrées dont la valeur numérique est NaN.
lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"].isnull())&(lab3["VALUE"]=="NEG"),"VALUENUM"]=-1
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"].isnull())].index).copy()

#Retirer les entrées avec NaN aux values et valuenum
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"].isnull())&(lab3["VALUE"].isnull())].index).copy()

#Remove the remaining NAN Values
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"].isnull())].index).copy()

#Remove anion gaps lower than 0
lab3=lab3.drop(lab3.loc[(lab3["VALUENUM"]<0)&(lab3["LABEL"]=="Anion Gap")].index).copy()

#Remove BE <-50
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Base Excess")&(lab3["VALUENUM"]<-50)].index).copy()
#Remove BE >50
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Base Excess")&(lab3["VALUENUM"]>50)].index).copy()

#Remove high Hemoglobins
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Hemoglobin")&(lab3["VALUENUM"]>25)].index).copy()

#Clean some glucose entries
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"]>2000)&(lab3["HADM_ID"]==103500.0)].index).copy()
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Glucose")&(lab3["VALUENUM"]>2000)&(lab3["HADM_ID"]==117066.0)].index).copy()

#Clean toO high levels of Potassium
lab3=lab3.drop(lab3.loc[(lab3["LABEL"]=="Potassium")&(lab3["VALUENUM"]>30)].index).copy()

lab3.to_csv(f"{file_path}LAB_processed.csv")
print(f'{file_path}LAB_processed.csv saved')