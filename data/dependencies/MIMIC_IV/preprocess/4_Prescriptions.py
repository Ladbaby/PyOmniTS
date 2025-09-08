import sys

import pandas as pd

file_path = sys.argv[1]

adm = pd.read_csv(f"{file_path}processed/Admissions_processed.csv")

# only choose previously selected admission ids
presc=pd.read_csv(f"{file_path}hosp/prescriptions.csv.gz")
adm_ids=list(adm["hadm_id"])
presc=presc.loc[presc["hadm_id"].isin(adm_ids)]

print(f"Patients after admission id filter: {presc['subject_id'].nunique()}")

#Select entries whose drug name is in the list from the paper.
drugs_list=["Acetaminophen", "Aspirin","Bisacodyl","Insulin","Heparin","Docusate Sodium","D5W","Humulin-R Insulin","Potassium Chloride","Magnesium Sulfate","Metoprolol Tartrate","Sodium Chloride 0.9%  Flush","Pantoprazole"]
presc2=presc.loc[presc["drug"].isin(drugs_list)]

print(f"Patients after drug name filter: {presc2['subject_id'].nunique()}")

#Units correction
presc2=presc2.drop(presc2.loc[presc2["dose_unit_rx"].isnull()].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Acetaminophen")&(presc2["dose_unit_rx"]!="mg")].index).copy()
presc2.loc[(presc2["drug"]=="D5W")&(presc2["dose_unit_rx"]=="ml"),"dose_unit_rx"]="mL"
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="D5W")&(presc2["dose_unit_rx"]!="mL")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Heparin")&(presc2["dose_unit_rx"]!="UNIT")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Insulin")&(presc2["dose_unit_rx"]!="UNIT")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Magnesium Sulfate")&(presc2["dose_unit_rx"]!="gm")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Potassium Chloride")&(presc2["dose_unit_rx"]!="mEq")].index).copy()
presc2.loc[(presc2["drug"]=="Sodium Chloride 0.9%  Flush")&(presc2["dose_unit_rx"]=="ml"),"dose_unit_rx"]="mL"
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Bisacodyl")&(presc2["dose_unit_rx"]!="mg")].index).copy()
presc2=presc2.drop(presc2.loc[(presc2["drug"]=="Pantoprazole")&(presc2["dose_unit_rx"]!="mg")].index).copy()

#To avoid confounding labels with labels from other tables, we add "drug" to the name
presc2['charttime']=pd.to_datetime(presc2["starttime"], format='%Y-%m-%d %H:%M:%S')
presc2["drug"]=presc2["drug"]+" Drug"

presc2.to_csv(f"{file_path}processed/PRESCRIPTIONS_processed.csv")
print(f'{file_path}processed/PRESCRIPTIONS_processed.csv saved')