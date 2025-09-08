import sys

import pandas as pd

file_path = sys.argv[1]

adm = pd.read_csv(f"{file_path}processed/Admissions_processed.csv")

df = pd.DataFrame()
for chunk in pd.read_csv(f"{file_path}hosp/labevents.csv.gz", chunksize=500000):
    adm_ids=list(adm["hadm_id"])
    chunk=chunk.loc[chunk["hadm_id"].isin(adm_ids)]
    df = df.append(chunk[["subject_id","hadm_id","charttime","valuenum","itemid"]])

# only choose previously selected admission ids.
print(f'Patients after admission id filter: {df["subject_id"].nunique()}')

# get item ids
item_id=pd.read_csv(f"{file_path}hosp/d_labitems.csv.gz")
item_id_1=item_id[["itemid","label"]]

# get names of administered items
lab2=pd.merge(df,item_id_1,on="itemid")
print(f'Patients after merging: {lab2["subject_id"].nunique()}')

# get only top 150 most used tests
n_best=150
pat_for_item=lab2.groupby("label")["subject_id"].nunique()
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]
lab3=lab2.loc[lab2["label"].isin(list(frequent_labels.index))].copy()

print(f'Patients after high occurence filter: {lab3["subject_id"].nunique()}')

# only select the subset that was used in the paper (only missing is INR(PT))
subset=["Albumin","Alanine Aminotransferase (ALT)","Alkaline Phosphatase","Anion Gap","Asparate Aminotransferase (AST)","Base Excess","Basophils","Bicarbonate","Bilirubin, Total","Calcium, Total","Calculated Total CO2","Chloride","Creatinine","Eosinophils","Glucose","Hematocrit","Hemoglobin",
"Lactate","Lymphocytes","MCH","MCV","Magnesium","Monocytes","Neutrophils","PT","PTT","Phosphate","Platelet Count","Potassium","RDW","Red Blood Cells","Sodium","Specific Gravity","Urea Nitrogen","White Blood Cells","pCO2","pH","pO2"]

lab3=lab3.loc[lab3["label"].isin(subset)].copy()

lab3.to_csv(f"{file_path}processed/LAB_processed.csv")
print(f'{file_path}processed/LAB_processed.csv saved')