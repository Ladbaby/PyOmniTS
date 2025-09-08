import sys

import pandas as pd

file_path = sys.argv[1]

adm = pd.read_csv(f"{file_path}processed/Admissions_processed.csv")

outputs = pd.read_csv(f'{file_path}icu/outputevents.csv.gz')

# only choose previously selected admission ids
adm_ids=list(adm["hadm_id"])
outputs=outputs.loc[outputs["hadm_id"].isin(adm_ids)]

print(f'Patients after admission id filter: {outputs["subject_id"].nunique()}')

# get item names
item_id=pd.read_csv(f'{file_path}icu/d_items.csv.gz')
item_id_1=item_id[["itemid","label"]]

outputs_2=pd.merge(outputs,item_id_1,on="itemid")
print(f'Patients after admission id filter: {outputs_2["subject_id"].nunique()}')

# take only the n most used items
n_best=15
pat_for_item=outputs_2.groupby("label")["subject_id"].nunique()
frequent_labels=pat_for_item.sort_values(ascending=False)[:n_best]
outputs_3=outputs_2.loc[outputs_2["label"].isin(list(frequent_labels.index))].copy()

print(f'Patients after high occurence filter: {outputs_3["subject_id"].nunique()}')

outputs_label_list=['Foley', 'Void', 'OR Urine', 'Chest Tube', 'Oral Gastric', 'Pre-Admission', 'TF Residual', 'OR EBL', 'Emesis', 'Nasogastric', 'Stool', 'Jackson Pratt', 'TF Residual Output', 'Fecal Bag', 'Straight Cath']
outputs_bis=outputs_2.loc[outputs_2["label"].isin(outputs_label_list)].copy()

print(f'Patients after label filter: {outputs_bis["subject_id"].nunique()}')

outputs_3=outputs_bis.copy()

outputs_3.to_csv(f"{file_path}processed/OUTPUTS_processed.csv")
print(f'{file_path}processed/OUTPUTS_processed.csv saved')