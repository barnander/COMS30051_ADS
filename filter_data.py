#%%
#Filter data from main data_set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
gtd_data = pd.read_excel("globalterrorismdb_0522dist.xlsx")

#%%
gtd_data.columns

#%% prepare data filtering 
def id_to_txt_dict(cat):
    cat_ids = gtd_data[cat].unique()
    cat_txts = gtd_data[cat + "_txt"].unique()
    id_to_txt = {cat_ids[i]: cat_txts[i] for i in range(len(cat_ids))}
    return id_to_txt

def make_binary_column(cats, value):
    idx = (gtd_data[cats] == value).any(axis=1)
    return idx



#%%
# Filter data for events with at least an injury or a death
gtd_data['injury_or_death'] = gtd_data.apply(lambda x: x['nkill'] > 0 or x['nwound'] > 0, axis=1)

# Filter data for events with at least a death
gtd_data['death'] = gtd_data['nkill'] > 0

#Create list of categories to count the number of incidents for each category



weap_cats = ['weaptype1', 'weaptype2', 'weaptype3']
#Filter data for events where firearms were used
gtd_data['firearms'] = (gtd_data[weap_cats] == 5).any(axis = 1)

#Filter data for events where explosives were used
gtd_data['explosives'] = (gtd_data[weap_cats] == 6).any(axis = 1)

#Filter data for events where incendiary devices were used
gtd_data['incendiary'] = (gtd_data[weap_cats] == 8).any(axis = 1)


#Filter data for target types
targ_cats = ["targtype" + str(i) for i in range(1, 4)]
target_types = gtd_data['targtype1'].unique()
targ_dict = id_to_txt_dict("targtype1")
targ_names = [targ_dict[target_type] for target_type in target_types]
for target_type in target_types:
    gtd_data[targ_dict[target_type]] = (gtd_data[targ_cats] == target_type).any(axis = 1)


#Filter data for attack types
attack_cats = ["attacktype" + str(i) for i in range(1, 4)]
attack_types = gtd_data['attacktype1'].unique()
attack_dict = id_to_txt_dict("attacktype1")
attack_names = [attack_dict[attack_type] for attack_type in attack_types]

for attack_type in attack_types:
    gtd_data[attack_dict[attack_type]] = (gtd_data[attack_cats] == attack_type).any(axis = 1)

#%% Group by country and year
grouped = gtd_data.groupby(['country_txt', 'iyear'])

# Aggregate the required data
result_df = grouped.agg(
    total_incidents=pd.NamedAgg(column='iyear', aggfunc='size'),
    incidents_with_injury_or_death=pd.NamedAgg(column='injury_or_death', aggfunc='sum'),
    incidents_with_death=pd.NamedAgg(column='death', aggfunc='sum'),
    total_injuries=pd.NamedAgg(column='nwound', aggfunc='sum'),
    total_deaths=pd.NamedAgg(column='nkill', aggfunc='sum'),
    firearms=pd.NamedAgg(column='firearms', aggfunc='sum'),
    explosives=pd.NamedAgg(column='explosives', aggfunc='sum'),
    incendiary=pd.NamedAgg(column='incendiary', aggfunc='sum'),
).reset_index()


#%%
sum_columns = ["injury_or_death", "death", "nkill","nwound","firearms","explosives","incendiary",*targ_names,*attack_names]
# Performing the groupby and aggregation
result_df = gtd_data.groupby(['country_txt','iyear']).agg(
    No_Incidents=('iyear', 'size'),  # Counting the number of occurrences in each group
    **{f'No_{col}': (col, 'sum') for col in sum_columns})
    

#%%
result_df.to_csv("yearly_agg.csv")
    





# %%
