#%%
#Filter data from main data_set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter


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



weap_cats = ['weaptype' + str(i) for i in range(1, 5)]
#Filter data for events where firearms were used
gtd_data['firearms'] = (gtd_data[weap_cats] == 5).any(axis = 1)

#Filter data for events where explosives were used
gtd_data['explosives'] = (gtd_data[weap_cats] == 6).any(axis = 1)

#Filter data for events where incendiary devices were used
gtd_data['incendiary'] = (gtd_data[weap_cats] == 8).any(axis = 1)



#Add columns for weapon subtypes
weap_subtype_cats = ['weapsubtype' + str(i) for i in range(1, 5)]
weap_sub_types = gtd_data['weapsubtype1'].unique()
weap_sub_types = weap_sub_types[~np.isnan(weap_sub_types)]
weap_sub_dict = id_to_txt_dict("weapsubtype1")
weap_sub_names = [weap_sub_dict[weap_sub_type] for weap_sub_type in weap_sub_types]
for weap_sub_type in weap_sub_types:
    gtd_data[weap_sub_dict[weap_sub_type]] = (gtd_data[weap_subtype_cats] == weap_sub_type).any(axis = 1)


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
"""
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
"""

#%%
sum_columns = ["injury_or_death", "death", "nkill","nwound","firearms","explosives","incendiary",*targ_names,*attack_names,*weap_sub_names]
# Performing the groupby and aggregation
result_df = gtd_data.groupby(['country_txt','iyear']).agg(
    No_Incidents=('iyear', 'size'),  # Counting the number of occurrences in each group
    **{f'No_{col}': (col, 'sum') for col in sum_columns})
    

#%%
result_df.to_csv("yearly_agg.csv")
    

#%%
def process_text(text):
    stop_words = set(stopwords.words('english'))
    #remove specific, motive, attack, group, claimed, responsibility, unknown.
    stop_words.update(['motive', 'specific', 'attack', 'group', 'claimed', 'responsibility', 'unknown', 'however','stated','alleged','noted','incident','targeted','may','source','incident','carried','suspected','part','larger','trend','victim'])
    lemmatizer = WordNetLemmatizer()

    # Tokenize text
    words = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalpha()]
    return lemmatized_words
def count_words(df, col):
    flattened = [word for row in df[col] for word in row]
    return Counter(flattened)
# %% investigate motivation
#count occurences of words in motive
gtd_motives = gtd_data.dropna(subset = ['motive'], inplace = False)
gtd_motives['motive'] = gtd_motives['motive'].apply(lambda x: process_text(x))
#remove rows with empty motive
gtd_motives = gtd_motives[gtd_motives['motive'].apply(lambda x: len(x) > 0)]
#count occurences of words in motive
counted_words = count_words(gtd_motives, 'motive')
print(counted_words.most_common(20))
# %% filter for higest correlation countries
high_corr_countries = ['Montenegro','Burkina Faso','Czechoslavakia','Saudi Arabia','Bosnia-Herzegovina','Hungary','Nicauragua','Dominican Republic','Sierra Leone','Lithuania','Mali','Singapore','China','Niger','Switzerland']
high_corr_data = gtd_data[gtd_data['country_txt'].isin(high_corr_countries)]
# %% investigate motivation
#count occurences of words in motive
high_corr_data = high_corr_data.dropna(subset = ['motive'], inplace = False)
high_corr_data['motive'] = high_corr_data['motive'].apply(lambda x: process_text(x))
#remove rows with empty motive
high_corr_data = high_corr_data[high_corr_data['motive'].apply(lambda x: len(x) > 0)]
#count occurences of words in motive
counted_words = count_words(high_corr_data, 'motive')
print(counted_words.most_common(20))

# %% filter for low correlation countries
low_corr_countries = ['Libya','Azerbaijan','Yugoslavia','Norway','Tajikistan','Yemen','Jamaica','Bahamas','Paraguay','Mauritania','Armenia','North Yemen','Syria','Uruguay','Soviet Union']
low_corr_data = gtd_data[gtd_data['country_txt'].isin(low_corr_countries)]
# %% investigate motivation
low_corr_data = low_corr_data.dropna(subset = ['motive'], inplace = False)
low_corr_data['motive'] = low_corr_data['motive'].apply(lambda x: process_text(x))
#remove rows with empty motive
low_corr_data = low_corr_data[low_corr_data['motive'].apply(lambda x: len(x) > 0)]
#count occurences of words in motive
counted_words = count_words(low_corr_data, 'motive')
print(counted_words.most_common(20))

# %%
for country_name in low_corr_countries:
    country_data = gtd_motives[gtd_motives['country_txt'] == country_name]
    counted_words = count_words(country_data, 'motive')
    print(f"Most common words in {country_name}")
    print(counted_words.most_common(20))
# %%
