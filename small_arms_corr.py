#%% Small arms data base correlation
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import pycountry
#%%
sa_data = pd.read_csv("C:\\Users\\Basil\\Downloads\\small_arms_FINAL.csv")
gtd_agg = pd.read_csv("yearly_agg.csv")
# %% aggregate transfer value by year for each country
# make float
sa_data = sa_data[sa_data['Value'] != 'Not Known']
sa_data['Value'] = sa_data['Value'].astype(float)

#create group object of df per year per country
grouped = sa_data.groupby(['Year', 'Reporter_Name'])
#sum the transfer values for each group
sa_agg = grouped.agg({'Value': 'sum'}).reset_index()
# %%
gtd_agg.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
sa_agg.rename(columns={'Reporter_Name': 'Country'}, inplace=True)

# %%
# merge the two dataframes
merged = pd.merge(gtd_agg, sa_agg, on=['Country', 'Year'], how='outer')
merged.fillna(value = 0,inplace = True)

#%% Take 1992 onwards
merged = merged[merged['Year'] >= 1992]

# %% 
countries = merged["Country"].unique()
corr_value = {country:0 for country in countries}
for country in merged["Country"].unique():
    country_data = merged[merged['Country'] == country] 
    num_el = country_data.shape[0]
    if num_el < 2:
        continue
    corr_i = country_data[["Value","No_injury_or_death"]].corr().loc["Value"]["No_injury_or_death"]
    corr_value[country] = [corr_i,num_el]
# %% find differences in country lists
sa_countries = sa_agg['Country'].unique()
gtd_countries = gtd_agg['Country'].unique()
diff_sa = set(sa_countries) - set(gtd_countries)
diff_gtd = set(gtd_countries) - set(sa_countries)


def get_count_code(country_name):
    try:
        code = pycountry.countries.lookup(country_name)
        return code.alpha_3
    except LookupError:
        return None

sa_agg['Country_code'] = sa_agg['Country'].apply(get_count_code)
gtd_agg['Country_code'] = gtd_agg['Country'].apply(get_count_code)


sa_country_codes = sa_agg['Country_code'].unique()
gtd_country_codes = gtd_agg['Country_code'].unique()


# %%
