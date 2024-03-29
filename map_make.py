# -*- coding: utf-8 -*-
#%%
import folium
import pandas as pd
import numpy as np
#%% Load Data
print("Loading Data...")
data = pd.read_excel("globalterrorismdb_0522dist.xlsx")
data = data[["iyear", "imonth", "iday", "country_txt", "region_txt", "city", "latitude", "longitude", "nkill", "nwound", "summary", "gname", "motive"]]

#%% Get user input
#TODO add range of years and regions  
#TODO GUI
country = input("Enter a country: ")
assert country in data["country_txt"].unique(), "Country not found in data"
year = int(input("Enter a year: "))
assert year in data["iyear"].unique(), "Year not found in data"

#%%
country = "United States"
year = 2001
# %% filter data
query_data = data[(data["country_txt"] == country) & (data["iyear"] == year)]
query_data = query_data.dropna(subset=["latitude", "longitude"])
# %% make markers for each attack
#TODO center map on country
map = folium.Map(tiles="cartodb dark_matter")
for i in range(len(query_data)):
    folium.Circle(
        location=[query_data.iloc[i]["latitude"], query_data.iloc[i]["longitude"]],
        radius=100,
        color="red",
        fill=True,
        fill_color="red",
        popup= query_data.iloc[i]["summary"]
    ).add_to(map)

# %%
