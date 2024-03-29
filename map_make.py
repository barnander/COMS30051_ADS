# -*- coding: utf-8 -*-
#%%
import folium
import pandas as pd
import numpy as np
#%% Load Data
#TODO choose relevant columns to speed up loading
print("Loading Data...")
#data = pd.read_excel("globalterrorismdb_0522dist.xlsx")
#data = data[["iyear", "imonth", "iday", "country_txt", "region_txt", "city", "latitude", "longitude", "nkill", "nwound", "summary", "gname", "motive"]]
data = pd.read_csv("GTD.csv")
#%% Get user input
#TODO add range of years and regions  
#TODO GUI
country = input("Enter a country: ")
assert country in data["country_txt"].unique(), "Country not found in data"
year = int(input("Enter a year: "))
assert year in data["iyear"].unique(), "Year not found in data"
# %% filter data
query_data = data[(data["country_txt"] == country) & (data["iyear"] == year)]
query_data = query_data.dropna(subset=["latitude", "longitude"])
# %% make markers for each attack
#TODO center map on country
map = folium.Map(tiles="cartodb dark_matter")
for i in range(len(query_data)):
    date_string = f"{query_data.iloc[i]['iday']}/{query_data.iloc[i]['imonth']}/{query_data.iloc[i]['iyear']}"
    tool_tip = date_string + f": \n{query_data.iloc[i]['gname']} attack: \n{int(query_data.iloc[i]['nkill'])} dead, {int(query_data.iloc[i]['nwound'])} wounded"
    folium.Circle(
        location=[query_data.iloc[i]["latitude"], query_data.iloc[i]["longitude"]],
        radius=10000,
        tooltip= tool_tip,
        color="red",
        fill=True,
        fill_color="red",
        popup= query_data.iloc[i]["summary"]
    ).add_to(map)
map
# %%
