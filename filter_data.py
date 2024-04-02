#Filter data from main data_set
import pandas as pd
#TODO add region filter and other filters
#TODO add range of years
#TODO think about how to handle missing data
def filter_basic(data, country, year):
    query_data = data[(data["country_txt"] == country) & (data["iyear"] == year)]
    query_data = query_data.dropna(subset=["latitude", "longitude","nkill","nwound"])
    return query_data