import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

# Load data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Filter data to keep records from 1991 onwards
trade_register_data = trade_register_data[trade_register_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['Year'] >= 1991]

# Aggregating arms imports data by country and year
arms_aggregated = trade_register_data.groupby(['Country', 'Year']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()

# Renaming columns for clarity
arms_aggregated.rename(columns={'Country': 'country', 'Year': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

# Rename columns in yearly_agg_data for merging
yearly_agg_data.rename(columns={'Country': 'country', 'Year': 'year'}, inplace=True)

# Merge datasets
merged_data = pd.merge(yearly_agg_data, arms_aggregated, how='left', on=['country', 'year'])

# Replace NaN values with 0
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Ensure data is sorted by year for each country (important for time series analysis)
merged_data.sort_values(by=['country', 'year'], inplace=True)

# Function to perform Granger Causality Test
def test_granger_causality(df, maxlag):
    test_results = grangercausalitytests(df, maxlag=maxlag, verbose=True)
    return test_results

# Selecting data for Granger Causality Test (e.g., total incidents and total arms value lagged)
data_for_test = merged_data.pivot(index='year', columns='country', values=['No_Incidents', 'total_arms_value']).dropna()

# Example Granger Causality Test (change variables as needed)
for country in data_for_test['No_Incidents'].columns:
    print(f"Testing for {country}:")
    df_test = data_for_test[['No_Incidents', 'total_arms_value']][country]
    test_granger_causality(df_test, maxlag=2)

# This block assumes that you've prepared your dataset correctly for the test.
# Adjust the pivot and column selections as per your data structure and analysis needs.
