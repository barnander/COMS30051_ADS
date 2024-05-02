
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

print(small_arms_data['Year'].unique())
print(yearly_agg_data['iyear'].unique())


# Load data (using dummy paths; replace with your actual paths)
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

# Convert 'Units' to numeric and aggregate by year
small_arms_data['Units'] = pd.to_numeric(small_arms_data['Units'], errors='coerce')
arms_yearly = small_arms_data.groupby('Year').agg({'Units': 'sum'}).rename(columns={'Units': 'total_units_imported'})

# Aggregate terror incidents by year
incidents_yearly = yearly_agg_data.groupby('iyear').agg({'No_Incidents': 'sum'}).rename(columns={'iyear': 'Year', 'No_Incidents': 'total_incidents'})

# Create a complete year range from the earliest year in either dataset to the latest
complete_years = pd.DataFrame({'Year': np.arange(min(small_arms_data['Year'].min(), yearly_agg_data['iyear'].min()), 
                                                 max(small_arms_data['Year'].max(), yearly_agg_data['iyear'].max()) + 1)})

# Merge data and ensure all data is in consecutive years for time-series analysis
data = pd.merge(arms_yearly, incidents_yearly, left_index=True, right_index=True)

# Check stationarity
result_arms = adfuller(data['total_units_imported'])
result_incidents = adfuller(data['total_incidents'])

# Print ADF Test results
print('ADF Statistic for Arms Imports:', result_arms[0])
print('p-value for Arms Imports:', result_arms[1])
print('ADF Statistic for Terror Incidents:', result_incidents[0])
print('p-value for Terror Incidents:', result_incidents[1])

# If the series are not stationary, differencing or other transformations may be required

# Perform Granger Causality Test
gc_results = grangercausalitytests(data[['total_incidents', 'total_units_imported']], maxlag=2, verbose=True)

# Visualize the data
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(data['total_units_imported'], label='Arms Imports')
plt.title('Annual Arms Imports')
plt.legend()

plt.subplot(1,2,2)
plt.plot(data['total_incidents'], label='Terror Incidents')
plt.title('Annual Terror Incidents')
plt.legend()
plt.tight_layout()
plt.show()
