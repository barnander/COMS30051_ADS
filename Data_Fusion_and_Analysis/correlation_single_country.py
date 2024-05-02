
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Filter data to keep records from 1991 onwards
trade_register_data = trade_register_data[trade_register_data['Year of order'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]


country = 'Israel'

# Aggregating arms imports data by country and year for Afghanistan
arms_aggregated = trade_register_data[trade_register_data['Recipient'] == country].groupby(['Recipient', 'Year of order']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()

# Renaming columns for clarity
arms_aggregated.rename(columns={'Recipient': 'country', 'Year of order': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

# Rename columns in terrorism data to match
yearly_agg_data.rename(columns={'country_txt': 'country', 'iyear': 'year'}, inplace=True)

# Merge the datasets on 'country' and 'year'
merged_data = pd.merge(yearly_agg_data[yearly_agg_data['country'] == country], arms_aggregated, how='left', on=['country', 'year'])

# Replace NaN values with 0 for arms data
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Calculate correlation matrix
correlation_matrix = merged_data[['No_Incidents', 'No_death', 'total_arms_delivered', 'total_arms_value']].corr()

# Plot heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Terrorism Metrics and Arms Imports in Afghanistan')
plt.show()

# Create time-lagged versions of the arms imports data for one-year and two-year lags
merged_data['total_arms_delivered_lag1'] = merged_data['total_arms_delivered'].shift(1)
merged_data['total_arms_value_lag1'] = merged_data['total_arms_value'].shift(1)
# merged_data['total_arms_delivered_lag2'] = merged_data['total_arms_delivered'].shift(2)
# merged_data['total_arms_value_lag2'] = merged_data['total_arms_value'].shift(2)

# Replace NaN values in the new lagged columns with 0
merged_data.fillna({'total_arms_delivered_lag1': 0, 'total_arms_value_lag1': 0, 'total_arms_delivered_lag2': 0, 'total_arms_value_lag2': 0}, inplace=True)

# Calculate correlation matrix including the lagged variables
correlation_matrix_lagged = merged_data[['No_Incidents', 'No_death', 'total_arms_delivered_lag1', 'total_arms_value_lag1']].corr()

# Plot heatmap of the correlation matrix with lagged values
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_lagged, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix with Lagged Arms Imports in Afghanistan')
plt.show()
