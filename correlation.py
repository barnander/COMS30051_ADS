import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Aggregating arms imports data by country and year
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()

# Renaming columns for clarity and to prevent conflicts in merging
arms_aggregated.rename(columns={'Recipient': 'country', 'Year of order': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

# Now, let's merge this with the terrorism data on 'country' and 'year'
yearly_agg_data.rename(columns={'country_txt': 'country', 'iyear': 'year'}, inplace=True)
merged_data = pd.merge(yearly_agg_data, arms_aggregated, how='left', on=['country', 'year'])

# Replace NaN values with 0 for arms data (assuming NaN means no arms were delivered in those years/countries)
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Calculate correlation matrix for the merged data
correlation_matrix = merged_data[['No_Incidents', 'No_death', 'total_arms_delivered', 'total_arms_value']].corr()

correlation_matrix


# Set up the matplotlib figure for the correlation heatmap
plt.figure(figsize=(10, 8))

# Plot heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Terrorism Metrics and Arms Imports')
plt.show()


# Create time-lagged versions of the arms imports data for one-year and two-year lags
merged_data['total_arms_delivered_lag1'] = merged_data.groupby('country')['total_arms_delivered'].shift(1)
merged_data['total_arms_value_lag1'] = merged_data.groupby('country')['total_arms_value'].shift(1)
merged_data['total_arms_delivered_lag2'] = merged_data.groupby('country')['total_arms_delivered'].shift(2)
merged_data['total_arms_value_lag2'] = merged_data.groupby('country')['total_arms_value'].shift(2)

# Replace NaN values in the new lagged columns with 0 (assuming no data means no arms were delivered in those years)
merged_data.fillna({'total_arms_delivered_lag1': 0, 'total_arms_value_lag1': 0, 'total_arms_delivered_lag2': 0, 'total_arms_value_lag2': 0}, inplace=True)

# Calculate correlation matrix including the lagged variables
correlation_matrix_lagged = merged_data[['No_Incidents', 'No_death', 'total_arms_delivered_lag1', 'total_arms_value_lag1', 'total_arms_delivered_lag2', 'total_arms_value_lag2']].corr()

# Set up the matplotlib figure for the correlation heatmap including lagged values
plt.figure(figsize=(12, 10))

# Plot heatmap of the correlation matrix with lagged values
sns.heatmap(correlation_matrix_lagged, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix with Lagged Arms Imports')
plt.show()
