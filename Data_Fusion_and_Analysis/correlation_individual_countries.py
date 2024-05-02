import pandas as pd

# Load data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Rename columns in datasets to facilitate merging
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()
arms_aggregated.rename(columns={'Recipient': 'country', 'Year of order': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

yearly_agg_data.rename(columns={'country_txt': 'country', 'iyear': 'year'}, inplace=True)

# Merge the datasets
merged_data = pd.merge(yearly_agg_data, arms_aggregated, how='left', on=['country', 'year'])
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Create time-lagged data
merged_data['total_arms_delivered_lag1'] = merged_data.groupby('country')['total_arms_delivered'].shift(1)
merged_data['total_arms_value_lag1'] = merged_data.groupby('country')['total_arms_value'].shift(1)
merged_data.fillna({'total_arms_delivered_lag1': 0, 'total_arms_value_lag1': 0}, inplace=True)

# Calculate and store correlation coefficients for each country
correlation_results = {}
for country in merged_data['country'].unique():
    country_data = merged_data[merged_data['country'] == country]
    if country_data.shape[0] < 2:  # Ensure there is enough data to calculate correlation
        continue
    correlation_matrix = country_data[['No_Incidents', 'total_arms_delivered_lag1']].corr()
    correlation_results[country] = correlation_matrix.loc['No_Incidents', 'total_arms_delivered_lag1']

# Convert the results to a DataFrame
results_df = pd.DataFrame(list(correlation_results.items()), columns=['Country', 'Correlation Coefficient'])

# Save to Excel
results_df.to_excel('/Users/jacovaneeden/Desktop/Metrics.xlsx', index=False)

print("Data saved successfully to '/Users/jacovaneeden/Desktop/Metrics.xlsx'")
