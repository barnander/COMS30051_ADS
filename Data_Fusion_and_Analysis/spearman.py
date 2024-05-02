
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr

# Load data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Filter data to keep records from 1991 onwards
trade_register_data = trade_register_data[trade_register_data['Year of order'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]

# Aggregating arms imports data by country and year
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()

# Renaming columns for clarity
arms_aggregated.rename(columns={'Recipient': 'country', 'Year of order': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

# Rename columns in yearly_agg_data for merging
yearly_agg_data.rename(columns={'country_txt': 'country', 'iyear': 'year'}, inplace=True)

# Merge datasets
merged_data = pd.merge(yearly_agg_data, arms_aggregated, how='left', on=['country', 'year'])

# Replace NaN values with 0
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Calculate Spearman correlation matrix for the merged data
correlation_matrix = merged_data[['No_Incidents', 'No_death', 'total_arms_delivered', 'total_arms_value']].corr(method='spearman')

# Plot heatmap of the Spearman correlation matrix
plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, annot_kws={"size": 14})
ax.set_xticklabels(['No. of Incidents', 'No. of Deaths', 'Total Arms Delivered', 'TIV'], fontsize=12)
ax.set_yticklabels(['No. of Incidents', 'No. of Deaths', 'Total Arms Delivered', 'TIV'], fontsize=12)
plt.show()

# Function to calculate Spearman correlation and p-value
def calculate_spearman_correlation(df, col1, col2):
    corr, p_value = spearmanr(df[col1], df[col2])
    print(f"Spearman correlation coefficient between {col1} and {col2}: {corr:.3f}")
    print(f"P-value of the correlation: {p_value}\n")

# Calculate and print Spearman correlation and p-values
calculate_spearman_correlation(merged_data, 'No_Incidents', 'No_death')
calculate_spearman_correlation(merged_data, 'No_Incidents', 'total_arms_delivered')
calculate_spearman_correlation(merged_data, 'No_Incidents', 'total_arms_value')
calculate_spearman_correlation(merged_data, 'No_death', 'total_arms_delivered')
calculate_spearman_correlation(merged_data, 'No_death', 'total_arms_value')
calculate_spearman_correlation(merged_data, 'total_arms_delivered', 'total_arms_value')
