import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv', low_memory=False)

# Convert 'Units' to numeric, ignoring errors to skip non-numeric values
small_arms_data['Units'] = pd.to_numeric(small_arms_data['Units'], errors='coerce')

# Filter out data before 1992
small_arms_data = small_arms_data[small_arms_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]

# Aggregate total units imported globally per year
global_arms_imports_aggregated = small_arms_data.groupby('Year').agg(
    total_units_imported=pd.NamedAgg(column='Units', aggfunc='sum')
).reset_index()

# Rename 'Year' to 'iyear' to match the yearly_agg_data column name for merging
global_arms_imports_aggregated.rename(columns={'Year': 'iyear'}, inplace=True)

# Create lagged arms import data
global_arms_imports_aggregated['total_units_imported_lag1'] = global_arms_imports_aggregated['total_units_imported'].shift(1)
global_arms_imports_aggregated['total_units_imported_lag2'] = global_arms_imports_aggregated['total_units_imported'].shift(2)

# Aggregate terror incidents and deaths data
global_terror_aggregated = yearly_agg_data.groupby('iyear').agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum'),
    total_deaths=pd.NamedAgg(column='No_death', aggfunc='sum')
).reset_index()

# Merge the datasets on the year for lagged analysis
merged_global_data_lagged = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# Calculate Pearson correlation coefficient for the lagged data
correlation_lagged = merged_global_data_lagged[['total_incidents', 'total_deaths', 'total_units_imported_lag1', 'total_units_imported_lag2']].corr()

# Plotting the correlation matrix as a heatmap for the lagged correlation
plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_lagged, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, annot_kws={"size": 14})
ax.set_xticklabels(['No. of Incidents', 'No. of Deaths', 'Imports lag 1', 'Imports lag 2'], fontsize=12, rotation=45)
ax.set_yticklabels(['No. of Incidents', 'No. of Deaths', 'Imports lag 1', 'Imports lag 2'], fontsize=12, rotation=45)
plt.show()

# Function to calculate and print Pearson correlation coefficient and p-value
def calculate_pearson_correlation(df, col1, col2):
    # Drop any rows that contain NaN in either column to ensure equal lengths
    filtered_df = df[[col1, col2]].dropna()
    correlation, p_value = pearsonr(filtered_df[col1], filtered_df[col2])
    print(f"Pearson correlation coefficient between {col1} and {col2}: {correlation:.3f}")
    print(f"P-value of the correlation: {p_value:.3g}\n")

# Example usage with the merged and lagged global data
calculate_pearson_correlation(merged_global_data_lagged, 'total_incidents', 'total_units_imported_lag1')
calculate_pearson_correlation(merged_global_data_lagged, 'total_deaths', 'total_units_imported_lag1')
calculate_pearson_correlation(merged_global_data_lagged, 'total_incidents', 'total_units_imported_lag2')
calculate_pearson_correlation(merged_global_data_lagged, 'total_deaths', 'total_units_imported_lag2')
