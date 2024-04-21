import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

# Convert 'Units' to numeric, ignoring errors to skip non-numeric values
small_arms_data['Units'] = pd.to_numeric(small_arms_data['Units'], errors='coerce')

# Aggregate total units imported globally per year
global_arms_imports_aggregated = small_arms_data.groupby('Year').agg(
    total_units_imported=pd.NamedAgg(column='Units', aggfunc='sum')
).reset_index()

# Rename 'Year' to 'iyear' to match the yearly_agg_data column name for merging
global_arms_imports_aggregated.rename(columns={'Year': 'iyear'}, inplace=True)

# Aggregate terror incidents data
global_terror_aggregated = yearly_agg_data.groupby('iyear').agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum')
).reset_index()

# Merge the datasets on the year
merged_global_data = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# Calculate Pearson correlation coefficient
correlation = merged_global_data[['total_units_imported', 'total_incidents']].corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))  # Set the figure size for better readability
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Between Arms Imports and Terror Incidents')
plt.show()

# Shift the terror incidents data by 1 year to create a 1-year lag
global_terror_aggregated['total_incidents_lagged'] = global_terror_aggregated['total_incidents'].shift(0)

# Merge the datasets on the year
merged_global_data = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# Calculate Pearson correlation coefficient for the lagged data
correlation_lagged = merged_global_data[['total_units_imported', 'total_incidents_lagged']].corr()

# Plotting the correlation matrix as a heatmap for the lagged correlation
plt.figure(figsize=(8, 6))  # Set the figure size for better readability
sns.heatmap(correlation_lagged, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Lagged Correlation Matrix Between Arms Imports and Terror Incidents')
plt.show()