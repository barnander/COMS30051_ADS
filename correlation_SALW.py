import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

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

# Aggregate terror incidents and deaths data
global_terror_aggregated = yearly_agg_data.groupby('iyear').agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum'),
    total_deaths=pd.NamedAgg(column='No_death', aggfunc='sum')  # Aggregating total deaths
).reset_index()

# Merge the datasets on the year
merged_global_data = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# Calculate Pearson correlation coefficient for the current year data
correlation = merged_global_data[['total_incidents', 'total_deaths', 'total_units_imported']].corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, annot_kws={"size": 14})
ax.set_xticklabels(['No. of Incidents', 'No. of Deaths', 'Total Units Imported'], fontsize=12)
ax.set_yticklabels(['No. of Incidents', 'No. of Deaths', 'Total Units Imported'], fontsize=12)
plt.show()

# # Shift the terror data by 1 year to create a 1-year lag for incidents and deaths
# global_terror_aggregated['total_incidents_lagged'] = global_terror_aggregated['total_incidents'].shift(1)
# global_terror_aggregated['total_deaths_lagged'] = global_terror_aggregated['total_deaths'].shift(1)

# # Merge the datasets on the year for lagged analysis
# merged_global_data_lagged = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# # Calculate Pearson correlation coefficient for the lagged data
# correlation_lagged = merged_global_data_lagged[['total_units_imported', 'total_incidents_lagged', 'total_deaths_lagged']].corr()

# # Plotting the correlation matrix as a heatmap for the lagged correlation
# plt.figure(figsize=(10, 8))  # Adjust the figure size for better readability
# sns.heatmap(correlation_lagged, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
# plt.title('Lagged Correlation Matrix Between Arms Imports, Terror Incidents, and Deaths')
# plt.show()
