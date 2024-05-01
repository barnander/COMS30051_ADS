
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

# Convert 'Units' to numeric, ignoring errors to skip non-numeric values
small_arms_data['Value'] = pd.to_numeric(small_arms_data['Value'], errors='coerce')

# Filter out data before 1992
small_arms_data = small_arms_data[small_arms_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]

# Ensure the column names for merging are consistent
small_arms_data.rename(columns={'Country': 'country', 'Year': 'iyear'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'country'}, inplace=True)

# Aggregate total units imported per country per year
country_arms_imports_aggregated = small_arms_data.groupby(['country', 'iyear']).agg(
    total_units_imported=pd.NamedAgg(column='Value', aggfunc='sum')
).reset_index()

# Aggregate terror incidents and deaths data per country per year
country_terror_aggregated = yearly_agg_data.groupby(['country', 'iyear']).agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum'),
    total_deaths=pd.NamedAgg(column='No_death', aggfunc='sum')
).reset_index()

# Merge the datasets on both country and year
merged_country_data = pd.merge(country_arms_imports_aggregated, country_terror_aggregated, on=['country', 'iyear'], how='inner')

# Performing linear regression
X = merged_country_data[['total_units_imported']]
y = merged_country_data['total_incidents']
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(y, X).fit()  # OLS regression
predictions = model.predict(X)  # make the predictions by the model

# Print out the statistics
print(model.summary())

# Plotting the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_units_imported', y='total_incidents', data=merged_country_data)
plt.plot(merged_country_data['total_units_imported'], predictions, color='red')  # Add the regression line
plt.title('Relationship between Total Arms Imported and Terrorist Incidents per Country per Year')
plt.xlabel('Total Arms Imported (Units)')
plt.ylabel('Total Terrorist Incidents')
plt.grid(True)
plt.show()
