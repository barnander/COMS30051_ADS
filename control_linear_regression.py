import pandas as pd
from linearmodels import PanelOLS
import statsmodels.api as sm

# Load the data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/all_extra_data.csv')

# Prepare and merge the data
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index().rename(columns={'Recipient': 'Country', 'Year of order': 'Year'})

# Renaming columns for merging in extra_data
extra_data.rename(columns={'Country Name_x': 'Country', 'year': 'Year'}, inplace=True)

# Merge datasets
yearly_agg_data.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Step 3: Clean the Data
merged_data.dropna(subset=['No_Incidents', 'SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy'], inplace=True)

# Set the 'Country' and 'Year' columns as the index
merged_data.set_index(['Country', 'Year'], inplace=True)

# Now, proceed with fitting the model
# Step 5: Fit the Linear Regression Model
exog_vars = ['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']
endog = merged_data['No_Incidents']

for var in exog_vars:
    # Exclude one variable at a time
    excluded_var = exog_vars.copy()
    excluded_var.remove(var)
    
    # Select exogenous variables
    exog = merged_data[excluded_var]
    
    # Add a constant term for the intercept
    exog = sm.add_constant(exog)
    
    try:
        # Fit the model
        model = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Print the results
        print(f"Excluded variable: {var}")
        print(results)
        print("\n")
    except ValueError as e:
        print(f"Error fitting model with excluded variable {var}: {e}")
