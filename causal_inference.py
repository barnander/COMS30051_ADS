
from statsmodels.tsa.stattools import grangercausalitytests

# For simplicity, let's demonstrate the Granger causality test for a specific country.
# Here, we select a country with significant entries. We will use Afghanistan as an example.

# Filter data for Afghanistan
afghanistan_data = merged_data[merged_data['country'] == 'Afghanistan']

# Ensure the data is sorted by year for the time-series causality test
afghanistan_data = afghanistan_data.sort_values(by='year')

# Check for any missing years and fill if necessary (simple forward fill for demonstration)
afghanistan_data = afghanistan_data.fillna(method='ffill')

# Running Granger Causality Test
# We test the null hypothesis that 'total_arms_delivered' does not Granger-cause 'No_Incidents'
granger_result_incidents = grangercausalitytests(afghanistan_data[['No_Incidents', 'total_arms_delivered']], maxlag=2, verbose=True)

# Test if 'total_arms_delivered' does not Granger-cause 'No_death'
granger_result_deaths = grangercausalitytests(afghanistan_data[['No_death', 'total_arms_delivered']], maxlag=2, verbose=True)

# The output will provide F-test and p-values for each lag to determine if past values of arms deliveries predict future terrorism metrics

