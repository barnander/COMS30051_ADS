import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# Handling missing values
merged_data.fillna(merged_data.mean(numeric_only=True), inplace=True)

# Select features and target
X = merged_data[['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]
y = merged_data['No_Incidents']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

import matplotlib.pyplot as plt
import seaborn as sns

# Load the necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume 'merged_data' is already loaded and contains 'SIPRI TIV for total order' and 'No_Incidents'
# If not loaded, you would need to load and prepare your data similar to previous steps.

# It's important to ensure that both 'SIPRI TIV for total order' and 'No_Incidents' are in numeric format
merged_data['SIPRI TIV for total order'] = pd.to_numeric(merged_data['SIPRI TIV for total order'], errors='coerce')
merged_data['No_Incidents'] = pd.to_numeric(merged_data['No_Incidents'], errors='coerce')

# Drop rows where any of these two columns might have NaN values after conversion (if any)
merged_data.dropna(subset=['SIPRI TIV for total order', 'No_Incidents'], inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SIPRI TIV for total order', y='No_Incidents', data=merged_data)

# Adding a regression line to the scatter plot
sns.regplot(x='SIPRI TIV for total order', y='No_Incidents', data=merged_data, scatter=False, color='red')

plt.title('Relationship between Arms Imports and Terrorist Incidents')
plt.xlabel('Total Value of Arms Imports (TIV)')
plt.ylabel('Number of Terrorist Incidents')
plt.grid(True)
plt.show()
