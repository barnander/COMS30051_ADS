import pandas as pd
from functools import reduce

# Load the data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/controls.csv')

# Rename and convert types in trade_register_data
trade_register_data.rename(columns={'Year of order': 'Year', 'Recipient': 'Country'}, inplace=True)
trade_register_data['Year'] = trade_register_data['Year'].astype(int)  # Convert Year to int

# Rename and convert types in yearly_agg_data
yearly_agg_data.rename(columns={'iyear': 'Year', 'country_txt': 'Country'}, inplace=True)
yearly_agg_data['Year'] = yearly_agg_data['Year'].astype(int)  # Convert Year to int

# Rename and convert types in extra_data
extra_data.rename(columns={'year': 'Year', 'Country Name_x': 'Country'}, inplace=True)
extra_data['Year'] = extra_data['Year'].astype(int)  # Convert Year to int

# Merge the datasets
dfs = [yearly_agg_data, trade_register_data, extra_data]
merged_data = reduce(lambda left, right: pd.merge(left, right, on=['Country', 'Year'], how='outer'), dfs)

# Fill missing values for numerical data with the mean
numerical_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numerical_cols] = merged_data[numerical_cols].fillna(merged_data[numerical_cols].mean())

# Check the head of the merged DataFrame to ensure it looks correct
print(merged_data.head())

# Additional analysis can be added here, such as statistical testing or machine learning modeling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Assuming 'merged_data' is your DataFrame

# Filter out unnecessary columns or fill missing values as needed
merged_data.dropna(subset=['No_Incidents', 'SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy'], inplace=True)

# Select relevant features
features = ['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']
target = 'No_Incidents'

# Prepare data for modeling
X = merged_data[features]
y = merged_data[target]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)  # Corrected to use keyword arguments
plt.xlabel('Actual Incidents')
plt.ylabel('Predicted Incidents')
plt.title('Actual vs. Predicted Terrorist Incidents')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Line of equality
plt.show()

