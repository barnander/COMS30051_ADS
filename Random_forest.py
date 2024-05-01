import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Load the dataset
merged_data = pd.read_csv('merged_data.csv')

# Select only numeric columns for calculating mean
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())

# Apply log transformation to features and target to reduce skewness and improve model performance
features_to_transform = ['gdp', 'population', 'SIPRI TIV for total order']  # Specify the features you want to transform
merged_data[features_to_transform] = merged_data[features_to_transform].apply(lambda x: np.log(x + 1))
merged_data['No_Incidents'] = np.log(merged_data['No_Incidents'] + 1)  # Transforming the target variable

# Select features and target
features = ['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']
target = 'No_Incidents'

X = merged_data[features]
y = merged_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(np.exp(y_test) - 1, np.exp(y_pred) - 1)  # Reverse the log transformation for MSE calculation
r2 = r2_score(np.exp(y_test) - 1, np.exp(y_pred) - 1)  # Reverse the log transformation for R² calculation

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plot Feature Importances
feature_importances = pd.Series(model.feature_importances_, index=features)
feature_importances.nlargest(len(features)).plot(kind='barh')
plt.title('Feature Importances in Predicting Terrorism Incidents')
plt.show()

# Residual Plot
residuals = np.exp(y_test) - 1 - np.exp(y_pred) + 1
plt.scatter(np.exp(y_pred) - 1, residuals)
plt.hlines(y=0, xmin=(np.exp(y_pred) - 1).min(), xmax=(np.exp(y_pred) - 1).max(), colors='red')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Actual vs Predicted Values Plot
plt.scatter(np.exp(y_test) - 1, np.exp(y_pred) - 1, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # Diagonal line
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Partial Dependence Plot
fig, ax = plt.subplots(figsize=(15, 10))
PartialDependenceDisplay.from_estimator(model, X_train, features, ax=ax, grid_resolution=20)
plt.suptitle('Partial Dependence of Features')
plt.subplots_adjust(top=0.9)
plt.show()
