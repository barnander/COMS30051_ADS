import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s, f  # Import GAM and spline functions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/controls.csv')

# Prepare and merge the data
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index().rename(columns={'Recipient': 'Country', 'Year of order': 'Year'})

extra_data.rename(columns={'Country Name_x': 'Country', 'year': 'Year'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

merged_data.update(extra_data[['gdp', 'population', 'hdi', 'v2x_polyarchy']])
merged_data.fillna(merged_data.mean(numeric_only=True), inplace=True)

# Log transformation to address non-linear relationships
features_to_transform = ['SIPRI TIV for total order', 'gdp', 'population']
merged_data[features_to_transform] = merged_data[features_to_transform].apply(lambda x: np.log(x + 1))
merged_data['No_Incidents'] = np.log(merged_data['No_Incidents'] + 1)

# Select features and target
X = merged_data[['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]
y = merged_data['No_Incidents']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# GAM Model
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).gridsearch(X_train, y_train)
y_pred = gam.predict(X_test)

# Evaluate the model
mse = mean_squared_error(np.exp(y_test) - 1, np.exp(y_pred) - 1)  # Reverse log transformation for MSE
r2 = r2_score(np.exp(y_test) - 1, np.exp(y_pred) - 1)  # Reverse log transformation for R2

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Plotting the relationship using GAM
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
titles = ['SIPRI TIV for total order', 'GDP', 'Population', 'HDI', 'Polyarchy']
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], color='r', ls='--')
    ax.set_title(titles[i])
plt.tight_layout()
plt.show()
