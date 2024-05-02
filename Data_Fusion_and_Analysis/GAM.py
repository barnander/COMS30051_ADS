#%% packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from pygam import LinearGAM, s
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#%%
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('all_extra_data.csv')

extra_data.rename(columns={'Country Name_x': 'Country', 'year': 'Year'}, inplace=True)
trade_register_data.rename(columns={'Recipient': 'Country', 'Year of order': 'Year'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
# Filter data to keep records from 1992 onwards
trade_register_data = trade_register_data[trade_register_data['Year'] >= 1992]
yearly_agg_data = yearly_agg_data[yearly_agg_data['Year'] >= 1992]
extra_data = extra_data[extra_data['Year'] >= 1992]

# Prepare and merge the data
arms_aggregated = trade_register_data.groupby(['Country', 'Year']).agg({
    'SIPRI TIV of delivered weapons': 'sum'
}).reset_index()

merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Fill missing values with 0 (assuming missing data implies no orders/incidents)
merged_data.fillna(0,inplace=True)
data = merged_data.copy()

# %%
data = data[data['population']>1e7]
data['SIPRI TIV of delivered weapons'] = np.log1p(data['SIPRI TIV of delivered weapons'])
#drop zero values of gdp, population, SIPRI TIV of delivered weapons and hdi
data = data[data['gdp'] > 0]
data = data[data['population'] > 0]
data = data[data['SIPRI TIV of delivered weapons'] > 0]
data = data[data['hdi'] > 0]
data = data[data['v2x_polyarchy'] > 0]
data = data[data['No_Incidents'] > 0]
data = data[data['No_death'] > 0]
#%%
data.columns
data['prop_firearms'] = data['No_firearms'] / data['No_Incidents']
data = data[['Country','Year','No_Incidents','No_death', 'prop_firearms', 'SIPRI TIV of delivered weapons', 'gdp', 'population',
       'hdi', 'v2x_polyarchy']]
# %% split train val test 
train_val,test = train_test_split(data, test_size=0.2, random_state=42)
train,val = train_test_split(train_val, test_size=0.2, random_state=42)

# %% write function to create generative additive model on the data for different target vars

def GAM(data, target, val = val):
    # create model
    y = data[target]
    X = data[['SIPRI TIV of delivered weapons', 'gdp', 'population',
       'hdi', 'v2x_polyarchy']]
    gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X, y)
    #predict on val
    X_pred = val[['SIPRI TIV of delivered weapons', 'gdp', 'population',
       'hdi', 'v2x_polyarchy']]
    y_pred = gam.predict(X_pred)
    for i, feature in enumerate(X.columns):
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        plt.figure()
        plt.title(f'Effect of {feature}')
        plt.scatter(X_pred[feature], y_pred)
        plt.plot(XX[:, i], pdep)
        plt.plot(XX[:, i], confi, c='r', ls='--')
        plt.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='r', alpha=0.1)
        plt.show()
    return gam
gam_death= GAM(train, 'No_death')
gam_death.summary()
# %%
gam_death.summary()
#import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Prepare your data
target = 'No_'
features = ['SIPRI TIV of delivered weapons', 'gdp', 'population', 'hdi', 'v2x_polyarchy']
X = data[features]
y = data[target]

# Split the data into training and validation sets
X_train_val,X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
# Define a range of n_splines to test
n_splines_range = np.arange(5, 25, 5)

best_score = np.inf
best_n_splines = None
best_model = None

# Iterate over each n_splines value
for n in n_splines_range:
    gam = LinearGAM(s(0, n_splines=n) + s(1, n_splines=n) + s(2, n_splines=n) + s(3, n_splines=n) + s(4, n_splines=n),distribution=Gamma(), link=LogLink()).fit(X_train, y_train)
    y_pred = gam.predict(X_val)
    score = mean_squared_error(y_val, y_pred)
    if score < best_score:
        best_score = score
        best_n_splines = n
        best_model = gam

    print(f'n_splines: {n}, MSE: {score}')

# Display the best n_splines and its MSE
print(f'Best n_splines: {best_n_splines}, with MSE: {best_score}')

# Plotting partial dependence for the best model
for i, feature in enumerate(features):
    XX = best_model.generate_X_grid(term=i)
    pdep, confi = best_model.partial_dependence(term=i, X=XX, width=0.95)
    plt.figure()
    plt.title(f'Effect of {feature}')
    plt.scatter(X_val[feature], y_pred, color='blue', alpha=0.5)
    plt.plot(XX[:, i], pdep, color='black')
    plt.plot(XX[:, i], confi, c='red', ls='--')
    plt.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='red', alpha=0.1)
    plt.show()
# %%
