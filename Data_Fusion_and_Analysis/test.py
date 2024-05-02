import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load datasets
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/all_extra_data.csv')

# Prepare and merge data
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index().rename(columns={'Recipient': 'country', 'Year of order': 'year'})
extra_data.rename(columns={'Country Name_x': 'country'}, inplace=True)

# Merge datasets
merged_data = pd.merge(yearly_agg_data[['country_txt', 'iyear', 'No_Incidents']],
                       arms_aggregated,
                       left_on=['country_txt', 'iyear'],
                       right_on=['country_txt', 'year'],
                       how='left').merge(extra_data, on=['country_txt', 'iyear'], how='left')

# Selecting target and features
y = merged_data['No_Incidents']
X = merged_data[['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]

# Handling missing data
X.fillna(X.mean(), inplace=True)

# Encoding categorical data if necessary
# Assuming no categorical data needs encoding here. Adjust if there are categorical variables.

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Converting to PyTorch tensors
X_train = torch.tensor(X_train.astype(np.float32))
X_test = torch.tensor(X_test.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
y_test = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

# Initialize and train the model
model = RegressionModel(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
print(f'Test Loss: {test_loss.item()}')

# Plotting actual vs predicted values
y_test_np = y_test.numpy()
predictions_np = predictions.numpy()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, predictions_np, color='blue')
plt.title('Actual vs Predicted Terrorist Incidents')
plt.xlabel('Actual Incidents')
plt.ylabel('Predicted Incidents')
plt.grid(True)
plt.show()
