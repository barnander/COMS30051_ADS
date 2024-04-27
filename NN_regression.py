import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# Assuming correct loading of datasets
import pandas as pd

trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Aggregating arms imports data by country and year
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'Number delivered': 'sum',
    'SIPRI TIV for total order': 'sum'
}).reset_index()
 
# Renaming columns for clarity and to prevent conflicts in merging
arms_aggregated.rename(columns={'Recipient': 'country', 'Year of order': 'year', 'Number delivered': 'total_arms_delivered', 'SIPRI TIV for total order': 'total_arms_value'}, inplace=True)

# Now, let's merge this with the terrorism data on 'country' and 'year'
yearly_agg_data.rename(columns={'country_txt': 'country', 'iyear': 'year'}, inplace=True)
merged_data = pd.merge(yearly_agg_data, arms_aggregated, how='left', on=['country', 'year'])

# Replace NaN values with 0 for arms data (assuming NaN means no arms were delivered in those years/countries)
merged_data.fillna({'total_arms_delivered': 0, 'total_arms_value': 0}, inplace=True)

# Assuming 'No_Incidents' is the column with the number of terrorist incidents
y = merged_data['No_Incidents']  # This sets the target variable

# Including arms import data and potentially other relevant features
X = merged_data.drop(['No_Incidents', 'country', 'year'], axis=1)  # Drop non-feature and target columns




# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting to PyTorch tensors
X_train = torch.tensor(X_train.astype(np.float32))
X_test = torch.tensor(X_test.astype(np.float32))
y_train = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)
y_test = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)


# Define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x

model = RegressionModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
print(f'Test Loss: {test_loss.item()}')

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_test.numpy(), predictions.numpy(), color='blue')
plt.title('Actual vs Predicted No_Incidents')
plt.xlabel('Actual No_Incidents')
plt.ylabel('Predicted No_Incidents')
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

# Convert tensors back to numpy arrays
y_test_np = y_test.numpy()
predictions_np = predictions.numpy()

# Plotting actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, predictions_np, color='blue')
plt.title('Actual vs Predicted No_Incidents')
plt.xlabel('Actual No_Incidents')
plt.ylabel('Predicted No_Incidents')
plt.grid(True)
plt.show()




