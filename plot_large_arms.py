import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')

# Filter the trade register data for years after 1991 and aggregate by year
trade_register_data_filtered = trade_register_data[trade_register_data['Year of order'] > 1991]
arms_aggregated = trade_register_data_filtered.groupby('Year of order').agg({
    'SIPRI TIV for total order': 'sum'  # Summing the total value instead of number delivered
}).reset_index().rename(columns={'Year of order': 'year', 'SIPRI TIV for total order': 'total_arms_value'})

# Filter the terrorism data for years after 1991 and aggregate by year
yearly_agg_data_filtered = yearly_agg_data[yearly_agg_data['iyear'] > 1991]
attacks_aggregated = yearly_agg_data_filtered.groupby('iyear').agg({
    'No_Incidents': 'sum'
}).reset_index().rename(columns={'iyear': 'year', 'No_Incidents': 'total_attacks'})

# Merge the datasets on year
merged_data = pd.merge(arms_aggregated, attacks_aggregated, on='year', how='outer').fillna(0)

# Create figure and axis objects with a shared x-axis
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting total attacks with year on the x-axis and total attacks on the left y-axis using line plot
color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Terrorist Attacks', color=color)
ax1.plot(merged_data['year'], merged_data['total_attacks'], color=color, marker='o', linestyle='-', label='Total Terrorist Attacks')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the total arms value using line plot
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Total Arms Value (in millions USD)', color=color)  # Adjusted label
ax2.plot(merged_data['year'], merged_data['total_arms_value'], color=color, marker='o', linestyle='-', label='Total Arms Value')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100000)

# Add a title and a legend
plt.title('Total Arms Value and Terrorist Attacks Per Year (Post-1991)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(handles=[ax1.lines[0], ax2.lines[0]], loc='upper left')
plt.show()
