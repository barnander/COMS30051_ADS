import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')



# Set up the matplotlib figure
plt.figure(figsize=(18, 5))

# Plot distributions of key numeric columns
plt.subplot(1, 3, 1)
sns.histplot(trade_register_data['Number ordered'].dropna(), kde=False, bins=50, color='blue')
plt.title('Distribution of Number Ordered')
plt.xlabel('Number Ordered')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(trade_register_data['Number delivered'].dropna(), kde=False, bins=50, color='green')
plt.title('Distribution of Number Delivered')
plt.xlabel('Number Delivered')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(trade_register_data['SIPRI TIV for total order'].dropna(), kde=False, bins=50, color='red')
plt.title('Distribution of SIPRI TIV for Total Order')
plt.xlabel('SIPRI TIV (millions $)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate skewness and kurtosis
skewness = {
    'Number Ordered': trade_register_data['Number ordered'].skew(),
    'Number Delivered': trade_register_data['Number delivered'].skew(),
    'SIPRI TIV for Total Order': trade_register_data['SIPRI TIV for total order'].skew()
}

kurtosis = {
    'Number Ordered': trade_register_data['Number ordered'].kurt(),
    'Number Delivered': trade_register_data['Number delivered'].kurt(),
    'SIPRI TIV for Total Order': trade_register_data['SIPRI TIV for total order'].kurt()
}

skewness, kurtosis


import numpy as np

# Apply logarithmic transformation to the columns with highly skewed distributions
# Adding a small constant to avoid taking log of zero
trade_register_data['log_Number_ordered'] = np.log10(trade_register_data['Number ordered'] + 1)
trade_register_data['log_Number_delivered'] = np.log10(trade_register_data['Number delivered'] + 1)
trade_register_data['log_SIPRI_TIV_for_total_order'] = np.log10(trade_register_data['SIPRI TIV for total order'] + 1)

# Set up the matplotlib figure for transformed data
plt.figure(figsize=(18, 5))

# Plot distributions of log-transformed columns
plt.subplot(1, 3, 1)
sns.histplot(trade_register_data['log_Number_ordered'].dropna(), kde=False, bins=50, color='blue')
plt.title('Log Distribution of Number Ordered')
plt.xlabel('Log10 Number Ordered')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
sns.histplot(trade_register_data['log_Number_delivered'].dropna(), kde=False, bins=50, color='green')
plt.title('Log Distribution of Number Delivered')
plt.xlabel('Log10 Number Delivered')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sns.histplot(trade_register_data['log_SIPRI_TIV_for_total_order'].dropna(), kde=False, bins=50, color='red')
plt.title('Log Distribution of SIPRI TIV for Total Order')
plt.xlabel('Log10 SIPRI TIV (millions $)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate skewness and kurtosis for the log-transformed data
log_skewness = {
    'Log Number Ordered': trade_register_data['log_Number_ordered'].skew(),
    'Log Number Delivered': trade_register_data['log_Number_delivered'].skew(),
    'Log SIPRI TIV for Total Order': trade_register_data['log_SIPRI_TIV_for_total_order'].skew()
}

log_kurtosis = {
    'Log Number Ordered': trade_register_data['log_Number_ordered'].kurt(),
    'Log Number Delivered': trade_register_data['log_Number_delivered'].kurt(),
    'Log SIPRI TIV for Total Order': trade_register_data['log_SIPRI_TIV_for_total_order'].kurt()
}

log_skewness, log_kurtosis

# Set up the matplotlib figure for KDE plots
plt.figure(figsize=(18, 5))

# Plot KDEs of log-transformed columns
plt.subplot(1, 3, 1)
sns.kdeplot(trade_register_data['log_Number_ordered'].dropna(), color='blue')
plt.title('KDE of Log Number Ordered')
plt.xlabel('Log10 Number Ordered')
plt.ylabel('Density')

plt.subplot(1, 3, 2)
sns.kdeplot(trade_register_data['log_Number_delivered'].dropna(), color='green')
plt.title('KDE of Log Number Delivered')
plt.xlabel('Log10 Number Delivered')
plt.ylabel('Density')

plt.subplot(1, 3, 3)
sns.kdeplot(trade_register_data['log_SIPRI_TIV_for_total_order'].dropna(), color='red')
plt.title('KDE of Log SIPRI TIV for Total Order')
plt.xlabel('Log10 SIPRI TIV (millions $)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Set up the matplotlib figure for combined histogram, KDE, and outlier detection on log-transformed distributions
plt.figure(figsize=(18, 5))

# Define function to calculate outliers using IQR
def detect_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    return data[(data < Q1 - outlier_step) | (data > Q3 + outlier_step)]

# Detect outliers
outliers_log_number_ordered = detect_outliers(trade_register_data['log_Number_ordered'].dropna())
outliers_log_number_delivered = detect_outliers(trade_register_data['log_Number_delivered'].dropna())
outliers_log_sipri_tiv_for_total_order = detect_outliers(trade_register_data['log_SIPRI_TIV_for_total_order'].dropna())

# Plotting combined histogram, KDE, and outliers for Log Number Ordered
plt.subplot(1, 3, 1)
sns.histplot(trade_register_data['log_Number_ordered'].dropna(), kde=True, bins=50, color='blue')
plt.scatter(outliers_log_number_ordered, np.zeros_like(outliers_log_number_ordered) - 0.01, color='red', s=30, label='Outliers')
plt.title('Log Number Ordered')
plt.xlabel('Log10 Number Ordered')
plt.ylabel('Density/Frequency')
plt.legend()

# Plotting combined histogram, KDE, and outliers for Log Number Delivered
plt.subplot(1, 3, 2)
sns.histplot(trade_register_data['log_Number_delivered'].dropna(), kde=True, bins=50, color='green')
plt.scatter(outliers_log_number_delivered, np.zeros_like(outliers_log_number_delivered) - 0.01, color='red', s=30, label='Outliers')
plt.title('Log Number Delivered')
plt.xlabel('Log10 Number Delivered')
plt.ylabel('Density/Frequency')
plt.legend()

# Plotting combined histogram, KDE, and outliers for Log SIPRI TIV for Total Order
plt.subplot(1, 3, 3)
sns.histplot(trade_register_data['log_SIPRI_TIV_for_total_order'].dropna(), kde=True, bins=50, color='red')
plt.scatter(outliers_log_sipri_tiv_for_total_order, np.zeros_like(outliers_log_sipri_tiv_for_total_order) - 0.01, color='red', s=30, label='Outliers')
plt.title('Log SIPRI TIV for Total Order')
plt.xlabel('Log10 SIPRI TIV (millions $)')
plt.ylabel('Density/Frequency')
plt.legend()

plt.tight_layout()
plt.show()
