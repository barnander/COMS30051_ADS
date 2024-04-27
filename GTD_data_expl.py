#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
import seaborn as sns
data = pd.read_excel("globalterrorismdb_0522dist.xlsx")
# %% do standard exploration tests (mean,median,std) for nkills
def numerical_var_stats(var_name,bins = 20, log = False):
    print(f"Mean: {data[var_name].mean()}")
    print(f"Median: {data[var_name].median()}")
    print(f"Standard Deviation: {data[var_name].std()}")
    print(f"Skewness: {data[var_name].skew()}")
    print(f"Kurtosis: {data[var_name].kurt()}")
    plt.figure()
    plt.ylabel("Number of Incidents")
    plt.xlabel("Number of Deaths" if var_name == "nkill" else "Number of Wounded")
    plt.grid(True)
    plt.hist(data[var_name], log = log,bins = bins)
    plt.show()

def no_zero_numerical_var(var_name):
    data_no_zero = data[data[var_name] != 0]
    print(f"Mean: {data_no_zero[var_name].mean()}")
    print(f"Median: {data_no_zero[var_name].median()}")
    print(f"Standard Deviation: {data_no_zero[var_name].std()}")
    plt.figure()
    plt.hist(data_no_zero[var_name], bins = 200)
    plt.show()

def id_to_txt_dict(cat):
    cat_ids = data[cat].unique()
    cat_txts = data[cat + "_txt"].unique()
    id_to_txt = {cat_ids[i]: cat_txts[i] for i in range(len(cat_ids))}
    return id_to_txt

def cat_var_stats(var_name):
    dict_names = id_to_txt_dict(var_name)
    values = dict_names.keys()
    num_val = np.zeros(len(values))
    for i,value in enumerate(values):
        num_val[i] = len(data[data[var_name] == value])
    value_names = [dict_names[value] for value in values]
    print(value_names)
    plt.figure()
    plt.bar(value_names,num_val)
    plt.xticks(rotation=45)
    plt.show()


def make_binary_column(cats, value):
    idx = (data[cats] == value).any(axis=1)
    return idx
#%%size of data


# %%
numerical_var_stats("nkill",bins = 100, log = True)
numerical_var_stats("nwound",bins = 100,log = True)


# %%
cat_var_stats("attacktype1")
cat_var_stats("attacktype2")
# %%
cross_tab = pd.crosstab(data["attacktype1_txt"],data["targtype1_txt"],margins=True)
crosstab_prop = cross_tab.div(cross_tab.loc["All", "All"], axis=0)
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab_prop, annot=True, cmap='viridis', fmt='.2f')  
plt.title('Heatmap of Attack Types and Target Types')
plt.xlabel('Target Type')
plt.ylabel('Attack Type')
plt.show()
# Perform the Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(cross_tab)

print("Chi-square Statistic:", chi2)
print("P-value:", p)
# %%
