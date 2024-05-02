#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
import seaborn as sns
data = pd.read_excel("globalterrorismdb_0522dist.xlsx")
# %% do standard exploration tests (mean,median,std) for nkills
def numerical_var_stats(var_name,x_axis,bins = 20, log = False):
    print(f"Mean: {data[var_name].mean()}")
    print(f"Median: {data[var_name].median()}")
    print(f"Standard Deviation: {data[var_name].std()}")
    print(f"Skewness: {data[var_name].skew()}")
    print(f"Kurtosis: {data[var_name].kurt()}")
    print(f"Number of Zeros: {len(data[data[var_name] == 0])}")
    print(f"3rd Quartile: {data[var_name].quantile(0.75)}")
    plt.figure()
    plt.ylabel("Number of Incidents")
    plt.xlabel(x_axis)
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
    if cat[:-1] == "weaptype":
        cat_txts[7] = 'Vehicle'
    id_to_txt = {cat_ids[i]: cat_txts[i] for i in range(len(cat_ids)) if type(cat_txts[i]) == str}
    return id_to_txt

def cat_var_stats(var_name, data = data, log = False):
    dict_names = id_to_txt_dict(var_name[:-1] + '1')
    values = dict_names.keys()
    num_val = np.zeros(len(values))
    for i,value in enumerate(values):
        num_val[i] = len(data[data[var_name] == value])
    value_names = [dict_names[value] for value in values]
    plt.bar(value_names,num_val, log = log)
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=90)
    plt.grid(True)
    #plt.show()

def cat_var_stats_multiple(var_name, number, data = data, log = False):
    dict_names = id_to_txt_dict(var_name + '1') 
    values = dict_names.keys()
    num_val = np.zeros(len(values))
    for i, value in enumerate(values):
        for j in range(1,number+1):
            num_val[i] += len(data[data[var_name + str(j)] == value])
    value_names = [dict_names[value] for value in values]
    plt.bar(value_names,num_val, log = log)
    plt.ylabel("Number of Incidents")
    plt.xticks(rotation=90)
    plt.grid(True)



        

def make_binary_column(cats, value):
    idx = (data[cats] == value).any(axis=1)
    return idx
#%%size of data
data["casualties"] = data["nkill"] + data["nwound"]

# %%
plt.figure()
cat_var_stats_multiple("weaptype",4,log = True)
plt.xlabel("Weapon Type")
plt.show()
#%%
#numerical_var_stats("nkill",bins = 100, log = True)
#numerical_var_stats("nwound",bins = 100,log = True)
numerical_var_stats("casualties","Number of Casualties",bins = 100,log = True)

#%% weapons used
plt.figure()
cat_var_stats("weaptype1", log = True)
plt.xlabel("Weapon Type")
plt.show()
plt.figure()
cat_var_stats("weaptype2", log = True)
plt.show()
plt.figure()
cat_var_stats("weaptype3", log = True)
plt.show()
# %% plot target types
plt.figure()
cat_var_stats("targtype1")
plt.show()
*
# %%

high_corr_countries = ['Montenegro','Burkina Faso','Czechoslavakia','Saudi Arabia','Bosnia-Herzegovina','Hungary','Nicauragua','Dominican Republic','Sierra Leone','Lithuania','Mali','Singapore','China','Niger','Switzerland']
high_corr_data = data[data['country_txt'].isin(high_corr_countries)]
plt.figure()
cat_var_stats("targtype1",data = high_corr_data)
plt.show()
low_corr_countries = ['Libya','Azerbaijan','Yugoslavia','Norway','Tajikistan','Yemen','Jamaica','Bahamas','Paraguay','Mauritania','Armenia','North Yemen','Syria','Uruguay','Soviet Union']
low_corr_data = data[data['country_txt'].isin(low_corr_countries)]
plt.figure()
cat_var_stats("targtype1",data = low_corr_data)
plt.show()
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
