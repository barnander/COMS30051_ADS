#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
data = pd.read_excel("globalterrorismdb_0522dist.xlsx")
# %%
columns = data.columns
for col in columns:
    with open("columns.txt", "a") as f:
        f.write(col + "\n")
# %%
