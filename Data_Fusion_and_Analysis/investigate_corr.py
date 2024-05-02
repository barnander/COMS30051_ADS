#%%
import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
#%%
gtd_gran = pd.read_csv("yearly_agg.csv")
arms_trans = pd.read_csv("trade-register.csv", encoding="ISO-8859-1")

#%% group by sending country
sent_agg = arms_trans.groupby(["Supplier","Year of order"]).agg(sent = ("SIPRI TIV for total order","sum")).reset_index()

#%% group by receiving country
rec_agg = arms_trans.groupby(["Recipient","Year of order"]).agg(received = ("SIPRI TIV for total order","sum")).reset_index()

# %%

# %%
