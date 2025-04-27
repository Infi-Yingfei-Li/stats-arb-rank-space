#%%
import os, sys, copy, h5py, datetime, gc, tqdm, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors 
import sklearn.linear_model
import multiprocessing
from joblib import Parallel, delayed
from polygon import RESTClient
import urllib.request, json

batch_idx = 1
start_year = 2004; end_year = 2022

#%% load equity_idx_PERMNO_ticker map
equity_idx_ticker_map = pd.read_csv("equity_idx_ticker_Polygon.csv")
equity_idx_list = equity_idx_ticker_map["equity_idx"].to_list()

#%% load time_axis from CRSP
data = np.load("equity_data_19700101_20221231_complete.npz")
time_axis_all = [datetime.datetime.strptime(str(j), "%Y%m%d") for j in data["time_axis_int"]]
#equity_idx_list = np.array(data["equity_idx_list"])
t_idx = [j for j in range(len(time_axis_all)) if time_axis_all[j]>=datetime.datetime(start_year, 1, 1) and time_axis_all[j]<=datetime.datetime(end_year, 12, 31)]
time_axis = copy.deepcopy([time_axis_all[j] for j in t_idx])
del data, time_axis_all; gc.collect()

#%%
num_per_batch = len(equity_idx_list)//10
equity_idx_list_sub = equity_idx_list[batch_idx*num_per_batch:min((batch_idx+1)*num_per_batch, len(equity_idx_list))]
equity_idx_list_sub

#%%
capitalization = np.zeros((len(equity_idx_list_sub), len(time_axis)))
share_outstanding = np.zeros((len(equity_idx_list_sub), len(time_axis)))
url_template = "https://api.polygon.io/v3/reference/tickers/{}?date={}&apiKey=Vpri9DW313MwLIKIRpiCzL5WeAVLs3Xi"

for equity_idx in tqdm.tqdm(equity_idx_list_sub):
    ticker = equity_idx_ticker_map.iloc[equity_idx, 1]
    is_success = True
    for t_idx in range(len(time_axis)):
        url_str = url_template.format(ticker, datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d"))
        print(url_str)
        try:
            with urllib.request.urlopen(url_str) as url:
                data = json.load(url)
                capitalization[equity_idx - equity_idx_list_sub[0], t_idx] = data["results"]["market_cap"]
                share_outstanding[equity_idx - equity_idx_list_sub[0], t_idx] = data["results"]["weighted_shares_outstanding"]
        except urllib.error.HTTPError:
            tqdm.tqdm.write("URL not found - equity_idx: {}, ticker: {}, time: {}".format(equity_idx, ticker, datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")))
            is_success = False
            break
    if is_success:
        tqdm.tqdm.write("request data complete - equity_idx: {}, ticker: {}".format(equity_idx, ticker))

save_file_name = "equity_data_polygon_capitalization_batch_{}.npz".format(batch_idx)
time_axis_stamp = [j.timestamp() for j in time_axis]
np.savez(save_file_name, time_axis_stamp = time_axis_stamp, equity_idx_list_sub=equity_idx_list_sub, capitalization=capitalization, share_outstanding=share_outstanding)

#%%
url_template = "https://api.polygon.io/v3/reference/tickers/{}?date={}&apiKey=Vpri9DW313MwLIKIRpiCzL5WeAVLs3Xi"
url = url_template.format("AAPL", "2022-01-03")
with urllib.request.urlopen(url) as url:
    data = json.load(url)
    print(data["results"]["market_cap"])
    print(data["results"]["weighted_shares_outstanding"])



# %%
