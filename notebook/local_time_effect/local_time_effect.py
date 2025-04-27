#%%
import os, sys, copy, h5py, datetime, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import utils.utils as utils

#%%
QUICK_TEST = False
PCA_TYPE = "name"
#PCA_TYPE = "rank_theta_transform"
#PCA_TYPE = "rank_hybrid_Atlas"

t_eval_start_list = [datetime.datetime(1991,1,1), datetime.datetime(1996,1,1), datetime.datetime(2001,1,1), datetime.datetime(2006,1,1), datetime.datetime(2011,1,1), datetime.datetime(2016,1,1)]
t_eval_end_list = [datetime.datetime(1995,12,31), datetime.datetime(2000,12,31), datetime.datetime(2005,12,31), datetime.datetime(2010,12,31), datetime.datetime(2015,12,31), datetime.datetime(2022,12,15)]
equity_data_config = {"Fama_French_3factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Fama_French_3factor_19700101_20221231.csv"),
            "Fama_French_5factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Fama_French_5factor_19700101_20221231.csv"),
            "equity_file_name":  os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_data_19700101_20221231.csv"),
            "filter_by_return_threshold": 3,
            "SPX_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/SPX_19700101_20221231.csv"),
            "Russel2000_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Russel2000_19879010_20231024.csv"),
            "Russel3000_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Russel3000_19879010_20231110.csv"),
            "macroeconomics_124_factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/macroeconomics_124_factor_19700101_20221231.csv")}
equity_data_ = data.equity_data(equity_data_config)

if PCA_TYPE == "name":
    data_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_idx_PERMNO_ticker.csv"))
    ticker = list(data_.iloc[:, 2])
    del data_

t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)
t_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end)+1, 1)

factor_evaluation_window = 252

# %%
rank = 0
#plt.scatter([equity_data_.time_axis[k] for k in t_idx], equity_data_.equity_idx_by_rank[rank, t_idx], label="{}".format(rank), s=1)
plt.hist(equity_data_.equity_idx_by_rank[rank, t_idx], bins=100)
plt.legend()

#%% number of rank variations
rank_range = [(0, 99), (100, 199), (500, 599), (1000, 1099), (2000, 2099)]
rank_variation = []
for t in tqdm.tqdm(t_idx):
    temp = []
    for r in rank_range:
        idx = [j for j in np.arange(r[0], r[1]+1, 1) if equity_data_.equity_idx_by_rank[j, t] != equity_data_.equity_idx_by_rank[j, t-1]]
        temp.append(len(idx))
    rank_variation.append(temp)
rank_variation = np.array(rank_variation).T

#%%
plt.figure(figsize=(20, 10))
for j in range(len(rank_range)):
    plt.plot([equity_data_.time_axis[j] for j in t_idx], rank_variation[j, :]/(rank_range[j][1]-rank_range[j][0]+1), label="{}-{}".format(rank_range[j][0], rank_range[j][1]), alpha=1)
plt.legend()
plt.ylabel("rank variation between t and t-1")
plt.xlabel("time")

#%%
eval_hist = []
for t in tqdm.tqdm(t_idx):
    t_eval_idx = np.arange(t-factor_evaluation_window+1, t+1, 1)
    temp = []
    for r in rank_range:
        rank_idx = [j for j in np.arange(r[0], r[1]+1, 1) if not any(np.isnan(equity_data_.capitalization[j, [t_eval_idx[0]-1]+list(t_eval_idx)]))]
        result = equity_data_.return_by_rank_func(rank_idx, equity_data_.time_axis[t_eval_idx[0]], equity_data_.time_axis[t_eval_idx[-1]], mode="hybrid-Atlas")
        R = result["return_by_rank"] - equity_data_.risk_free_rate[t_eval_idx]
        U, S, V_T = np.linalg.svd(R, full_matrices=True)
        eval = np.power(S, 2)
        temp.append(eval[0]/np.sum(eval))
    eval_hist.append(temp)
eval_hist = np.array(eval_hist).T


#%%
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
for j in range(len(rank_range)):
    plt.plot([equity_data_.time_axis[j] for j in t_idx], rank_variation[j, :]/(rank_range[j][1]-rank_range[j][0]+1), label="{}-{}".format(rank_range[j][0], rank_range[j][1]), alpha=1)
plt.legend()
plt.ylabel("rank variation between t and t-1")
plt.xlabel("time")
plt.subplot(2, 1, 2)
for j in range(len(rank_range)):
    plt.plot([equity_data_.time_axis[j] for j in t_idx], eval_hist[j, :], label="{}-{}".format(rank_range[j][0], rank_range[j][1]), alpha=1)
plt.ylabel(r'$\frac{\lambda_{(1)}}{\sum_{i=1}^{n}\lambda_i}$')
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "eigenvalue_vs_rank_variation.pdf"))


#%%


