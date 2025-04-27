#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import utils.utils as utils

QUICK_TEST = False
PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_dict = {"PCA_name": color_cycle[0], "PCA_rank_permutation": color_cycle[1], "PCA_rank_hybrid_Atlas": color_cycle[2], 
              "PCA_rank_hybrid_Atlas_high_freq": color_cycle[3], "PCA_rank_theta": color_cycle[4]}

t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)

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

equity_data_high_freq_config = {}
equity_data_high_freq_ = data.equity_data_high_freq(equity_data_high_freq_config)

#%% calculate portfolio weights in rank space
rank_min = 0; rank_max = 499
t_eval_start = datetime.datetime(2010,1,1); t_eval_end = datetime.datetime(2022,12,15)
t_idx_list = np.arange(np.searchsorted(equity_data_high_freq_.time_axis_daily, t_eval_start), np.searchsorted(equity_data_high_freq_.time_axis_daily, t_eval_end)+1, 1)
time_hist = [equity_data_high_freq_.time_axis_daily[t_idx] for t_idx in t_idx_list]
epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []

for t_idx in t_idx_list:
    epsilon_idx = np.arange(rank_min, rank_max+1, 1)
    equity_idx = equity_data_high_freq_.equity_idx_by_rank_daily[epsilon_idx, t_idx].astype(int)
    portfolio_weights_R_rank = copy.deepcopy(equity_data_high_freq_.capitalization_daily[equity_idx, t_idx])
    portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
    portfolio_weights_R_rank_all = np.zeros(len(equity_data_.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank
    epsilon_idx_hist.append(epsilon_idx.astype(int))
    portfolio_weights_epsilon_hist.append([np.nan for _ in range(len(equity_data_.equity_idx_list))])
    portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))


#%% calculate artificial PnL
artificial_return_hist = []
for j in tqdm.tqdm(range(len(t_idx_list))):
    t_idx = t_idx_list[j]
    equity_idx_1 = equity_data_high_freq_.equity_idx_by_rank_daily[rank_min:(rank_max+1), t_idx].astype(int)
    capitalization_1 = equity_data_high_freq_.capitalization_daily[equity_idx_1, t_idx]
    equity_idx_2 = equity_data_high_freq_.equity_idx_by_rank_daily[rank_min:(rank_max+1), t_idx+1].astype(int)
    capitalization_2 = equity_data_high_freq_.capitalization_daily[equity_idx_2, t_idx+1]
    return_by_rank = capitalization_2/capitalization_1 - 1
    artificial_return_hist.append(np.dot(return_by_rank, np.array(portfolio_weights_R_rank_hist[j])[epsilon_idx_hist[j]]))

artificial_return_PnL = np.cumprod(1+np.array(artificial_return_hist))

#%% PnL by intra-day rebalance
result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_hist, epsilon_idx_hist, portfolio_weights_R_rank_hist, leverage=1, transaction_cost_factor=0.0002, shorting_cost_factor=0.0000)

#%%
SPX = pd.read_csv(equity_data_config["SPX_file_name"])
SPX_time = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in SPX.iloc[:, 0]]
SPX_index = SPX.iloc[:, -2].to_numpy()
t_idx = np.arange(np.searchsorted(SPX_time, t_eval_start), np.searchsorted(SPX_time, t_eval_end)+1, 1)
SPX_time = [SPX_time[x] for x in t_idx]
SPX_index = SPX_index[t_idx]

plt.figure(figsize=(9, 6))
plt.plot(time_hist, artificial_return_PnL/artificial_return_PnL[0], label="artificial PnL")
plt.plot(result["time_hist"], result["asset_hist_R_name"], label="intraday rebalance")
plt.plot(SPX_time, SPX_index/SPX_index[0], label="SPX index")

plt.legend(); plt.ylabel("PnL ($)"); plt.grid()
plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_cap_weighted_portfolio.pdf"))


# %%
