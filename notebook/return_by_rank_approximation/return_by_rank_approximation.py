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
#PCA_TYPE = "name"
#PCA_TYPE = "rank_theta_transform"
PCA_TYPE = "rank_hybrid_Atlas"

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

if PCA_TYPE == "name":
    data_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_idx_PERMNO_ticker.csv"))
    ticker = list(data_.iloc[:, 2])
    del data_

#%%
return_by_cap = np.zeros((len(equity_data_.equity_idx_list), len(equity_data_.time_axis))); return_by_cap[:] = np.nan
for t in tqdm.tqdm(np.arange(1, len(equity_data_.time_axis), 1)):
    return_by_cap[:, t] = (equity_data_.capitalization[:, t] - equity_data_.capitalization[:, t-1])/equity_data_.capitalization[:, t-1]
cap_diff = return_by_cap - equity_data_.return_

#%% mismatch proportion between return calculated from capitalization and directly from CRSP return
mismatch_prop = []
for t in tqdm.tqdm(np.arange(1, len(equity_data_.time_axis), 1)):
    idx = equity_data_.equity_idx_by_rank[0:2500, t]
    valid_idx = [j for j in idx if not np.isnan(j)]
    valid_idx = np.array(valid_idx).astype(int)
    idx = [j for j in valid_idx if np.abs(cap_diff[j, t]) > 1e-3]
    mismatch_prop.append(len(idx)/len(equity_data_.equity_idx_list))
plt.hist(mismatch_prop, bins=100, density=True)

#%% mismatch proportion between return calculated from capitalization and directly from CRSP return at a single day
t_idx = np.random.choice(range(len(equity_data_.time_axis)), 1)[0]
equity_idx = equity_data_.equity_idx_by_rank[0:2500, t_idx].astype(int)
print(equity_data_.time_axis[t_idx])
plt.subplot(2, 2, 1)
plt.scatter(equity_data_.return_[equity_idx, :][:, t_idx], return_by_cap[equity_idx, :][:, t_idx], s=5)
plt.subplot(2, 2, 2)
plt.scatter(equity_idx, equity_data_.return_[equity_idx, :][:, t_idx]-return_by_cap[equity_idx, :][:, t_idx], s=5)
idx = [j for j in equity_idx if np.abs(equity_data_.return_[j, t_idx]-return_by_cap[j, t_idx]) > 1e-3]
print(len(idx), len(idx)/len(equity_idx))

#%% comparison between price, share outstanding, and return at a single day
plt.subplot(3, 1, 1)
plt.scatter(equity_data_.return_[idx, t_idx], (equity_data_.price[idx, t_idx]-equity_data_.price[idx, t_idx-1])/equity_data_.price[idx, t_idx-1])
plt.subplot(3, 1, 2)
plt.plot((equity_data_.share_outstanding[equity_idx, t_idx]-equity_data_.share_outstanding[equity_idx, t_idx-1])/equity_data_.share_outstanding[equity_idx, t_idx-1])
plt.subplot(3, 1, 3)
plt.scatter(equity_data_.share_outstanding[idx, t_idx], equity_data_.share_outstanding[idx, t_idx-1])

#%%





