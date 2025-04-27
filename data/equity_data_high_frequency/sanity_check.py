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


#%% create equity_idx_ticker map only based on Polygon daily agg data
'''
data = np.load("equity_data_19700101_20221231_complete.npz")
time_axis_all = [datetime.datetime.strptime(str(j), "%Y%m%d") for j in data["time_axis_int"]]
t_idx = [j for j in range(len(time_axis_all)) if time_axis_all[j]>=datetime.datetime(2003, 9, 10) and time_axis_all[j]<=datetime.datetime(2022, 12, 31)]
time_axis = copy.deepcopy([time_axis_all[j] for j in t_idx])
del data, time_axis_all; gc.collect()

ticker_list = set()
file_name = os.path.join(os.path.dirname(__file__), "day_aggs_v1/day_aggs_v1_{}.csv.gz")
for t in tqdm.tqdm(time_axis):
    df = pd.read_csv(file_name.format(datetime.datetime.strftime(t, "%Y-%m-%d")), compression="gzip")
    ticker_list.update(set(df["ticker"].to_list()))

equity_idx_list = np.arange(0, len(ticker_list), 1)
equity_idx_ticker = pd.DataFrame({"equity_idx": equity_idx_list, "ticker": list(ticker_list)})
equity_idx_ticker.to_csv("equity_idx_ticker_Polygon.csv", index=False)
'''

#%% global parameters
time_interval_in_min = 5
R2_threshold = 0.9
beta_threshold = 0.1
proportion_threshold = 0.95
IS_OUTPUT_BY_YEAR = False

consistency_evaluation_window = 60
start_time = datetime.datetime(2003, 9, 10)
end_time = datetime.datetime(2022, 12, 31)
#start_time = datetime.datetime(2010, 1, 1)
#end_time = datetime.datetime(2019, 12, 31)

#%% load equity_idx_PERMNO_ticker map
equity_idx_PERMNO_ticker = pd.read_csv("equity_idx_PERMNO_ticker.csv")

#%% load complete daily data from CRSP
data = np.load("equity_data_19700101_20221231_complete.npz")
time_axis_all = [datetime.datetime.strptime(str(j), "%Y%m%d") for j in data["time_axis_int"]]

equity_idx_list = np.array(data["equity_idx_list"])
equity_idx_by_rank_all = np.array(data["equity_idx_by_rank"])
return_all = np.array(data["return_"])
share_outstanding_all = np.array(data["share_outstanding"])
price_all = np.array(data["price"])
capitalization_all = np.array(data["capitalization"])
rank_all = np.array(data["rank"])

t_idx = [j for j in range(len(time_axis_all)) if time_axis_all[j]>=start_time and time_axis_all[j]<=end_time]
time_axis = copy.deepcopy([time_axis_all[j] for j in t_idx])
equity_idx_by_rank = copy.deepcopy(equity_idx_by_rank_all[:, t_idx])
return_ = copy.deepcopy(return_all[:, t_idx])
share_outstanding = copy.deepcopy(share_outstanding_all[:, t_idx])
price = copy.deepcopy(price_all[:, t_idx])
capitalization = copy.deepcopy(capitalization_all[:, t_idx])
rank = copy.deepcopy(rank_all[:, t_idx])

del data, time_axis_all, equity_idx_by_rank_all, return_all, share_outstanding_all, price_all, capitalization_all, rank_all
gc.collect()

#%% create high frequency time axis
time_axis_high_freq = []
for j in range(len(time_axis)):
    t_day_start = datetime.datetime(time_axis[j].year, time_axis[j].month, time_axis[j].day, 4, 0)
    t_day_end = datetime.datetime(time_axis[j].year, time_axis[j].month, time_axis[j].day, 20, 0)
    t = t_day_start
    while t <= t_day_end:
        time_axis_high_freq.append(t)
        t += datetime.timedelta(minutes=time_interval_in_min)

#%% load daily data derived from Polygon
data = np.load("equity_data_Polygon_daily_from_high_freq_year_2003_2022.npz")
price_daily = data["price"]; capitalization_daily = data["capitalization"]
rank_daily = data["rank"]; equity_idx_by_rank_daily = data["equity_idx_by_rank"]

equity_idx_top_2000 = np.sort(np.unique(equity_idx_by_rank[0:2000, :].flatten()))
equity_idx_top_2000 = equity_idx_top_2000[~np.isnan(equity_idx_top_2000)].astype(int)

#%% load linear regression results
result = np.load("linreg_result_all_time.npz")
linreg_result = result["linreg_result"]

#%% 
equity_idx_select = np.random.choice(equity_idx_top_2000, 1)[0]
ticker = equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==equity_idx_select]["Ticker"].values[-1]
plt.figure(figsize=(10, 10))
plt.subplot(3,2,1)
plt.plot(time_axis, price_daily[equity_idx_select, :], label="from Polygon")
plt.scatter(time_axis, price[equity_idx_select, :], label="from CRSP", s=2, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Price")
plt.subplot(3,2,2)
plt.plot(time_axis, rank_daily[equity_idx_select, :], label="from Polygon")
plt.scatter(time_axis, rank[equity_idx_select, :], label="from CRSP", s=2, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Rank")
plt.subplot(3,2,3)
plt.plot(time_axis, capitalization_daily[equity_idx_select, :], label="from Polygon")
plt.scatter(time_axis, capitalization[equity_idx_select, :], label="from CRSP", s=2, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Capitalization")
plt.subplot(3,2,4)
plt.plot(time_axis, share_outstanding[equity_idx_select, :], label="from CRSP")
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Share Outstanding")
plt.subplot(3,2,5)
plt.plot(time_axis, linreg_result[0, equity_idx_select, :], label="R2")
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("R2")
plt.subplot(3,2,6)
plt.plot(time_axis, linreg_result[1, equity_idx_select, :], label="Beta")
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Beta")
plt.suptitle("Equity: {}".format(ticker))


#%% check consistency between daily price data and high frequency price data from Polygon
while True:
    equity_idx_select = np.random.choice(equity_idx_top_2000, 1)[0]
    ticker = equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==equity_idx_select]["Ticker"].values[-1]
    time_axis_high_freq = []; price_high_freq = []; capitalization_high_freq = []; rank_high_freq = []

    t_idx_select = np.random.choice(np.arange(0, len(time_axis), 1), 1)[0]
    for t_idx in [t_idx_select, t_idx_select+1]:
    #for t_idx in tqdm.tqdm(range(len(time_axis))):
        file_name = os.path.join(os.path.dirname(__file__), "time_interval_5_min/minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_{}.npz".format(datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")))
        data = np.load(file_name, allow_pickle=True)
        if len(data["time_axis_high_freq"]) >= 193:
            time_axis_high_freq.extend([datetime.datetime.fromtimestamp(j) for j in data["time_axis_high_freq"]])
            price_high_freq.extend(list(data["price_high_freq"][equity_idx_select, :]))
            capitalization_high_freq.extend(list(data["capitalization_high_freq"][equity_idx_select, :]))
            rank_high_freq.extend(list(data["rank_high_freq"][equity_idx_select, :]))
    
    if ~np.isnan(capitalization_high_freq).any():
        break

plt.figure(figsize=(10, 5))
plt.plot(time_axis_high_freq, capitalization_high_freq, label="high freq")
plt.scatter([time_axis[j]+datetime.timedelta(hours=20) for j in [t_idx_select, t_idx_select+1]], capitalization_daily[equity_idx_select, t_idx_select:t_idx_select+2], label="daily", s=5, color='red')
plt.legend()


#%%
IS_CRSP = True
plt.figure(figsize=(10, 10))
plt.subplot(3,1,1)
plt.plot(time_axis_high_freq, price_high_freq, label="high freq")
if IS_CRSP:
    plt.scatter(time_axis, price[equity_idx_select, :], label="CRSP", s=5, color='red')
else:
    plt.scatter(time_axis, price_daily[equity_idx_select, :], label="daily", s=5, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Price")
plt.subplot(3,1,2)
plt.plot(time_axis_high_freq, rank_high_freq, label="high freq")
if IS_CRSP:
    plt.scatter(time_axis, rank[equity_idx_select, :], label="CRSP", s=5, color='red')
else:
    plt.scatter(time_axis, rank_daily[equity_idx_select, :], label="daily", s=5, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Rank")
plt.subplot(3,1,3)
plt.plot(time_axis_high_freq, capitalization_high_freq, label="high freq")
if IS_CRSP:
    plt.scatter(time_axis, capitalization[equity_idx_select, :], label="CRSP", s=5, color='red')
else:
    plt.scatter(time_axis, capitalization_daily[equity_idx_select, :], label="daily", s=5, color='red')
plt.xlim([time_axis[0], time_axis[-1]]); plt.legend(); plt.ylabel("Capitalization")
plt.suptitle("Equity: {}".format(ticker))

#%%
t_idx = np.searchsorted(time_axis, datetime.datetime(2022, 7, 1))
file_name = os.path.join(os.path.dirname(__file__), "time_interval_5_min/minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_{}.npz".format(datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")))
data = np.load(file_name, allow_pickle=True)
time_axis_high_freq = [datetime.datetime.fromtimestamp(j) for j in data["time_axis_high_freq"]]
price_high_freq = data["price_high_freq"]
capitalization_high_freq = data["capitalization_high_freq"]
rank_high_freq = data["rank_high_freq"]
equity_idx_by_rank_high_freq = data["equity_idx_by_rank_high_freq"]


#%%
rank_idx = np.arange(0, 1050, 50)
plt.figure(figsize=(18, 18))
for j in range(len(rank_idx)):
    plt.subplot(7, 3, j+1)
    plt.plot(time_axis_high_freq, equity_idx_by_rank_high_freq[rank_idx[j], :], label="rank {}".format(rank_idx[j]))
    plt.xlim([time_axis_high_freq[0], time_axis_high_freq[-1]]); plt.legend(); plt.ylabel("Equity Index")


#%%
interchange_time = []
for j in range(capitalization_high_freq.shape[0]):
    diff = np.abs(np.diff(equity_idx_by_rank_high_freq[j, :]))
    interchange_time.append(np.sum(diff>0))

plt.plot(interchange_time[0:1000])
np.mean(interchange_time[0:1000])

#%%
rank_idx = np.arange(500, 502, 1)
equity_idx_select = np.unique(equity_idx_by_rank_high_freq[rank_idx, :].flatten()).astype(int)
plt.figure(figsize=(10, 5))
for j in equity_idx_select:
    plt.plot(time_axis_high_freq, capitalization_high_freq[j, :], label="equity_idx: {}".format(j))


#%%




