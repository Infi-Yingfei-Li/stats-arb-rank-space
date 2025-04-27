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

#%%
time_interval_in_min = 1
R2_threshold = 0.9

parser = argparse.ArgumentParser(description='Input year for high frequency data processing.')
parser.add_argument('year', type=int, help='an integer for the year')
args = parser.parse_args()
year = args.year
#year = 2022 # for debug
save_file_name = "equity_data_high_freq_{}_min_{}.npz".format(time_interval_in_min, year)

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

if year == 2003:
    t_idx = [j for j in range(len(time_axis_all)) if time_axis_all[j]>=datetime.datetime(year, 9, 10) and time_axis_all[j]<=datetime.datetime(year, 12, 31)]
else:
    t_idx = [j for j in range(len(time_axis_all)) if time_axis_all[j]>=datetime.datetime(year, 1, 1) and time_axis_all[j]<=datetime.datetime(year, 12, 31)]
time_axis = copy.deepcopy([time_axis_all[j] for j in t_idx])
equity_idx_by_rank = copy.deepcopy(equity_idx_by_rank_all[:, t_idx])
#return_ = copy.deepcopy(return_all[:, t_idx])
#share_outstanding = copy.deepcopy(share_outstanding_all[:, t_idx])
price = copy.deepcopy(price_all[:, t_idx])
#capitalization = copy.deepcopy(capitalization_all[:, t_idx])
#rank = copy.deepcopy(rank_all[:, t_idx])
time_axis_stamp = np.array([datetime.datetime.timestamp(j) for j in time_axis])

del data, time_axis_all, equity_idx_by_rank_all, return_all, share_outstanding_all, price_all, capitalization_all, rank_all
#del data, time_axis_all, equity_idx_by_rank_all, return_all, share_outstanding_all, price_all, capitalization_all, rank_all
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
time_axis_high_freq_stamp = np.array([datetime.datetime.timestamp(j) for j in time_axis_high_freq])

#%%
file_name_high_freq = os.path.join(os.path.dirname(__file__), "minute_aggs/minute_aggs_v1_{}.csv.gz")
log_file_name = os.path.join(os.path.dirname(__file__), "time_interval_{}_min/log_process_data/log_process_data_year_{}.npz".format(time_interval_in_min, year))
if os.path.exists(log_file_name):
    result = np.load(log_file_name)
    t_idx_complete = result["t_idx_complete"]
    price_daily_high_freq_derived = result["price_daily_high_freq_derived"]
    price_high_freq = result["price_high_freq"]
else:
    t_idx_complete = 0
    price_daily_high_freq_derived = np.zeros((len(equity_idx_list), len(time_axis))); price_daily_high_freq_derived[:] = np.nan
    price_high_freq = np.zeros((len(equity_idx_list), len(time_axis_high_freq))); price_high_freq[:] = np.nan

for t_idx in tqdm.tqdm(np.arange(t_idx_complete, len(time_axis), 1)):
    time_day_start = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 4, 0)
    time_day_end = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 20, 0)
    time_str = datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")
    data = pd.read_csv(file_name_high_freq.format(time_str), compression='gzip')
    ticker_list = data["ticker"].unique()

    for ticker in ticker_list:
        sub_data = data[data["ticker"]==ticker]
        equity_idx_all = equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["Ticker"]==ticker]
        # Note that a ticker may correspond to multiple PERMNOs/equity idx due to ticker overuse in different time periods. It will be taken care of in the subsequent filtering.
        if equity_idx_all.shape[0] > 0:
            # convert from pacific standard time to eastern time since the data is downloaded from polygon.io at Stanford in 2024
            time_intraday = [datetime.datetime.fromtimestamp(j/1e9)+datetime.timedelta(hours=3) for j in sub_data["window_start"].to_list()]
            price_intraday = sub_data["close"].to_numpy().flatten()

            x = np.array([datetime.datetime.timestamp(j) for j in time_intraday])
            idx = np.arange(np.searchsorted(time_axis_high_freq, time_day_start), np.searchsorted(time_axis_high_freq, time_day_end)+1, 1)
            x_interp = np.array([datetime.datetime.timestamp(time_axis_high_freq[j]) for j in idx])
            y = price_intraday
            y_interp = np.interp(x_interp, x, y)

            # if is_plot:
            #     plt.plot([datetime.datetime.fromtimestamp(j) for j in x], y)
            #     plt.scatter(time_intraday, price_intraday, color='r', s=5)

            for equity_idx in equity_idx_all.iloc[:, 0].to_list():
                price_high_freq[equity_idx, idx] = copy.deepcopy(y_interp)
                price_daily_high_freq_derived[equity_idx, t_idx] = copy.deepcopy(y_interp[-1])

    del data, ticker_list; gc.collect()
    if t_idx%10 == 0:
        np.savez_compressed(log_file_name, t_idx_complete=t_idx, price_daily_high_freq_derived=price_daily_high_freq_derived, price_high_freq=price_high_freq)

np.savez_compressed(log_file_name, t_idx_complete=len(time_axis), price_daily_high_freq_derived=price_daily_high_freq_derived, price_high_freq=price_high_freq)
np.savez_compressed("time_interval_{}_min/minute_aggs_tensorized/equity_data_Polygon_uncensored_year_{}.npz".format(time_interval_in_min, year), time_axis=time_axis_stamp, time_axis_high_freq=time_axis_high_freq_stamp, 
                    price_daily_high_freq_derived=price_daily_high_freq_derived, price_high_freq=price_high_freq)


# %%
