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

#%% load daily data derived from Polygon.io
price_daily_high_freq_derived = np.zeros((len(equity_idx_list), 0))
for year in tqdm.tqdm(np.arange(start_time.year, end_time.year+1, 1)):
    file_name = "log_process_data/log_process_data_year_{}.npz".format(year)
    result = np.load(file_name)
    price_daily_high_freq_derived = np.hstack((price_daily_high_freq_derived, result["price_daily_high_freq_derived"]))
    del result

#%% compare the similarity between the two sources by linear regression
if os.path.exists("linreg_result_all_time.npz"):
    result = np.load("linreg_result_all_time.npz")
    linreg_result = result["linreg_result"]
    del result
else:
    print("Comparing the similarity between CRSP and Polygon.io by linear regression...")
    linreg_result = np.zeros((3, len(equity_idx_list), len(time_axis))); linreg_result[:] = np.nan
    for equity_idx in tqdm.tqdm(range(len(equity_idx_list))):
        for t_idx in np.arange(consistency_evaluation_window-1, len(time_axis), 1):
            x = price[equity_idx, (t_idx-consistency_evaluation_window+1):(t_idx+1)]
            y = price_daily_high_freq_derived[equity_idx, (t_idx-consistency_evaluation_window+1):(t_idx+1)]
            idx = np.logical_and(~np.isnan(x), ~np.isnan(y))
            if np.sum(idx) > consistency_evaluation_window//2:
                reg = sklearn.linear_model.LinearRegression().fit(x[idx].reshape(-1, 1), y[idx].reshape(-1, 1))
                R2 = reg.score(x[idx].reshape(-1, 1), y[idx].reshape(-1, 1))
                linreg_result[:, equity_idx, t_idx] = np.array([R2, reg.coef_[0, 0], reg.intercept_[0]])

    np.savez("linreg_result_all_time.npz", linreg_result=linreg_result)

#%% filter and generate data by year
print("Filtering and generating high frequency and daily data...")
consistent_equity_idx_R2 = np.array([j for j in range(len(equity_idx_list)) if np.sum(~np.isnan(linreg_result[0, j, :]))>0 and np.sum(linreg_result[0, j, :]>R2_threshold)/np.sum(~np.isnan(linreg_result[0, j, :]))>proportion_threshold])
consistent_equity_idx_slope = np.array([j for j in range(len(equity_idx_list)) if np.sum(~np.isnan(linreg_result[1, j, :]))>0 and np.sum(np.abs(linreg_result[1, j, :]-1)<beta_threshold)/np.sum(~np.isnan(linreg_result[1, j, :]))>proportion_threshold])
consistent_equity_idx = np.intersect1d(consistent_equity_idx_R2, consistent_equity_idx_slope)

price_daily_high_freq_derived_filtered = np.zeros((len(equity_idx_list), len(time_axis))); price_daily_high_freq_derived_filtered[:] = np.nan
capitalization_daily_high_freq_derived_filtered = np.zeros((len(equity_idx_list), len(time_axis))); capitalization_daily_high_freq_derived_filtered[:] = np.nan
rank_daily_high_freq_derived_filtered = np.zeros((len(equity_idx_list), len(time_axis))); rank_daily_high_freq_derived_filtered[:] = np.nan
equity_idx_by_rank_daily_high_freq_derived_filtered = np.zeros((len(equity_idx_list), len(time_axis))); equity_idx_by_rank_daily_high_freq_derived_filtered[:] = np.nan

if IS_OUTPUT_BY_YEAR:
    for t_idx in tqdm.tqdm(range(len(time_axis))):
        if t_idx == 0 or time_axis[t_idx].year != time_axis[t_idx-1].year:
            current_year = time_axis[t_idx].year
            tqdm.tqdm.write("Processing year {}...".format(current_year))
            pt_start_high_freq_current_year = np.searchsorted(time_axis_high_freq, datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 4, 0))
            time_axis_high_freq_len_current_year = np.sum([j.year==current_year for j in time_axis_high_freq])
            price_high_freq_current_year = np.zeros((len(equity_idx_list), time_axis_high_freq_len_current_year)); price_high_freq_current_year[:] = np.nan
            capitalization_high_freq_current_year = np.zeros((len(equity_idx_list), time_axis_high_freq_len_current_year)); capitalization_high_freq_current_year[:] = np.nan
            rank_high_freq_current_year = np.zeros((len(equity_idx_list), time_axis_high_freq_len_current_year)); rank_high_freq_current_year[:] = np.nan
            equity_idx_by_rank_high_freq_current_year = np.zeros((len(equity_idx_list), time_axis_high_freq_len_current_year)); equity_idx_by_rank_high_freq_current_year[:] = np.nan
            gc.collect()

            file_name = "log_process_data/log_process_data_year_{}.npz".format(current_year)
            result = np.load(file_name)
            price_high_freq = result["price_high_freq"]

        # filter and generate high frequency data
        time_day_start = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 4, 0)
        time_day_end = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 20, 0)
        t_idx_high_freq = np.arange(np.searchsorted(time_axis_high_freq, time_day_start), np.searchsorted(time_axis_high_freq, time_day_end)+1, 1) - pt_start_high_freq_current_year
        valid_idx_equity_axis = np.array([j for j in range(len(equity_idx_list)) if ~np.isnan(linreg_result[0, j, t_idx]) and linreg_result[0, j, t_idx]>R2_threshold])
        if len(valid_idx_equity_axis) == 0:
            continue

        for j in t_idx_high_freq:
            price_high_freq_current_year[valid_idx_equity_axis, j] = price_high_freq[valid_idx_equity_axis, j]

            # adjust the share outstanding based on the linear regression result such that the capitalization is consistent between CRSP and Polygon.io
            capitalization_high_freq_current_year[valid_idx_equity_axis, j] = price_high_freq[valid_idx_equity_axis, j]*share_outstanding[valid_idx_equity_axis, t_idx]/linreg_result[1, valid_idx_equity_axis, t_idx]
            # for equity that is consistent with CRSP throughout the history (i.e. in consistent_equity_idx), directly use the share outstanding from CRSP
            capitalization_high_freq_current_year[consistent_equity_idx, j] = price_high_freq[consistent_equity_idx, j]*share_outstanding[consistent_equity_idx, t_idx]

            valid_idx_equity_axis_all = np.array([k for k in range(capitalization_high_freq_current_year.shape[0]) if ~np.isnan(capitalization_high_freq_current_year[k, j])])
            sort_idx = np.argsort(capitalization_high_freq_current_year[valid_idx_equity_axis_all, j])[::-1]
            rank_high_freq_current_year[valid_idx_equity_axis_all[sort_idx], j] = np.arange(0, len(valid_idx_equity_axis_all), 1)
            equity_idx_by_rank_high_freq_current_year[0:len(valid_idx_equity_axis_all), j] = valid_idx_equity_axis_all[sort_idx]

        price_daily_high_freq_derived_filtered[:, t_idx] = price_high_freq_current_year[:, t_idx_high_freq[-1]]
        capitalization_daily_high_freq_derived_filtered[:, t_idx] = capitalization_high_freq_current_year[:, t_idx_high_freq[-1]]
        rank_daily_high_freq_derived_filtered[:, t_idx] = rank_high_freq_current_year[:, t_idx_high_freq[-1]]
        equity_idx_by_rank_daily_high_freq_derived_filtered[:, t_idx] = equity_idx_by_rank_high_freq_current_year[:, t_idx_high_freq[-1]]

        if t_idx == len(time_axis)-1 or time_axis[t_idx+1].year != time_axis[t_idx].year:
            time_axis_high_freq_stamp_current_year = [j.timestamp() for j in time_axis_high_freq if j.year==current_year]
            save_file_name = "minute_aggs_tensorized_filtered/minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_year_{}.npz".format(time_axis[t_idx].year)
            np.savez_compressed(save_file_name, time_axis_high_freq=time_axis_high_freq_stamp_current_year, price_high_freq=price_high_freq_current_year,
                        capitalization_high_freq=capitalization_high_freq_current_year, rank_high_freq=rank_high_freq_current_year, equity_idx_by_rank_high_freq=equity_idx_by_rank_high_freq_current_year)
            del price_high_freq, price_high_freq_current_year, capitalization_high_freq_current_year, rank_high_freq_current_year, equity_idx_by_rank_high_freq_current_year; gc.collect()

else:
    for t_idx in tqdm.tqdm(range(len(time_axis))):
        if t_idx == 0 or time_axis[t_idx].year != time_axis[t_idx-1].year:
            current_year = time_axis[t_idx].year
            pt_start_high_freq_current_year = np.searchsorted(time_axis_high_freq, datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 4, 0))
            tqdm.tqdm.write("Processing year {}...".format(current_year))
            file_name = "log_process_data/log_process_data_year_{}.npz".format(current_year)
            result = np.load(file_name)
            price_high_freq = result["price_high_freq"]
            time_axis_high_freq_len_current_year = np.sum([j.year==current_year for j in time_axis_high_freq])

        # filter and generate high frequency data
        time_day_start = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 4, 0)
        time_day_end = datetime.datetime(time_axis[t_idx].year, time_axis[t_idx].month, time_axis[t_idx].day, 20, 0)
        t_idx_high_freq = np.arange(np.searchsorted(time_axis_high_freq, time_day_start), np.searchsorted(time_axis_high_freq, time_day_end)+1, 1) - pt_start_high_freq_current_year
        price_high_freq_current_day = np.zeros((len(equity_idx_list), len(t_idx_high_freq))); price_high_freq_current_day[:] = np.nan
        capitalization_high_freq_current_day = np.zeros((len(equity_idx_list), len(t_idx_high_freq))); capitalization_high_freq_current_day[:] = np.nan
        rank_high_freq_current_day = np.zeros((len(equity_idx_list), len(t_idx_high_freq))); rank_high_freq_current_day[:] = np.nan
        equity_idx_by_rank_high_freq_current_day = np.zeros((len(equity_idx_list), len(t_idx_high_freq))); equity_idx_by_rank_high_freq_current_day[:] = np.nan

        valid_idx_equity_axis = np.array([j for j in range(len(equity_idx_list)) if ~np.isnan(linreg_result[0, j, t_idx]) and linreg_result[0, j, t_idx]>R2_threshold])
        if len(valid_idx_equity_axis) == 0:
            np.savez_compressed("minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_{}.npz".format(datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")),
                    time_axis_high_freq = [], price_high_freq = price_high_freq_current_day, capitalization_high_freq = capitalization_high_freq_current_day,
                    rank_high_freq = rank_high_freq_current_day, equity_idx_by_rank_high_freq = equity_idx_by_rank_high_freq_current_day)
            continue
        for j in t_idx_high_freq:
            price_high_freq_current_day[valid_idx_equity_axis, j-t_idx_high_freq[0]] = price_high_freq[valid_idx_equity_axis, j]

            # adjust the share outstanding based on the linear regression result such that the capitalization is consistent between CRSP and Polygon.io
            capitalization_high_freq_current_day[valid_idx_equity_axis, j-t_idx_high_freq[0]] = price_high_freq_current_day[valid_idx_equity_axis, j-t_idx_high_freq[0]]*share_outstanding[valid_idx_equity_axis, t_idx]/linreg_result[1, valid_idx_equity_axis, t_idx]
            # for equity that is consistent with CRSP throughout the history (i.e. in consistent_equity_idx), directly use the share outstanding from CRSP without any linear regression fine adjustment
            capitalization_high_freq_current_day[consistent_equity_idx, j-t_idx_high_freq[0]] = price_high_freq_current_day[consistent_equity_idx, j-t_idx_high_freq[0]]*share_outstanding[consistent_equity_idx, t_idx]

            valid_idx_equity_axis_all = np.array([k for k in range(capitalization_high_freq_current_day.shape[0]) if ~np.isnan(capitalization_high_freq_current_day[k, j-t_idx_high_freq[0]])])
            sort_idx = np.argsort(capitalization_high_freq_current_day[valid_idx_equity_axis_all, j-t_idx_high_freq[0]])[::-1]
            rank_high_freq_current_day[valid_idx_equity_axis_all[sort_idx], j-t_idx_high_freq[0]] = np.arange(0, len(valid_idx_equity_axis_all), 1)
            equity_idx_by_rank_high_freq_current_day[0:len(valid_idx_equity_axis_all), j-t_idx_high_freq[0]] = valid_idx_equity_axis_all[sort_idx]

        price_daily_high_freq_derived_filtered[:, t_idx] = price_high_freq_current_day[:, t_idx_high_freq[-1]-t_idx_high_freq[0]]
        capitalization_daily_high_freq_derived_filtered[:, t_idx] = capitalization_high_freq_current_day[:, t_idx_high_freq[-1]-t_idx_high_freq[0]]
        rank_daily_high_freq_derived_filtered[:, t_idx] = rank_high_freq_current_day[:, t_idx_high_freq[-1]-t_idx_high_freq[0]]
        equity_idx_by_rank_daily_high_freq_derived_filtered[:, t_idx] = equity_idx_by_rank_high_freq_current_day[:, t_idx_high_freq[-1]-t_idx_high_freq[0]]

        np.savez_compressed("minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_{}.npz".format(datetime.datetime.strftime(time_axis[t_idx], "%Y-%m-%d")), 
                 time_axis_high_freq = [datetime.datetime.timestamp(time_axis_high_freq[j+pt_start_high_freq_current_year]) for j in t_idx_high_freq], 
                 price_high_freq = price_high_freq_current_day, capitalization_high_freq = capitalization_high_freq_current_day,
                 rank_high_freq = rank_high_freq_current_day, equity_idx_by_rank_high_freq = equity_idx_by_rank_high_freq_current_day)

'''
print("calculating daily return...")
return_daily_high_freq_derived_filtered = np.zeros((len(equity_idx_list), len(time_axis))); return_daily_high_freq_derived_filtered[:] = np.nan
for t_idx in tqdm.tqdm(range(len(time_axis))):
    valid_idx = np.logical_and(~np.isnan(price_daily_high_freq_derived_filtered[:, t_idx]), ~np.isnan(price_daily_high_freq_derived_filtered[:, t_idx-1]))
    return_daily_high_freq_derived_filtered[valid_idx, t_idx] = (price_daily_high_freq_derived_filtered[valid_idx, t_idx] - price_daily_high_freq_derived_filtered[valid_idx, t_idx-1])/price_daily_high_freq_derived_filtered[valid_idx, t_idx-1]
'''

save_file_name = "equity_data_Polygon_daily_from_high_freq_year_{}_{}.npz".format(start_time.year, end_time.year)
np.savez_compressed(save_file_name, time_axis=[j.timestamp() for j in time_axis], price=price_daily_high_freq_derived_filtered, capitalization=capitalization_daily_high_freq_derived_filtered, 
         rank=rank_daily_high_freq_derived_filtered, equity_idx_by_rank=equity_idx_by_rank_daily_high_freq_derived_filtered)


#%% visualize consistency between CRSP and Polygon.io
equity_idx_top_2000 = np.sort(np.unique(equity_idx_by_rank[0:2000, :].flatten()))
equity_idx_top_2000 = equity_idx_top_2000[~np.isnan(equity_idx_top_2000)].astype(int)

#%% random sample price trajectory and compare the two sources
is_intraday_include = True
while True:
    #equity_idx_select = np.random.choice(equity_idx_top_2000, 1)[0]
    equity_idx_select = 1
    if ~ np.isnan(price_daily_high_freq_derived[equity_idx_select, :]).all() and ~ np.isnan(price[equity_idx_select, :]).all():
        price_high_freq_select = []
        if is_intraday_include:
            for year in tqdm.tqdm(np.arange(start_time.year, end_time.year+1, 1)):
                file_name = "log_process_data/log_process_data_year_{}.npz".format(year)
                result = np.load(file_name)
                price_high_freq = result["price_high_freq"]
                price_high_freq_select.extend(price_high_freq[equity_idx_select, :].tolist())
                del result, price_high_freq
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 2, 1)
        if is_intraday_include:
            plt.plot(time_axis_high_freq, price_high_freq_select, label="from Polygon - high_freq")
            plt.plot(time_axis, price_daily_high_freq_derived[equity_idx_select, :], label="from Polygon - daily", alpha=.5, linestyle='--', color='blue')
            plt.scatter(time_axis, price[equity_idx_select, :], label="from CRSP", s=5, color='red')
        else:
            plt.plot(time_axis, price_daily_high_freq_derived[equity_idx_select, :], label="from Polygon - daily")
            plt.scatter(time_axis, price[equity_idx_select, :], label="from CRSP", s=5, color='red')
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.plot(time_axis,rank[equity_idx_select, :], label="rank")
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.legend()
        plt.subplot(3, 2, 3)
        plt.plot(time_axis, capitalization[equity_idx_select, :], label="capitalization")
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.legend()
        plt.subplot(3, 2, 4)
        plt.plot(time_axis, share_outstanding[equity_idx_select, :], label="share outstanding")
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.legend()        
        plt.subplot(3, 2, 5)
        plt.plot(time_axis, linreg_result[0, equity_idx_select, :], label="R2")
        plt.hlines(R2_threshold, time_axis[0], time_axis[-1], color='red', linestyle='--')
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.subplot(3, 2, 6)
        plt.plot(time_axis, linreg_result[1, equity_idx_select, :], label="slope")
        plt.xlim([time_axis[0], time_axis[-1]])
        plt.fill_between(time_axis, 1.1, 0.9, color='red', alpha=0.2)
        plt.legend()
        break

plt.suptitle("Comparison between CRSP and Polygon.io for Ticker = {}".format(equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==equity_idx_select].iloc[-1, 2]))
plt.savefig("consistency_CRSP_Polygon_sample_trajectory.pdf", dpi=300)
print(equity_idx_select, equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==equity_idx_select].iloc[-1, 2])
print("R2 proportion: ", np.sum(linreg_result[0, equity_idx_select, :]>R2_threshold)/np.sum(~np.isnan(linreg_result[0, equity_idx_select, :])))
print("slope proportion: ", np.sum(np.abs(linreg_result[1, equity_idx_select, :]-1)<0.1)/np.sum(~np.isnan(linreg_result[1, equity_idx_select, :])))

#%% R2 and coefficient for top 2000 market subspace
rank_max = 2000
R2 = np.zeros((2000, len(time_axis))); R2[:] = np.nan
slope = np.zeros((2000, len(time_axis))); slope[:] = np.nan
for t_idx in range(len(time_axis)):
    idx = equity_idx_by_rank[0:rank_max, t_idx]
    valid_rank = ~np.isnan(idx)
    idx = idx[valid_rank].astype(int)
    R2[valid_rank, t_idx] = linreg_result[0, idx, t_idx]
    slope[valid_rank, t_idx] = linreg_result[1, idx, t_idx]

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
#colors = ['white', 'black']
#boundaries = [0, R2_threshold, 1]
#colormap = matplotlib.colors.ListedColormap(colors)
#norm = matplotlib.colors.BoundaryNorm(boundaries, colormap.N, clip=True)
plt.imshow(R2, aspect='auto', cmap="RdBu", vmin=0.8, vmax=1, interpolation="none")
plt.colorbar()
plt.ylabel("rank"); plt.xlabel("time")
plt.title(r"$R^2$ between CRSP and Polygon.io")
plt.subplot(2,1,2)
plt.imshow(slope, aspect='auto', cmap="RdBu", vmin=0.9, vmax=1.1, interpolation="none")
plt.colorbar()
plt.ylabel("rank"); plt.xlabel("time"); plt.title(r"$\beta$ between CRSP and Polygon.io")
plt.tight_layout()
plt.savefig("consistency_CRSP_Polygon.png", dpi=300)

#%% R2 and coefficient for top 2000 market subspace
plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
ratio = [np.sum(R2[:, t_idx]>R2_threshold)/len(R2[:, t_idx]) for t_idx in range(len(time_axis))]
plt.plot(time_axis, ratio, label="top 2000")
plt.legend(title=r"$R^2$ between CRSP and Polygon.io")
plt.ylabel("proportion of R2 > {}".format(R2_threshold))
plt.xlabel("time")
plt.subplot(2,1,2)
ratio = [np.sum(np.abs(slope[:, t_idx]-1)<0.1)/len(slope[:, t_idx]) for t_idx in range(len(time_axis))]
plt.plot(time_axis, ratio, label="top 2000")
plt.legend(title=r"$\beta$ between CRSP and Polygon.io")
plt.ylabel("proportion of |slope - 1| < 0.1")
plt.xlabel("time")
plt.tight_layout()
plt.savefig("consistency_CRSP_Polygon.pdf", dpi=300)


# %%

