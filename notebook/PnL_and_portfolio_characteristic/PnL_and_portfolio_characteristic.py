#%%
import os, sys, copy, h5py, datetime, tqdm, gc, pickle
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
#PCA_TYPE = "name"; factor_name = "PCA_name"
#PCA_TYPE = "rank_hybrid_Atlas"; factor_name = "PCA_rank_hybrid_Atlas"
PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"
#PCA_TYPE = "rank_permutation"; factor_name = "PCA_rank_permutation"
#PCA_TYPE = "rank_theta_transform"; factor_name = "PCA_rank_theta"

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

if PCA_TYPE == "name":
    data_ = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_idx_PERMNO_ticker.csv"))
    ticker = list(data_.iloc[:, 2])
    del data_

time_tick_label = ["1991/1/1", "1996/1/1", "2001/1/1", "2006/1/1", "2011/1/1", "2016/1/1", "2022/12/15"]
time_tick_idx = [np.searchsorted(equity_data_.time_axis, datetime.datetime.strptime(t, "%Y/%m/%d")) for t in time_tick_label]
time_tick_idx = np.array(time_tick_idx) - time_tick_idx[0]

PCA_factor_config = {"factor_evaluation_window_length": 252,
                "loading_evaluation_window_length": 60, 
                "residual_return_evaluation_window_length": 60,
                "rank_min": 0,
                "rank_max": 999,
                "factor_number": 3,
                "type": PCA_TYPE,
                "max_cache_len": 100,
                "quick_test": QUICK_TEST}
PCA_factor_ = factor.PCA_factor(equity_data_, equity_data_high_freq_, PCA_factor_config)
PCA_factor_._initialize_residual_return_all()

#%% re-calculate the PnL with/without transaction cost for OU_process-OU_process
transaction_cost_factor = 0.0002; shorting_cost_factor = 0.0000; leverage = 1
#t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)
t_eval_start = datetime.datetime(2007,1,1); t_eval_end = datetime.datetime(2022,12,15)

file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)

if PCA_TYPE == "name":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R = result["portfolio_weights_R"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R = portfolio_weights_R[:, t_idx]
    portfolio_weights_R = list(portfolio_weights_R.T)
    equity_idx = []
    for j in range(len(portfolio_weights_R)):
        equity_idx.append(np.where(np.abs(portfolio_weights_R[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 3))
    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R, leverage=leverage, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = np.array(result["asset_hist"]); return_hist = np.array(result["return_hist"]); transaction_cost_hist = np.array(result["transaction_cost_hist"])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.plot(time_axis_PnL, asset_hist, label="with transaction cost", color=color_dict["PCA_name"])
    plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name)), 
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name)))

    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R, leverage=leverage, transaction_cost_factor=0, shorting_cost_factor=0, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = np.array(result["asset_hist"]); return_hist = np.array(result["return_hist"]); transaction_cost_hist = np.array(result["transaction_cost_hist"])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.plot(time_axis_PnL, asset_hist, label="without transaction cost", color=color_dict["PCA_name"], linestyle="--")
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    #plt.title("PnL: {}-OU_process-OU_process".format(factor_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process.pdf".format(factor_name)))
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name)))

if PCA_TYPE in ["rank_permutation", "rank_theta_transform"]:
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    portfolio_weights_R_name = result["portfolio_weights_R_name"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]; portfolio_weights_R_name = portfolio_weights_R_name[:, t_idx]
    portfolio_weights_R_name = list(portfolio_weights_R_name.T)
    equity_idx = []
    for j in range(len(portfolio_weights_R_name)):
        equity_idx.append(np.where(np.abs(portfolio_weights_R_name[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 3))
    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R_name, leverage=leverage, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]; transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.plot(time_axis_PnL, asset_hist, label="with transaction cost", color=color_dict[factor_name])
    plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name)))

    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R_name, leverage=leverage, transaction_cost_factor=0, shorting_cost_factor=0, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]; transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.plot(time_axis_PnL, asset_hist, label="without transaction cost", color=color_dict[factor_name], linestyle="--")
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    #plt.title("PnL: {}-OU_process-OU_process".format(factor_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process.pdf".format(factor_name)))
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name)))

if PCA_TYPE == "rank_hybrid_Atlas":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    portfolio_weights_R_name = result["portfolio_weights_R_name"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]; portfolio_weights_R_name = portfolio_weights_R_name[:, t_idx]
    portfolio_weights_R_rank = list(portfolio_weights_R_rank.T); portfolio_weights_R_name = list(portfolio_weights_R_name.T)
    epsilon_idx = []
    for j in range(len(portfolio_weights_R_rank)):
        epsilon_idx.append(np.where(np.abs(portfolio_weights_R_rank[j]) > 1e-8)[0])
    equity_idx = []
    for j in range(len(portfolio_weights_R_name)):
        equity_idx.append(np.where(np.abs(portfolio_weights_R_name[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 3))
    #result = utils.evaluate_PnL_R_rank_space(equity_data_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=0, shorting_cost_factor=0)
    #time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist"]; return_hist = result["return_hist"]
    #plt.plot(time_axis_PnL, asset_hist_R_rank, label="from continuous-time return", linestyle=":", color=color_dict[factor_name])

    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R_name, leverage=leverage, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]; transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.plot(time_axis_PnL, asset_hist, label="with transaction cost", color=color_dict[factor_name])
    plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name)))

    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R_name, leverage=leverage, transaction_cost_factor=0, shorting_cost_factor=0, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]; transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.plot(time_axis_PnL, asset_hist, label="without transaction cost", color=color_dict[factor_name], linestyle="--")
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    #plt.title("PnL: {}-OU_process-OU_process".format(factor_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process.pdf".format(factor_name)))
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name)))

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]
    portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]
    portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]
    portfolio_weights_R_rank = list(portfolio_weights_R_rank.T)
    epsilon_idx = []
    for j in range(len(portfolio_weights_R_rank)):
        epsilon_idx.append(np.where(np.abs(portfolio_weights_R_rank[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 3))
    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor)
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]; transaction_cost_hist = result["transaction_cost_hist"]
    return_hist_R_name = np.array([np.nan]+[asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.plot(time_axis_PnL, asset_hist_R_name, label="intraday rebalance, with t.c.", color=color_dict[factor_name])
    plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist_R_name, return_hist=return_hist_R_name, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name)))

    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=0, shorting_cost_factor=0)
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    return_hist_R_name = np.array([np.nan]+[asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.plot(time_axis_PnL, asset_hist_R_rank, label="from continuous-time return", linestyle=":", color=color_dict[factor_name], alpha=0.5)
    plt.plot(time_axis_PnL, asset_hist_R_name, label="intraday rebalance, w.o. t.c.", linestyle="--", color=color_dict[factor_name])
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    #plt.title("PnL: {}-OU_process-OU_process".format(factor_name))
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process.pdf".format(factor_name)))
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist_R_name, return_hist=return_hist_R_name)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name)))


#%% re-calculate the PnL with/without transaction cost for CNN_transformer
#PCA_TYPE = "name"; factor_name = "PCA_name"
PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"

transaction_cost_factor = 0.0002; shorting_cost_factor = 0.0000; leverage = 1
t_eval_start = datetime.datetime(2007,1,1); t_eval_end = datetime.datetime(2022,12,15)
#t_eval_start = datetime.datetime(2005,4,5); t_eval_end = datetime.datetime(2012, 12, 31) # quarterly update
#t_eval_start = datetime.datetime(2006, 1, 3); t_eval_end = datetime.datetime(2012, 12, 31) # annual update

file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)

if PCA_TYPE == "name":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R = result["portfolio_weights_R"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R = portfolio_weights_R[:, t_idx]
    portfolio_weights_R = list(portfolio_weights_R.T)
    equity_idx = []
    for j in range(len(portfolio_weights_R)):
        equity_idx.append(np.where(np.abs(portfolio_weights_R[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 6))
    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R, leverage=leverage, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]
    return_hist = np.array(result["return_hist"]); transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])

    plt.subplot(2, 1, 1)
    plt.plot(time_axis_PnL, asset_hist, label="with transaction cost", color=color_dict[factor_name])
    #plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)

    plt.subplot(2, 1, 2)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    plt.plot(df["year"], df["sharp_ratio"], marker="o", label="with transaction cost", color=color_dict[factor_name])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name)))

    result = utils.evaluate_PnL_name_space(equity_data_, time_axis, equity_idx, portfolio_weights_R, leverage=leverage, transaction_cost_factor=0, shorting_cost_factor=0, is_vanilla=False)
    time_axis_PnL = result["time_hist"]; asset_hist = result["asset_hist"]
    return_hist = np.array(result["return_hist"]); transaction_cost_hist = result["transaction_cost_hist"]
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist[idx[-1]]/asset_hist[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.subplot(2, 1, 1)
    plt.plot(time_axis_PnL, asset_hist, label="without transaction cost", color=color_dict[factor_name], linestyle="--")
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist, return_hist=return_hist, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name)))
    plt.subplot(2, 1, 2)
    plt.plot(df["year"], df["sharp_ratio"], marker="o", label="without transaction cost", color=color_dict[factor_name], linestyle="--")
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer.pdf".format(factor_name)))

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]
    portfolio_weights_R_rank = list(portfolio_weights_R_rank.T)
    epsilon_idx = []
    for j in range(len(portfolio_weights_R_rank)):
        epsilon_idx.append(np.where(np.abs(portfolio_weights_R_rank[j]) > 1e-8)[0])
    plt.figure(figsize=(9, 6))
    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=0, shorting_cost_factor=0, rebalance_interval=45)
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    return_hist_R_name = np.array([np.nan]+[asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.plot(time_axis_PnL, asset_hist_R_rank, label="from continuous-time return", linestyle=":", color=color_dict[factor_name])
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist_R_name, return_hist=return_hist_R_name)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name)))
    f = open(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.pkl".format(factor_name)), "wb")
    pickle.dump(result, f); f.close()

    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=0, rebalance_interval=45)
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]; transaction_cost_hist = result["transaction_cost_hist"]
    return_hist_R_name = np.array([np.nan] + [asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    plt.subplot(2, 1, 1)
    plt.plot(time_axis_PnL, asset_hist_R_name, label="with T.C. & M.C", color=color_dict[factor_name])
    plt.plot(time_axis_PnL, np.cumsum(transaction_cost_hist), label="cumulative transaction cost", color="gray", alpha=0.5)
    plt.ylabel("dollar"); plt.grid(axis="x"); plt.legend(); plt.xlim([time_axis_PnL[0], time_axis_PnL[-1]])
    #plt.title("PnL: {}-CNN_transformer".format(factor_name))
    time_axis_PnL_timestamp = [j.timestamp() for j in time_axis_PnL]
    np.savez(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.npz".format(factor_name)),
            time_hist=time_axis_PnL_timestamp, asset_hist=asset_hist_R_name, return_hist=return_hist_R_name, transaction_cost_hist=transaction_cost_hist)
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    df.to_csv(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name)))
    plt.subplot(2, 1, 2)
    plt.plot(df["year"], df["sharp_ratio"], marker="o", label="with T.C. & M.C", color=color_dict[factor_name])
    plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer.pdf".format(factor_name)))

    f = open(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.pkl".format(factor_name)), "wb")
    pickle.dump(result, f); f.close()

#%% summary plot for manuscript
plt.figure(figsize=(18, 9))
xlim = [datetime.datetime(2005,12,1), datetime.datetime(2023,1,31)]
plt.subplot(3,3,1)
factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = [datetime.datetime.fromtimestamp(j) for j in result["time_hist"]]
asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
plt.plot(time_hist, asset_hist, label="without T.C.", color=color_cycle[0], linestyle="--")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = [datetime.datetime.fromtimestamp(j) for j in result["time_hist"]]
asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
plt.plot(time_hist, asset_hist, label="with T.C.", color=color_cycle[0])
plt.ylabel("PnL ($)"); plt.grid(axis="x")
plt.legend(title="name space + parametric model")
plt.tick_params(direction="in", axis="both")
plt.xlim(xlim)

plt.subplot(3,3,2)
factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
time_hist = [datetime.datetime.fromtimestamp(j) for j in time_hist]
plt.plot(time_hist, asset_hist, label="without T.C.", color=color_cycle[1], linestyle="--")
plt.ylabel("PnL ($)"); plt.grid(axis="x")
plt.legend(title="rank space + parametric model")
plt.tick_params(direction="in", axis="both")
plt.xlim(xlim)

plt.subplot(3,3,3)
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
time_hist = [datetime.datetime.fromtimestamp(j) for j in time_hist]
plt.plot(time_hist, asset_hist, label="with T.C.", color=color_cycle[1])
plt.ylabel("PnL ($)"); plt.grid(axis="x")
plt.legend(title="rank space + parametric model")
plt.tick_params(direction="in", axis="both")
plt.xlim(xlim)

plt.subplot(3,3,4)
factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
time_hist = [datetime.datetime.fromtimestamp(j) for j in time_hist]
plt.plot(time_hist, asset_hist, label="without T.C.", color=color_cycle[0], linestyle="--")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time_hist = result["time_hist"]; asset_hist = result["asset_hist"]; return_hist = result["return_hist"]
time_hist = [datetime.datetime.fromtimestamp(j) for j in time_hist]
plt.plot(time_hist, asset_hist, label="with T.C.", color=color_cycle[0])
plt.ylabel("PnL ($)"); plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.legend(title="name space + neural networks")
plt.xlim(xlim)

plt.subplot(3,3,5)
factor_name = "PCA_rank_hybrid_Atlas_high_freq"
f = open(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.pkl".format(factor_name)), "rb")
result = pickle.load(f); f.close()
time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]; transaction_cost_hist = result["transaction_cost_hist"]
plt.plot(time_axis_PnL, asset_hist_R_name, label="without T.C.", color=color_cycle[1], linestyle="--")
plt.ylabel("PnL ($)"); plt.grid(axis="x"); plt.legend()
plt.yscale("log"); plt.tick_params(direction="in", axis="both")
plt.legend(title="rank space + neural networks")
plt.tick_params(direction="in", axis="both")

plt.subplot(3,3,6)
factor_name = "PCA_rank_hybrid_Atlas_high_freq"
f = open(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer.pkl".format(factor_name)), "rb")
result = pickle.load(f); f.close()
time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]; transaction_cost_hist = result["transaction_cost_hist"]
plt.plot(time_axis_PnL, asset_hist_R_name, label="with T.C.", color=color_cycle[1])
plt.ylabel("PnL ($)"); plt.grid(axis="x"); plt.legend()
plt.yscale("log"); plt.tick_params(direction="in", axis="both")
plt.tick_params(direction="in", axis="both")
plt.legend(title="rank space + neural networks")
plt.xlim(xlim)

'''
plt.subplot(3,2,7)
factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="o", color=color_cycle[0], linestyle="--", label="parametric model + without T.C.")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="o", color=color_cycle[0], label="parametric + with T.C.")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="*", color=color_cycle[0], linestyle="--", label="neural networks + without T.C.")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="*", color=color_cycle[0], linestyle="--", label="neural networks + with T.C.")
plt.ylabel("sharp ratio"); plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.legend(title="name space")
plt.xlim(xlim)

plt.subplot(3,3,8)
#plt.subplot(3,3,8, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="o", color=color_cycle[0], linestyle="--", label="parametric model + without T.C.")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="o", color=color_cycle[0], linestyle="--", label="neural network + without T.C.")

plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.legend(title="rank space")
plt.xlim(xlim)

plt.subplot(3,3,9)
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="o", color=color_cycle[0], label="parametric model + with T.C.")
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name)
plt.plot([datetime.datetime(int(j),7,1) for j in df["year"]], df["sharp_ratio"], marker="*", color=color_cycle[0], label="neural networks + with T.C.")
plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.legend(title="rank space")
plt.xlim(xlim)
'''

plt.subplot(3,2,5)
factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["name space \nparametric model"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[0])

factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["rank space \nparametric model"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[1])

factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["name space \nneural networks"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[0])

factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_without_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["rank space \nneural networks"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[1])
plt.ylabel("Sharpe ratio"); plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], color="black")
plt.legend(title="without T.C.", loc="upper left")

plt.subplot(3,2,6)
factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["name space \nparametric model"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[0])

factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-OU_process-OU_process_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["rank space \nparametric model"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[1])

factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["name space \nneural networks"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[0])

factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_with_transaction_cost.csv".format(factor_name))
df = pd.read_csv(file_name); sharp_ratio = df["sharp_ratio"]
plt.bar(["rank space \nneural networks"], [np.mean(sharp_ratio)], yerr=[np.std(sharp_ratio)], capsize = 5, color=color_cycle[1])
plt.ylabel("Sharpe ratio"); plt.grid(axis="x")
plt.tick_params(direction="in", axis="both")
plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], color="black")
plt.legend(title="with T.C.", loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_summary.pdf"), dpi=300)

#%% rank hybrid Atlas high freq - optimize rebalance interval for intraday rebalance
transaction_cost_factor = 0.0002; shorting_cost_factor = 0.0000; leverage = 1
t_eval_start = datetime.datetime(2006, 1, 1); t_eval_end = datetime.datetime(2022,12,15)
#t_eval_start = datetime.datetime(2006, 1, 3); t_eval_end = datetime.datetime(2012, 12, 31)

#rebalance_interval_list = [1, 5, 10, 30, 45, 60, 120, 180]
rebalance_interval_list = [1, 6, 12, 24, 45, 60, 120, 180]

def core(rebalance_interval):
    file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
    result = np.load(file_name, allow_pickle=True)
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]
    portfolio_weights_R_rank = list(portfolio_weights_R_rank.T)
    epsilon_idx = []
    for j in range(len(portfolio_weights_R_rank)):
        epsilon_idx.append(np.where(np.abs(portfolio_weights_R_rank[j]) > 1e-8)[0])
    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, rebalance_interval=rebalance_interval)
    return result

file_name = "PnL_{}-CNN_transformer_dependence_rebalance_interval.pkl".format(factor_name)
if os.path.exists(os.path.join(os.path.dirname(__file__), file_name)):
    result_all = pickle.load(open(os.path.join(os.path.dirname(__file__), file_name), "rb"))
else:
    cpu_core = len(rebalance_interval_list)
    result_all = Parallel(n_jobs=cpu_core)(delayed(core)(j) for j in rebalance_interval_list)
    f = open(os.path.join(os.path.dirname(__file__), file_name), "wb")
    pickle.dump(result_all, f); f.close()

#%%
terminal_PnL = []
annual_return_avg = []
sharp_ratio_avg = []
plt.figure(figsize=(9, 6))
for j in range(len(result_all)):
    result = result_all[j]
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    transaction_cost_hist = result["transaction_cost_hist"]; maintenance_cost_hist = result["maintenance_cost_hist"]
    plt.subplot(2, 1, 1)
    plt.plot(time_axis_PnL, asset_hist_R_name, label="{} min".format(5*rebalance_interval_list[j]))

    return_hist_R_name = np.array([np.nan] + [asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
    sharp_ratio_summary = []
    year_start = t_eval_start.year; year_end = t_eval_end.year
    for year in np.arange(year_start, year_end+1, 1):
        idx = np.where([j.year == year for j in time_axis_PnL])[0]
        return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
        vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
        sharp_ratio = return_annual/vol_annual
        sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
    df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
    terminal_PnL.append(asset_hist_R_name[-1])
    annual_return_avg.append(np.mean(df["return_annual"].to_list()))
    sharp_ratio_avg.append(np.mean(df["sharp_ratio"].to_list()))

plt.ylabel("PnL ($)"); plt.legend(ncol=3)
plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_dependence_rebalance_interval.pdf".format(factor_name)))

plt.figure(figsize=(8, 4.5))
plt.ylim([-10, 200]); plt.tick_params(direction="in", axis="both")
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(5 * np.array(rebalance_interval_list), sharp_ratio_avg, marker="o", color=color_cycle[0], label='Sharpe Ratio')
#ax2.plot(5 * np.array(rebalance_interval_list), annual_return_avg, marker="o", color=color_cycle[1], label='Annual Return')
ax2.plot(5 * np.array(rebalance_interval_list), terminal_PnL, marker="o", color=color_cycle[1], label='Terminal PnL')
ax1.set_ylabel("Sharpe Ratio", color=color_cycle[0])
#ax2.set_ylabel("Annual Return", color=color_cycle[1])
ax2.set_ylabel("Terminal PnL", color=color_cycle[1])
ax1.tick_params(axis='y', labelcolor=color_cycle[0])
ax2.tick_params(axis='y', labelcolor=color_cycle[1])
ax1.tick_params(direction="in", axis="both")
ax2.tick_params(direction="in", axis="both")
ax1.set_ylim([0, 3.5]); ax2.set_ylim([0, 200])
plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_dependence_rebalance_interval_2.pdf".format(factor_name)))


#%%
terminal_PnL = []
annual_return_avg = []
sharp_ratio_avg = []
plt.figure(figsize=(18, 9))
for j in range(len(result_all)):
    result = result_all[j]
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    transaction_cost_hist = result["transaction_cost_hist"]; maintenance_cost_hist = result["maintenance_cost_hist"]
    normalized_transaction_cost = (np.array(transaction_cost_hist)[1:])/(np.array(asset_hist_R_name)[:-1])
    normalized_maintenance_cost = (np.array(maintenance_cost_hist)[1:])/(np.array(asset_hist_R_name)[:-1])
    plt.subplot(2, 2, 1)
    plt.plot(time_axis_PnL[1:], normalized_transaction_cost, label="{} min".format(5*rebalance_interval_list[j]))

    plt.subplot(2, 2, 2)
    plt.plot(time_axis_PnL[1:], normalized_maintenance_cost, label="{} min".format(5*rebalance_interval_list[j]))

    plt.subplot(2, 2, 3)
    plt.plot(time_axis_PnL[1:], np.cumsum(normalized_transaction_cost), label="{} min".format(5*rebalance_interval_list[j]))

    plt.subplot(2, 2, 4)
    plt.plot(time_axis_PnL[1:], np.cumsum(normalized_maintenance_cost), label="{} min".format(5*rebalance_interval_list[j]))

plt.subplot(2, 2, 1)
plt.ylabel("normalized transaction cost")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(2, 2, 2)
plt.ylabel("normalized latency cost")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(2, 2, 3)
plt.ylabel("normalized transaction cost (cumulative)")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(2, 2, 4)
plt.ylabel("normalized latency cost (cumulative)")
plt.legend(title="rebalance interval", ncol=3)

#%% rank hybrid Atlas high freq - dependence on transaction cost
transaction_cost_factor_list = [0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
shorting_cost_factor = 0.0000; leverage = 1; rebalance_interval = 45
t_eval_start = datetime.datetime(2006, 1, 1); t_eval_end = datetime.datetime(2022,12,15)

def core(transaction_cost_factor):
    file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
    result = np.load(file_name, allow_pickle=True)
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R_rank = portfolio_weights_R_rank[:, t_idx]
    portfolio_weights_R_rank = list(portfolio_weights_R_rank.T)
    epsilon_idx = []
    for j in range(len(portfolio_weights_R_rank)):
        epsilon_idx.append(np.where(np.abs(portfolio_weights_R_rank[j]) > 1e-8)[0])
    result = utils.evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, rebalance_interval=rebalance_interval)
    return result

file_name = "PnL_{}-CNN_transformer_dependence_transaction_cost.pkl".format(factor_name)
if os.path.exists(os.path.join(os.path.dirname(__file__), file_name)):
    result_all = pickle.load(open(os.path.join(os.path.dirname(__file__), file_name), "rb"))
else:
    cpu_core = len(transaction_cost_factor_list)
    result_all = Parallel(n_jobs=cpu_core)(delayed(core)(j) for j in transaction_cost_factor_list)
    f = open(os.path.join(os.path.dirname(__file__), file_name), "wb")
    pickle.dump(result_all, f); f.close()
    
#%%
sharp_ratio_avg = []
plt.figure(figsize=(18, 3))
for j in range(len(result_all)):
    result = result_all[j]
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    transaction_cost_hist = result["transaction_cost_hist"]; maintenance_cost_hist = result["maintenance_cost_hist"]
    if transaction_cost_factor_list[j] > 0:
        plt.subplot(1,2,1)
        plt.plot(time_axis_PnL, asset_hist_R_name, label="{}".format(transaction_cost_factor_list[j]))
        return_hist_R_name = np.array([np.nan] + [asset_hist_R_name[j+1]/asset_hist_R_name[j]-1 for j in range(len(asset_hist_R_name)-1)])
        sharp_ratio_summary = []
        year_start = t_eval_start.year; year_end = t_eval_end.year
        for year in np.arange(year_start, year_end+1, 1):
            idx = np.where([j.year == year for j in time_axis_PnL])[0]
            return_annual = np.power(asset_hist_R_name[idx[-1]]/asset_hist_R_name[idx[0]], 252/(len(idx)-1))-1
            vol_annual = np.nanstd(return_hist_R_name[idx])*np.sqrt(252)
            sharp_ratio = return_annual/vol_annual
            sharp_ratio_summary.append([year, return_annual, vol_annual, sharp_ratio])
        df = pd.DataFrame(np.array(sharp_ratio_summary), columns=["year", "return_annual", "vol_annual", "sharp_ratio"])
        sharp_ratio_avg.append([np.mean(df["sharp_ratio"].to_list()), np.std(df["sharp_ratio"].to_list())])

plt.subplot(1,2,1)
plt.yscale("log")
plt.tick_params(direction="in", axis="both", which="both")
plt.legend(title="transaction cost factor")
plt.xlim(datetime.datetime(2005,1,1), datetime.datetime(2023,12,31))
plt.subplot(1,2,2)
plt.plot(np.array(transaction_cost_factor_list)[1:], np.array(sharp_ratio_avg)[:, 0], marker="o", color=color_cycle[0])
plt.errorbar(np.array(transaction_cost_factor_list)[1:], np.array(sharp_ratio_avg)[:, 0], yerr=np.array(sharp_ratio_avg)[:, 1], fmt="o", color=color_cycle[0], capsize=5)
plt.hlines(0, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], color="gray", linestyle="--")
plt.tick_params(direction="in", axis="both")
plt.savefig(os.path.join(os.path.dirname(__file__), "PnL_{}-CNN_transformer_dependence_transaction_cost.pdf".format(factor_name)))

#%%
sharp_ratio_avg

#%% characteristic of portfolio weights - OU process
PCA_TYPE = "name"; factor_name = "PCA_name"
#PCA_TYPE = "rank_hybrid_Atlas"; factor_name = "PCA_rank_hybrid_Atlas"
#PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"
#PCA_TYPE = "rank_permutation"; factor_name = "PCA_rank_permutation"
#PCA_TYPE = "rank_theta_transform"; factor_name = "PCA_rank_theta"

t_eval_start = datetime.datetime(2002,1,1); t_eval_end = datetime.datetime(2016,12,31)
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)

if PCA_TYPE == "name":
    time_axis = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
    portfolio_weights_epsilon = result["portfolio_weights_epsilon"]; portfolio_weights_R = result["portfolio_weights_R"]
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    time_axis = [time_axis[j] for j in t_idx]; portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]; portfolio_weights_R = portfolio_weights_R[:, t_idx]

    plt.figure(figsize=(9, 9))
    plt.subplot(3,1,1)
    active_position = np.array([np.sum(np.abs(portfolio_weights_epsilon[:, j])>0) for j in range(len(time_axis))])
    plt.plot(time_axis, active_position, label="active position number")
    plt.ylabel("active position number"); plt.grid(axis="x"); plt.legend()

    plt.subplot(3,1,2)
    market_neurality = []
    for t in tqdm.tqdm(time_axis):
        t_idx_temp = np.searchsorted(equity_data_.time_axis, t)
        result = PCA_factor_.residual_return(t)
        epsilon_idx = result["epsilon_idx"]
        R = copy.deepcopy(equity_data_.return_[epsilon_idx, :][:, (t_idx_temp-PCA_factor_.config["factor_evaluation_window_length"]+1):(t_idx_temp+1)]) - equity_data_.risk_free_rate[(t_idx_temp-PCA_factor_.config["factor_evaluation_window_length"]+1):(t_idx_temp+1)]
        U, S, V_T = np.linalg.svd(R, full_matrices=True)
        F = np.diag(S[0:PCA_factor_.factor_number]).dot(V_T[0:PCA_factor_.factor_number, :])
        omega = np.linalg.lstsq(R.T, F.T, rcond=None)[0].T
        F = copy.deepcopy(F[:, (-PCA_factor_.config["loading_evaluation_window_length"]):])
        R = copy.deepcopy(R[:, (-PCA_factor_.config["loading_evaluation_window_length"]):])
        beta = np.linalg.lstsq(F.T, R.T, rcond=None)[0].T
        market_neurality.append(np.linalg.norm(beta.T.dot(portfolio_weights_R[epsilon_idx, np.searchsorted(time_axis, t)]), 1))
    market_neurality = np.array(market_neurality)
    plt.plot(time_axis, market_neurality, label="market neurality")
    plt.ylabel(r"$||\beta_t^Tw_t^R||_1$"); plt.grid(axis="x"); plt.legend()

    plt.subplot(3,1,3)
    dollar_neutrality = np.sum(portfolio_weights_R, axis=0)/np.sum(np.abs(portfolio_weights_R), axis=0)
    plt.plot(time_axis, dollar_neutrality, label="dollar neutrality")
    plt.ylabel(r"$\frac{\sum_i w_{i,t}^R}{||w_t^R||_1}$"); plt.grid(axis="x"); plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_characteristic_{}-OU_process-OU_process.pdf".format(factor_name)))


