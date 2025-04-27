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


#%%
def evaluate_PnL_name_space_high_freq(equity_data, equity_data_high_freq, time_list, epsilon_idx_list, portfolio_weights_R_rank_list, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, rebalance_interval=1):
    asset_R_rank = 1; asset_R_name = 1
    time_hist = [time_list[0]]; asset_R_rank_hist = [1]; asset_R_name_hist = [1]
    transaction_cost_lower_bound_hist = [0]
    transaction_cost_overnight_hist = [0]; latency_cost_overnight_hist = [0]
    transaction_cost_intraday_hist = [0]; latency_cost_intraday_hist = [0]
    portfolio_dollar_weights_R_name = np.zeros(len(equity_data.equity_idx_list))
    for j in tqdm.tqdm(np.arange(0, len(time_list), 1)):
        # at day t, rebalance the portfolio at the market close
        transaction_cost_intraday = 0; latency_cost_intraday = 0
        t_idx_global = np.searchsorted(equity_data.time_axis, time_list[j])
        r_f = equity_data.risk_free_rate[t_idx_global+1]
        t_idx = np.searchsorted(equity_data_high_freq.time_axis_daily, time_list[j])
        result = equity_data_high_freq.intraday_data(equity_data_high_freq.time_axis_daily[t_idx])
        time_axis_high_freq = result["time"]
        capitalization_high_freq = result["capitalization"]
        rank_high_freq = result["rank"]
        equity_idx_by_rank_high_freq = result["equity_idx_by_rank"]

        epsilon_idx = epsilon_idx_list[j]
        portfolio_weights_norm_R_rank = np.array(portfolio_weights_R_rank_list[j])[epsilon_idx]
        if np.linalg.norm(portfolio_weights_norm_R_rank,ord=1) > 1e-8:
            portfolio_weights_norm_R_rank /= np.linalg.norm(portfolio_weights_norm_R_rank,ord=1)
        # if there is transaction cost, we instead treat asset_R_rank as auxiliary process and align it with asset_R_name at the beginning of every day; otherwise, asset_R_rank is the artificial PnL regardless of realizability
        if transaction_cost_factor + shorting_cost_factor > 0:
            asset_R_rank = copy.deepcopy(asset_R_name)
        portfolio_dollar_weights_R_rank = asset_R_rank*leverage*portfolio_weights_norm_R_rank
        asset_R_rank -= np.sum(portfolio_dollar_weights_R_rank)
        portfolio_dollar_weights_R_name_old = copy.deepcopy(portfolio_dollar_weights_R_name)
        equity_idx = equity_idx_by_rank_high_freq[epsilon_idx, -1].astype(int)
        portfolio_dollar_weights_R_name = np.zeros(len(equity_data.equity_idx_list))
        portfolio_dollar_weights_R_name[equity_idx] = portfolio_dollar_weights_R_rank
        tc = transaction_cost_factor*np.linalg.norm(portfolio_dollar_weights_R_name-portfolio_dollar_weights_R_name_old,ord=1)
        transaction_cost_lower_bound_hist.append(tc)
        asset_R_name = asset_R_name - np.sum(portfolio_dollar_weights_R_name) - tc
        #transaction_cost_intraday += tc

        equity_idx_high_freq_prev_close = equity_idx_by_rank_high_freq[epsilon_idx, -1].astype(int)
        capitalization_high_freq_prev_close = capitalization_high_freq[equity_idx_high_freq_prev_close, -1]

        result = equity_data_high_freq.intraday_data(equity_data_high_freq.time_axis_daily[t_idx+1])
        time_axis_high_freq = result["time"]
        capitalization_high_freq = result["capitalization"]
        rank_high_freq = result["rank"]
        equity_idx_by_rank_high_freq = result["equity_idx_by_rank"]

        # at day t+1, simulate the portfolio process in rank space and name space based on intra-day data
        for k in range(len(time_axis_high_freq)):
            if k == 0: # at the market open
                # update portfolio weights in rank space
                equity_idx_new = equity_idx_by_rank_high_freq[epsilon_idx, k].astype(int)
                capitalization_new = capitalization_high_freq[equity_idx_new, k]
                return_by_rank = capitalization_new/capitalization_high_freq_prev_close-1
                portfolio_dollar_weights_R_rank *= (1+return_by_rank)

                # update portfolio weights in name space
                capitalization_new = capitalization_high_freq[equity_idx, k]
                return_by_name = capitalization_new/capitalization_high_freq_prev_close-1
                for l in range(len(equity_idx)):
                    if np.isnan(return_by_name[l]):
                        return_by_name[l] = 0
                portfolio_dollar_weights_R_name[equity_idx] *= (1+return_by_name)

                # rebalance the portfolio in name space based on rank space
                portfolio_dollar_weights_R_name_old = copy.deepcopy(portfolio_dollar_weights_R_name)
                portfolio_dollar_weights_R_name = np.zeros(len(equity_data.equity_idx_list))
                equity_idx = equity_idx_new
                portfolio_dollar_weights_R_name[equity_idx] = portfolio_dollar_weights_R_rank

                # update asset
                tc = transaction_cost_factor*np.linalg.norm(portfolio_dollar_weights_R_name-portfolio_dollar_weights_R_name_old,ord=1)+shorting_cost_factor*np.linalg.norm(np.minimum(portfolio_dollar_weights_R_name_old,0),ord=1)*(8*60)/(24*60)
                asset_R_name = asset_R_name + (np.sum(portfolio_dollar_weights_R_name_old) - np.sum(portfolio_dollar_weights_R_name))/(8*60/rebalance_interval) - tc
                #transaction_cost_intraday += tc
                #latency_cost_intraday += (np.sum(portfolio_dollar_weights_R_name) - np.sum(portfolio_dollar_weights_R_name_old))/(8*60/rebalance_interval)
                transaction_cost_overnight_hist.append(tc)
                latency_cost_overnight_hist.append((np.sum(portfolio_dollar_weights_R_name) - np.sum(portfolio_dollar_weights_R_name_old))/(8*60/rebalance_interval))
            else:
                # update portfolio weights in rank space
                equity_idx_old = equity_idx_by_rank_high_freq[epsilon_idx, k-1].astype(int)
                capitalization_old = capitalization_high_freq[equity_idx_old, k-1]
                equity_idx_new = equity_idx_by_rank_high_freq[epsilon_idx, k].astype(int)
                capitalization_new = capitalization_high_freq[equity_idx_new, k]
                return_by_rank = capitalization_new/capitalization_old-1
                portfolio_dollar_weights_R_rank *= (1+return_by_rank)

                # update portfolio weights in name space
                capitalization_old = capitalization_high_freq[equity_idx, k-1]
                capitalization_new = capitalization_high_freq[equity_idx, k]
                return_by_name = capitalization_new/capitalization_old-1
                for l in range(len(equity_idx)):
                    if np.isnan(return_by_name[l]):
                        return_by_name[l] = 0
                portfolio_dollar_weights_R_name[equity_idx] *= (1+return_by_name)

                if k % rebalance_interval == 0 or k == len(time_axis_high_freq)-1:
                    # rebalance the portfolio in name space based on rank space
                    portfolio_dollar_weights_R_name_old = copy.deepcopy(portfolio_dollar_weights_R_name)
                    portfolio_dollar_weights_R_name = np.zeros(len(equity_data.equity_idx_list))
                    equity_idx = equity_idx_new
                    portfolio_dollar_weights_R_name[equity_idx] = portfolio_dollar_weights_R_rank

                    # update asset
                    tc = transaction_cost_factor*np.linalg.norm(portfolio_dollar_weights_R_name-portfolio_dollar_weights_R_name_old,ord=1)+shorting_cost_factor*np.linalg.norm(np.minimum(portfolio_dollar_weights_R_name_old,0),ord=1)/(24*60/rebalance_interval)
                    asset_R_name = asset_R_name + (np.sum(portfolio_dollar_weights_R_name_old) - np.sum(portfolio_dollar_weights_R_name)) - tc
                    transaction_cost_intraday += tc
                    latency_cost_intraday += np.sum(portfolio_dollar_weights_R_name) - np.sum(portfolio_dollar_weights_R_name_old)

        # at the market close, update the balance sheet
        asset_R_rank = asset_R_rank*(1+r_f) + np.sum(portfolio_dollar_weights_R_rank)
        asset_R_name = asset_R_name*(1+r_f) + np.sum(portfolio_dollar_weights_R_name)
        time_hist.append(equity_data_high_freq.time_axis_daily[t_idx+1]); asset_R_rank_hist.append(copy.deepcopy(asset_R_rank)); asset_R_name_hist.append(copy.deepcopy(asset_R_name))
        transaction_cost_intraday_hist.append(transaction_cost_intraday); latency_cost_intraday_hist.append(latency_cost_intraday)

    return {"time_hist": time_hist, "asset_hist_R_rank": asset_R_rank_hist, "asset_hist_R_name": asset_R_name_hist, 
            "transaction_cost_lower_bound_hist": transaction_cost_lower_bound_hist, 
            "transaction_cost_overnight_hist": transaction_cost_overnight_hist, "latency_cost_overnight_hist": latency_cost_overnight_hist,
            "transaction_cost_intraday_hist": transaction_cost_intraday_hist, "latency_cost_intraday_hist": latency_cost_intraday_hist}


#%%
transaction_cost_factor = 0.0002; shorting_cost_factor = 0.0000
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
    result = evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, rebalance_interval=rebalance_interval)
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
plt.figure(figsize=(18, 9))
for j in range(len(result_all)):
    result = result_all[j]
    time_axis_PnL = result["time_hist"]; asset_hist_R_rank = result["asset_hist_R_rank"]; asset_hist_R_name = result["asset_hist_R_name"]
    transaction_cost_lower_bound_hist = result["transaction_cost_lower_bound_hist"]
    transaction_cost_overnight_hist = result["transaction_cost_overnight_hist"]; latency_cost_overnight_hist = result["latency_cost_overnight_hist"]
    transaction_cost_intraday_hist = result["transaction_cost_intraday_hist"]; latency_cost_intraday_hist = result["latency_cost_intraday_hist"]

    asset_hist_R_name = np.array(asset_hist_R_name)
    transaction_cost_lower_bound_hist = np.array(transaction_cost_lower_bound_hist)
    transaction_cost_overnight_hist = np.array(transaction_cost_overnight_hist)
    latency_cost_overnight_hist = np.array(latency_cost_overnight_hist)
    transaction_cost_intraday_hist = np.array(transaction_cost_intraday_hist)
    latency_cost_intraday_hist = np.array(latency_cost_intraday_hist)

    plt.subplot(3, 2, 1)
    plt.plot(time_axis_PnL, asset_hist_R_name, label="{} min".format(5*rebalance_interval_list[j]))
    plt.subplot(3, 2, 2)
    plt.plot(time_axis_PnL[1:], np.cumsum(transaction_cost_lower_bound_hist[1:]/asset_hist_R_name[:-1]), label="{} min".format(5*rebalance_interval_list[j]))
    plt.subplot(3, 2, 3)
    plt.plot(time_axis_PnL[1:], np.cumsum(transaction_cost_overnight_hist[1:]/asset_hist_R_name[:-1]), label="{} min".format(5*rebalance_interval_list[j]))
    plt.subplot(3, 2, 4)
    plt.plot(time_axis_PnL[1:], np.cumsum(latency_cost_overnight_hist[1:]/asset_hist_R_name[:-1]), label="{} min".format(5*rebalance_interval_list[j]))
    plt.subplot(3, 2, 5)
    plt.plot(time_axis_PnL[1:], np.cumsum(transaction_cost_intraday_hist[1:]/asset_hist_R_name[:-1]), label="{} min".format(5*rebalance_interval_list[j]))
    plt.subplot(3, 2, 6)
    plt.plot(time_axis_PnL[1:], np.cumsum(latency_cost_intraday_hist[1:]/asset_hist_R_name[:-1]), label="{} min".format(5*rebalance_interval_list[j]))

plt.subplot(3, 2, 1)
plt.ylabel("PnL")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(3, 2, 2)
plt.ylabel("transaction cost\n lower bound")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(3, 2, 3)
plt.ylabel("transaction cost\n overnight")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(3, 2, 4)
plt.ylabel("maintenance cost\n overnight")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(3, 2, 5)
plt.ylabel("transaction cost\n intraday")
plt.legend(title="rebalance interval", ncol=3)

plt.subplot(3, 2, 6)
plt.ylabel("maintenance cost\n intraday")
plt.legend(title="rebalance interval", ncol=3)

#%%
rebalance_interval = 160

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
result = evaluate_PnL_name_space_high_freq(equity_data_, equity_data_high_freq_, time_axis, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=transaction_cost_factor, shorting_cost_factor=shorting_cost_factor, rebalance_interval=rebalance_interval)


