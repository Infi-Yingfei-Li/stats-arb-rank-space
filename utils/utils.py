#%% environment
import os, sys, copy, pickle, h5py, tqdm

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import data.data as data
import market_decomposition.market_factor_classic as market_factor_classic
import trading_signal.trading_signal as trading_signal
import portfolio_weights.portfolio_weights as portfolio_weights

#%%
def evaluate_PnL_name_space(equity_data, time_list, equity_idx, portfolio_weights_list, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, is_vanilla=False):
    '''
    params:
        equity_data: class equity data
        time_list: list of time in datetime format
        equity_idx: list of equity index which marks the investment space, each entry has len(equity_dta.equity_idx_list)
        portfolio_weights_list: list of portfolio weights in equity space
    '''
    if is_vanilla:
        time_hist = []; asset_hist = [1]; return_hist = [np.nan]
        for j in np.arange(0, len(time_list), 1):
            time_idx = np.searchsorted(equity_data.time_axis, time_list[j])
            equity_idx_current = np.array(equity_idx[j]); portfolio_weights_current = np.array(portfolio_weights_list[j])[equity_idx_current]
            return_ = np.sum(np.multiply(portfolio_weights_current, equity_data.return_[equity_idx_current, time_idx+1]))
            time_hist.append(equity_data.time_axis[time_idx]); return_hist.append(return_); asset_hist.append(copy.deepcopy(asset_hist[-1]*(1+return_)))
        time_hist.append(equity_data.time_axis[time_idx+1])
        return {"time_hist": time_hist, "asset_hist": asset_hist, "return_hist": return_hist}
    else:
        asset = 1; portfolio_dollar_weights_prev = np.zeros(len(equity_data.equity_idx_list))
        time_hist = []; transaction_cost_hist = []
        asset_hist = [1]; return_hist = [np.nan]
        for j in tqdm.tqdm(np.arange(0, len(time_list), 1)):
            # at day t, rebalance the portfolio after market close
            time_idx = np.searchsorted(equity_data.time_axis, time_list[j])
            portfolio_norm_weights_current = np.array(portfolio_weights_list[j])
            if np.linalg.norm(portfolio_norm_weights_current,ord=1) > 1e-8:
                portfolio_norm_weights_current /= np.linalg.norm(portfolio_norm_weights_current,ord=1)
            if leverage == "auto":
                leverage = 1/np.sum(portfolio_norm_weights_current)
            portfolio_dollar_weights_current = leverage*asset*portfolio_norm_weights_current
            transaction_cost = transaction_cost_factor*np.linalg.norm(portfolio_dollar_weights_current-portfolio_dollar_weights_prev,ord=1)+shorting_cost_factor*np.linalg.norm(np.minimum(portfolio_dollar_weights_current,0),ord=1)
            asset -= np.sum(portfolio_dollar_weights_current) + transaction_cost
            time_hist.append(equity_data.time_axis[time_idx]); transaction_cost_hist.append(transaction_cost)

            # at day t+1, update the balance sheet after market close
            #portfolio_dollar_weights_current = np.array([portfolio_dollar_weights_current[k]*(1+equity_data.return_[k, time_idx+1]) if np.abs(portfolio_dollar_weights_current[k])>1e-8 else 0 for k in range(len(equity_data.equity_idx_list))])
            for k in equity_idx[j]: portfolio_dollar_weights_current[k] *= (1+equity_data.return_[k, time_idx+1])
            asset = asset*(1+equity_data.risk_free_rate[time_idx+1]) + np.sum(portfolio_dollar_weights_current)
            portfolio_dollar_weights_prev = copy.deepcopy(portfolio_dollar_weights_current)
            asset_hist.append(copy.deepcopy(asset)); return_hist.append(asset_hist[-1]/asset_hist[-2]-1)

        time_hist.append(equity_data.time_axis[time_idx+1]); transaction_cost_hist.append(np.nan)
        return {"time_hist": time_hist, "asset_hist": asset_hist, "return_hist": return_hist, "transaction_cost_hist": transaction_cost_hist}

def evaluate_PnL_name_space_high_freq(equity_data, equity_data_high_freq, time_list, epsilon_idx_list, portfolio_weights_R_rank_list, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, rebalance_interval=1):
    asset_R_rank = 1; asset_R_name = 1
    time_hist = [time_list[0]]; asset_R_rank_hist = [1]; asset_R_name_hist = [1]; transaction_cost_hist = [0]; maintenance_cost_hist = [0]
    portfolio_dollar_weights_R_name = np.zeros(len(equity_data.equity_idx_list))
    for j in tqdm.tqdm(np.arange(0, len(time_list), 1)):
        # at day t, rebalance the portfolio at the market close
        transaction_cost_intraday = 0; maintenance_cost_intraday = 0
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
        asset_R_name = asset_R_name - np.sum(portfolio_dollar_weights_R_name) - tc
        transaction_cost_intraday += tc

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
                transaction_cost_intraday += tc
                maintenance_cost_intraday += (np.sum(portfolio_dollar_weights_R_name) - np.sum(portfolio_dollar_weights_R_name_old))/(8*60/rebalance_interval)
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

                #if k % rebalance_interval == 0:
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
                    maintenance_cost_intraday += np.sum(portfolio_dollar_weights_R_name) - np.sum(portfolio_dollar_weights_R_name_old)

        # at the market close, update the balance sheet
        asset_R_rank = asset_R_rank*(1+r_f) + np.sum(portfolio_dollar_weights_R_rank)
        asset_R_name = asset_R_name*(1+r_f) + np.sum(portfolio_dollar_weights_R_name)
        time_hist.append(equity_data_high_freq.time_axis_daily[t_idx+1]); asset_R_rank_hist.append(copy.deepcopy(asset_R_rank)); asset_R_name_hist.append(copy.deepcopy(asset_R_name))
        transaction_cost_hist.append(transaction_cost_intraday); maintenance_cost_hist.append(maintenance_cost_intraday)

    return {"time_hist": time_hist, "asset_hist_R_rank": asset_R_rank_hist, "asset_hist_R_name": asset_R_name_hist, "transaction_cost_hist": transaction_cost_hist, "maintenance_cost_hist": maintenance_cost_hist}

def evaluate_PnL_epsilon_space(market_factor, time_list, epsilon_idx, portfolio_weights_list, leverage=1):
    '''
    params:
        market_factor: class market factor
        time_list: list of time in datetime format
        epsilon_idx: list of epsilon index which marks the investment space
        portfolio_weights_list: list of portfolio weights in epsilon space
    '''
    time_hist = []; asset_hist = [1]; return_hist = [np.nan]; drop_out_rate_hist = []
    market_factor._initialize_residual_return_all()
    for j in tqdm.tqdm(np.arange(0, len(time_list), 1)):
        # at day t, rebalalnce the portfolio in epsilon space after market close
        time_idx = np.searchsorted(market_factor.time_axis, time_list[j])
        valid_idx = [k for k in range(len(epsilon_idx[j])) if ~np.isnan(market_factor.epsilon_all[epsilon_idx[j][k], time_idx+1])]
        drop_out_rate_hist.append(1-len(valid_idx)/len(epsilon_idx[j]))
        portfolio_norm_weights_current = np.array(portfolio_weights_list[j])[valid_idx]
        if np.linalg.norm(portfolio_norm_weights_current,ord=1) > 1e-8:
            portfolio_norm_weights_current /= np.linalg.norm(portfolio_norm_weights_current,ord=1)
        epsilon_return_ = market_factor.epsilon_all[np.array(epsilon_idx[j][valid_idx]), time_idx+1]
        return_ = np.sum(np.multiply(portfolio_norm_weights_current, epsilon_return_))
        time_hist.append(time_list[j])
        return_hist.append(epsilon_return_); asset_hist.append(asset_hist[-1]*(1+return_))
    time_hist.append(market_factor.time_axis[time_idx+1])
    return {"time_hist": time_hist, "asset_hist": asset_hist, "return_hist": return_hist, "drop_out_rate_hist": drop_out_rate_hist}

def evaluate_PnL_R_rank_space(equity_data, time_list, epsilon_idx, portfolio_weights_R_rank_list, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, mode='hybrid-Atlas'):
    '''
    params:
        equity_data: class equity data
        time_list: list of time in datetime format
        epsilon_idx: list of epsilon index which marks the investment space
        portfolio_weights_R_rank_list: list of portfolio weights in R rank space
    '''
    asset = 1
    time_hist = [time_list[0]]
    asset_hist = [1]; return_hist = [np.nan]
    for j in np.arange(0, len(time_list), 1):
        # at day t, rebalance the portfolio after market close
        time_idx = np.searchsorted(equity_data.time_axis, time_list[j])
        portfolio_norm_weights_R_rank_current = np.array(portfolio_weights_R_rank_list[j])
        if np.linalg.norm(portfolio_norm_weights_R_rank_current,ord=1) > 1e-8:
            portfolio_norm_weights_R_rank_current /= np.linalg.norm(portfolio_norm_weights_R_rank_current,ord=1)
        portfolio_dollar_weights_R_rank_current = leverage*asset*portfolio_norm_weights_R_rank_current
        asset -= np.sum(portfolio_dollar_weights_R_rank_current)

        # at day t+1, update the balance sheet after market close
        equity_idx_now = equity_data.equity_idx_by_rank[epsilon_idx[j], time_idx+1].astype(int)
        capitalization_now = equity_data.capitalization[equity_idx_now, time_idx+1]
        equity_idx_prev = equity_data.equity_idx_by_rank[epsilon_idx[j], time_idx].astype(int)
        capitalization_prev = equity_data.capitalization[equity_idx_prev, time_idx]
        return_by_rank = capitalization_now/capitalization_prev-1
        portfolio_dollar_weights_R_rank_current[epsilon_idx[j]] *= (1+return_by_rank)
        asset = asset + np.sum(portfolio_dollar_weights_R_rank_current)
        time_hist.append(equity_data.time_axis[time_idx+1])
        asset_hist.append(copy.deepcopy(asset)); return_hist.append(asset_hist[-1]/asset_hist[-2]-1)
    
    return {"time_hist": time_hist, "asset_hist": asset_hist, "return_hist": return_hist}

'''
portfolio_performance_config = {"is_update_network": False,
                                "transaction_cost_factor": 0.0005,
                                "shorting_cost_factor": 0.0001}
'''

class portfolio_performance:
    def __init__(self, equity_data, equity_data_high_freq, portfolio_performance_config):
        self.equity_data = equity_data; self.equity_data_high_freq = equity_data_high_freq; self.config = portfolio_performance_config
        self.save_file_name = os.path.join(os.path.dirname(__file__), "../results/portfolio_performance_label.npz")

    def register_portfolio(self, portfolio_label, portfolio_weights, eval_t_start, eval_t_end):
        print("Initiate registration of portfolio {} on {}/{} GPU.".format(portfolio_label, self.config["GPU_id"]+1, self.config["GPU_number"]))
        PCA_factor_type = portfolio_weights.factor.config["type"]
        eval_t_idx = np.arange(np.searchsorted(self.equity_data.time_axis, eval_t_start), np.searchsorted(self.equity_data.time_axis, eval_t_end)+1, 1)
        t_idx = copy.deepcopy(eval_t_idx[0])

        match PCA_factor_type:
            case "name":
                time_hist = []; epsilon_idx = []; portfolio_weights_epsilon = []; equity_idx = []; portfolio_weights_R = []
            case "rank_permutation":
                time_hist = []; epsilon_idx = []; portfolio_weights_epsilon = []; portfolio_weights_R_rank = []; equity_idx = []; portfolio_weights_R_name = []
            case "rank_hybrid_Atlas":
                time_hist = []; epsilon_idx = []; portfolio_weights_epsilon = []; portfolio_weights_R_rank = []; equity_idx = []; portfolio_weights_R_name = []
            case "rank_hybrid_Atlas_high_freq":
                time_hist = []; epsilon_idx = []; portfolio_weights_epsilon = []; portfolio_weights_R_rank = []
            case "rank_theta_transform":
                time_hist = []; epsilon_idx = []; portfolio_weights_epsilon = []; portfolio_weights_R_rank = []; equity_idx = []; portfolio_weights_R_name = []

        eval_t_idx_log = []
        while t_idx < eval_t_idx[-1]:
            eval_t_idx_log.append(t_idx)
            t_idx += self.config["reevaluation_interval"]

        train_t_interval_batch_len = np.ceil(len(eval_t_idx_log)/self.config["GPU_number"]).astype(int)
        for pt in np.arange(self.config["GPU_id"]*train_t_interval_batch_len, min((self.config["GPU_id"]+1)*train_t_interval_batch_len, len(eval_t_idx_log)), 1):
            t_idx = eval_t_idx_log[pt]
            train_t_start = self.equity_data.time_axis[t_idx-self.config["train_lookback_window"]]; train_t_end = self.equity_data.time_axis[t_idx-self.config["validation_lookback_window"]-1]
            valid_t_start = self.equity_data.time_axis[t_idx-self.config["validation_lookback_window"]]; valid_t_end = self.equity_data.time_axis[t_idx-1]

            # use during debug/sanity check
            #train_t_start = self.equity_data.time_axis[t_idx-self.config["train_lookback_window"]]; train_t_end = self.equity_data.time_axis[t_idx-1]
            #valid_t_start = self.equity_data.time_axis[t_idx-self.config["train_lookback_window"]]; valid_t_end = self.equity_data.time_axis[t_idx-1]
            #train_t_start = self.equity_data.time_axis[t_idx-99]; train_t_end = self.equity_data.time_axis[t_idx-1]
            #valid_t_start = self.equity_data.time_axis[t_idx-99]; valid_t_end = self.equity_data.time_axis[t_idx-1]

            portfolio_weights.train(train_t_start, train_t_end, valid_t_start, valid_t_end)
            result = portfolio_weights.portfolio_weights(self.equity_data.time_axis[t_idx], self.equity_data.time_axis[min(t_idx+self.config["reevaluation_interval"]-1, eval_t_idx[-1])])
            match PCA_factor_type:
                case "name":
                    time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"])
                    equity_idx.extend(result["equity_idx"]); portfolio_weights_R.extend(result["portfolio_weights_R"])
                case "rank_permutation":
                    time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"]); portfolio_weights_R_rank.extend(result["portfolio_weights_R_rank"])
                    equity_idx.extend(result["equity_idx"]); portfolio_weights_R_name.extend(result["portfolio_weights_R_name"])
                case "rank_hybrid_Atlas":
                    if "CNN" in portfolio_label:
                        time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"]); portfolio_weights_R_rank.extend(result["portfolio_weights_R_rank"])
                    else:
                        time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"]); portfolio_weights_R_rank.extend(result["portfolio_weights_R_rank"])
                        equity_idx.extend(result["equity_idx"]); portfolio_weights_R_name.extend(result["portfolio_weights_R_name"])
                case "rank_hybrid_Atlas_high_freq":
                    time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"]); portfolio_weights_R_rank.extend(result["portfolio_weights_R_rank"])
                case "rank_theta_transform":
                    time_hist.extend(result["time"]); epsilon_idx.extend(result["epsilon_idx"]); portfolio_weights_epsilon.extend(result["portfolio_weights_epsilon"]); portfolio_weights_R_rank.extend(result["portfolio_weights_R_rank"])
                    equity_idx.extend(result["equity_idx"]); portfolio_weights_R_name.extend(result["portfolio_weights_R_name"])

            print("Evaluate portfolio {} weights from {} to {} complete.".format(portfolio_label, self.equity_data.time_axis[t_idx], self.equity_data.time_axis[min(t_idx+self.config["reevaluation_interval"]-1, eval_t_idx[-1])]))

        match PCA_factor_type:
            case "name":
                time_hist_timestamp = [t.timestamp() for t in time_hist]
                portfolio_weights_epsilon_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_epsilon_ar[:] = np.nan
                portfolio_weights_R_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_ar[:] = np.nan
                for j in range(len(time_hist)):
                    portfolio_weights_epsilon_ar[:, j] = portfolio_weights_epsilon[j]
                    portfolio_weights_R_ar[:, j] = portfolio_weights_R[j]
                np.savez_compressed(self.save_file_name.replace("label", portfolio_label), time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R=portfolio_weights_R_ar)

            case "rank_permutation":
                time_hist_timestamp = [t.timestamp() for t in time_hist]
                portfolio_weights_epsilon_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_epsilon_ar[:] = np.nan
                portfolio_weights_R_rank_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_rank_ar[:] = np.nan
                portfolio_weights_R_name_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_name_ar[:] = np.nan
                for j in range(len(time_hist)):
                    portfolio_weights_epsilon_ar[:, j] = portfolio_weights_epsilon[j]
                    portfolio_weights_R_rank_ar[:, j] = portfolio_weights_R_rank[j]
                    portfolio_weights_R_name_ar[:, j] = portfolio_weights_R_name[j]
                np.savez_compressed(self.save_file_name.replace("label", portfolio_label), 
                                    time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R_rank=portfolio_weights_R_rank_ar, portfolio_weights_R_name=portfolio_weights_R_name_ar)

            case "rank_hybrid_Atlas":
                time_hist_timestamp = [t.timestamp() for t in time_hist]
                portfolio_weights_epsilon_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_epsilon_ar[:] = np.nan
                portfolio_weights_R_rank_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_rank_ar[:] = np.nan
                portfolio_weights_R_name_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_name_ar[:] = np.nan
                for j in range(len(time_hist)):
                    portfolio_weights_epsilon_ar[:, j] = portfolio_weights_epsilon[j]
                    portfolio_weights_R_rank_ar[:, j] = portfolio_weights_R_rank[j]
                    if not ("CNN" in portfolio_label):
                        portfolio_weights_R_name_ar[:, j] = portfolio_weights_R_name[j]
                if "CNN" in portfolio_label:
                    np.savez_compressed(self.save_file_name.replace("label", portfolio_label), 
                                        time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R_rank=portfolio_weights_R_rank_ar)
                else:
                    np.savez_compressed(self.save_file_name.replace("label", portfolio_label), 
                                        time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R_rank=portfolio_weights_R_rank_ar, portfolio_weights_R_name=portfolio_weights_R_name_ar)

            case "rank_hybrid_Atlas_high_freq":
                time_hist_timestamp = [t.timestamp() for t in time_hist]
                portfolio_weights_epsilon_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_epsilon_ar[:] = np.nan
                portfolio_weights_R_rank_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_rank_ar[:] = np.nan
                for j in range(len(time_hist)):
                    portfolio_weights_epsilon_ar[:, j] = portfolio_weights_epsilon[j]
                    portfolio_weights_R_rank_ar[:, j] = portfolio_weights_R_rank[j]
                np.savez_compressed(self.save_file_name.replace("label", portfolio_label),
                                    time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R_rank=portfolio_weights_R_rank_ar)

            case "rank_theta_transform":
                time_hist_timestamp = [t.timestamp() for t in time_hist]
                portfolio_weights_epsilon_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_epsilon_ar[:] = np.nan
                portfolio_weights_R_rank_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_rank_ar[:] = np.nan
                portfolio_weights_R_name_ar = np.zeros((len(self.equity_data.equity_idx_list), len(time_hist))); portfolio_weights_R_name_ar[:] = np.nan
                for j in range(len(time_hist)):
                    portfolio_weights_epsilon_ar[:, j] = portfolio_weights_epsilon[j]
                    portfolio_weights_R_rank_ar[:, j] = portfolio_weights_R_rank[j]
                    portfolio_weights_R_name_ar[:, j] = portfolio_weights_R_name[j]
                np.savez_compressed(self.save_file_name.replace("label", portfolio_label), 
                                    time=time_hist_timestamp, portfolio_weights_epsilon=portfolio_weights_epsilon_ar, portfolio_weights_R_rank=portfolio_weights_R_rank_ar, portfolio_weights_R_name=portfolio_weights_R_name_ar)

        print("Registration of portfolio {} complete.".format(portfolio_label))



    


