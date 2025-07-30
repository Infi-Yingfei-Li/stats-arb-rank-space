#%%
import os, sys, copy, h5py, datetime, pickle, tqdm, gc, pynvml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional
torch.set_printoptions(precision=7)

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import data.data as data
import market_decomposition.market_factor_classic as market_factor_classic
import trading_signal.trading_signal as trading_signal
import utils.utils as utils
import neural_network.neural_network as neural_network

#%%
'''
portfolio_weights_OU_process_config = {"mean_reversion_speed_filter_quantile": 0.2,
                                    "R2_filter": 0.7, 
                                    "threshold_open": 1.25,
                                    "threshold_close": 0.25,
                                    "max_holding_time": 60}
'''

class portfolio_weights_OU_process:
    def __init__(self, equity_data, equity_data_high_freq, factor, trading_signal, portfolio_weights_OU_process_config):
        self.equity_data = equity_data; self.equity_data_high_freq = equity_data_high_freq; self.factor = factor; self.trading_signal = trading_signal
        self.config = portfolio_weights_OU_process_config
        self.time_axis = copy.deepcopy(self.equity_data.time_axis)
        self.portfolio_weights_epsilon = np.zeros(len(self.equity_data.equity_idx_list))
        self.active_portfolio_size = 0

    def portfolio_weights(self, t_start, t_end):
        match self.factor.config["type"]:
            case "name":
                return self.portfolio_weights_PCA_name(t_start, t_end)
            case "rank_hybrid_Atlas":
                return self.portfolio_weights_PCA_rank_hybrid_Atlas(t_start, t_end)
            case "rank_hybrid_Atlas_high_freq":
                return self.portfolio_weights_PCA_rank_hybrid_Atlas_high_freq(t_start, t_end)
            case "rank_permutation":
                return self.portfolio_weights_PCA_rank_permutation(t_start, t_end)
            case "rank_theta_transform":
                return self.portfolio_weights_PCA_rank_theta_transform(t_start, t_end)

    def portfolio_weights_PCA_name(self, t_start, t_end, method="empirical_mean_variance"):
        if method == "Yeo_GP":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):
                result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
                result = self.trading_signal.trading_signal(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
                cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue
                
                kappa_cache = np.zeros((len(self.equity_data.equity_idx_list), self.config["mean_reversion_time_filter_lookback_window"])); kappa_cache[:] = np.nan
                for t_idx_lookback in np.arange(0, self.config["mean_reversion_time_filter_lookback_window"], 1):
                    result = self.trading_signal.trading_signal(self.time_axis[t_idx-t_idx_lookback])
                    valid_idx = np.where(result["R_sq"]>self.config["R2_filter"])[0]
                    kappa_cache[result["epsilon_idx"][valid_idx], -(t_idx_lookback+1)] = result["kappa"][valid_idx]
                
                kappa_cache = kappa_cache[epsilon_idx, :]
                kappa_avg = np.array([np.nanmean(kappa_cache[j, :]) if ~np.isnan(kappa_cache[j, -1]) else np.nan for j in range(kappa_cache.shape[0])])
                tau_avg = 252/kappa_avg
                sort_idx = np.argsort(tau_avg)

                # update new position
                cache = []
                for j in range(len(sort_idx)):
                    if (np.isnan(kappa[sort_idx[j]])) or (R_sq[sort_idx[j]]<self.config["R2_filter"]) or tau_avg[sort_idx[j]] < 3:
                        continue
                    trading_signal = (cummulative_epsilon_end[sort_idx[j]]-mu[sort_idx[j]])/sigma[sort_idx[j]]
                    if self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] = 0
                            self.active_portfolio_size -= 1
                    elif self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] = 0
                            self.active_portfolio_size -= 1
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            cache.append([epsilon_idx[sort_idx[j]], -1])
                            #self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] = -1
                            #self.active_portfolio_size += 1
                        if trading_signal < -self.config["threshold_open"]:
                            cache.append([epsilon_idx[sort_idx[j]], 1])
                            #self.portfolio_weights_epsilon[epsilon_idx[sort_idx[j]]] = 1
                            #self.active_portfolio_size += 1

                # close position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if j not in epsilon_idx:
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1
                
                while self.active_portfolio_size < self.config["active_portfolio_size_threshold"] and len(cache)>0:
                    result = cache.pop(0)
                    self.portfolio_weights_epsilon[result[0]] = result[1]
                    self.active_portfolio_size += 1

                portfolio_weights_R = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R, 1)>0: portfolio_weights_R /= np.linalg.norm(portfolio_weights_R, ord=1)
                portfolio_weights_R_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_all[epsilon_idx] = portfolio_weights_R
                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_hist.append(list(copy.deepcopy(portfolio_weights_R_all)))

            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, 
                    "equity_idx": epsilon_idx_hist, "portfolio_weights_R": portfolio_weights_R_hist}

        if method == "Avellaneda_Lee":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):
                result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
                result = self.trading_signal.trading_signal(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
                cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue

                # update new position
                for j in range(len(epsilon_idx)):
                    #if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]) or 252/kappa[j] > 30:
                    #if (np.isnan(kappa[j])) or 252/kappa[j] > 30:
                    if np.isnan(kappa[j]):
                        continue
                    trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                    if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                            self.active_portfolio_size += 1
                        if trading_signal < -self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 1
                            self.active_portfolio_size += 1

                # close position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if j not in epsilon_idx:
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1
                        if j in epsilon_idx:
                            idx = np.where(epsilon_idx==j)[0]
                            #if (R_sq[idx]<self.config["R2_filter"]) or (252/kappa[idx] > 30):
                            #if (252/kappa[idx] > 30):
                            #    self.portfolio_weights_epsilon[j] = 0
                            #    self.active_portfolio_size -= 1
                
                portfolio_weights_R = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R, 1)>0: portfolio_weights_R /= np.linalg.norm(portfolio_weights_R, ord=1)
                portfolio_weights_R_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_all[epsilon_idx] = portfolio_weights_R
                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_hist.append(list(copy.deepcopy(portfolio_weights_R_all)))
                
            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, 
                    "equity_idx": epsilon_idx_hist, "portfolio_weights_R": portfolio_weights_R_hist}

        if method == "empirical_mean_variance":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):
                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]; Phi = result["Phi"]
                cumulative_epsilon = np.cumsum(epsilon, axis=1)
                mu = np.nanmean(cumulative_epsilon, axis=1); sigma = np.nanstd(cumulative_epsilon, axis=1)
                cummulative_epsilon_end = np.cumsum(epsilon, axis=1)[:,-1]

                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue

                # update new position
                for j in range(len(epsilon_idx)):
                    if np.isnan(cummulative_epsilon_end[j]) or np.isnan(mu[j]) or np.isnan(sigma[j]):
                        continue
                    trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                    if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                            self.active_portfolio_size += 1
                        if trading_signal < -self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 1
                            self.active_portfolio_size += 1

                # close position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if j not in epsilon_idx:
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1
                
                portfolio_weights_R = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R, 1)>0: portfolio_weights_R /= np.linalg.norm(portfolio_weights_R, ord=1)
                portfolio_weights_R_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_all[epsilon_idx] = portfolio_weights_R
                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_hist.append(list(copy.deepcopy(portfolio_weights_R_all)))
                
            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, 
                    "equity_idx": epsilon_idx_hist, "portfolio_weights_R": portfolio_weights_R_hist}

    def portfolio_weights_PCA_rank_hybrid_Atlas(self, t_start, t_end, rank2name_strategy="max_occupation_rate", method="Avellaneda_Lee"):
        t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
        equity_idx_hist = []; portfolio_weights_R_name_hist = []
        for t_idx in tqdm.tqdm(t_idx_list):
            result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
            result = self.trading_signal.trading_signal(self.time_axis[t_idx])
            epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
            cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
            if np.isnan(epsilon_idx[0]):
                epsilon_idx_hist.append(epsilon_idx)
                portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                portfolio_weights_R_name_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                continue

            # update new position
            for j in range(len(epsilon_idx)):
                if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]) or (252/kappa[j]>30):
                    continue
                trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                    if trading_signal > -self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                    if trading_signal < self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                else:
                    if trading_signal > self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                    if trading_signal < -self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 1

            # clear position for epsilon portfolios out of track
            for j in range(len(self.portfolio_weights_epsilon)):
                if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                    if (j not in epsilon_idx) or (R_sq[j]<self.config["R2_filter"]) or (252/kappa[j]>30):
                        self.portfolio_weights_epsilon[j] = 0

            portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
            if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
            portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank
            if np.linalg.norm(portfolio_weights_R_rank_all, 1)>0: portfolio_weights_R_rank_all /= np.linalg.norm(portfolio_weights_R_rank_all, ord=1)

            # transform from rank space to name space
            if rank2name_strategy == "max_occupation_rate":
                # criteria: in the past 30 days, choose the equity that has the highest occupation rate at rank k
                portfolio_weights_R_name_all = np.zeros(len(self.equity_data.equity_idx_list))
                theta_lookback = 30
                # Note: it introduce a slight look-ahead bias, but it is not significant because we take the maximum occupation rate in the past 30 days and the next day's contribution is small
                # To strictly avoid look-ahead bias, de-annotate the annotated code below
                result = self.equity_data.occupation_rate_by_rank(epsilon_idx, self.time_axis[t_idx-theta_lookback+1], self.time_axis[t_idx+1])
                equity_idx_temp = np.array(result["equity_idx"]); theta = result["occupation_rate"]
                argmax_idx = np.argmax(theta, axis=1)
                portfolio_weights_R_name_all[equity_idx_temp[argmax_idx]] = portfolio_weights_R_rank
                '''
                sort_idx = np.argsort(theta, axis=1)
                for j in range(len(epsilon_idx)):
                    pt = len(equity_idx_temp)-1
                    while pt >= 0:
                        if ~np.isnan(self.equity_data.return_[equity_idx_temp[sort_idx[j, pt]], t_idx+1]):
                            portfolio_weights_R_name_all[equity_idx_temp[sort_idx[j, pt]]] = portfolio_weights_R_rank[j]
                            break
                        pt -= 1
                '''
            if rank2name_strategy == "last_occupation":
                portfolio_weights_R_name_all = np.zeros(len(self.equity_data.equity_idx_list))
                equity_idx_temp = self.equity_data.equity_idx_by_rank[epsilon_idx, t_idx].astype(int)
                portfolio_weights_R_name_all[equity_idx_temp] = portfolio_weights_R_rank

            if np.linalg.norm(portfolio_weights_R_name_all, 1)>0: portfolio_weights_R_name_all /= np.linalg.norm(portfolio_weights_R_name_all, ord=1)

            epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))
            equity_idx_hist.append(list(copy.deepcopy(equity_idx_temp))); portfolio_weights_R_name_hist.append(list(copy.deepcopy(portfolio_weights_R_name_all)))
        return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist,"portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist,
                "equity_idx": equity_idx_hist, "portfolio_weights_R_name": portfolio_weights_R_name_hist}

    def portfolio_weights_PCA_rank_hybrid_Atlas_high_freq(self, t_start, t_end,  method="empirical_mean_variance"):
        if method == "Yeo_GP":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):
                result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
                result = self.trading_signal.trading_signal(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
                cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue

                kappa_cache = np.zeros((len(self.equity_data.equity_idx_list), self.config["mean_reversion_speed_filter_lookback_window"])); kappa_cache[:] = np.nan
                for t_idx_lookback in np.arange(0, self.config["mean_reversion_speed_filter_lookback_window"], 1):
                    result_lookback = self.trading_signal.trading_signal(self.time_axis[t_idx-t_idx_lookback])
                    valid_idx = np.where(result_lookback["R_sq"]>self.config["R2_filter"])[0]
                    kappa_cache[result_lookback["epsilon_idx"][valid_idx], -(t_idx_lookback+1)] = result_lookback["kappa"][valid_idx]
                kappa_lookback = np.nanmean(kappa_cache, axis=1)
                tau_lookback = 252/kappa_lookback
                sort_idx = np.argsort(tau_lookback)
                tau_lookback_rank = np.zeros(len(tau_lookback)); tau_lookback_rank[:] = np.nan; tau_lookback_rank[sort_idx] = np.arange(len(tau_lookback))

                # update new position
                for j in range(len(epsilon_idx)):
                    if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]) or (tau_lookback_rank[epsilon_idx[j]]>len(epsilon_idx)*self.config["mean_reversion_speed_filter_quantile"]):
                        continue
                    trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                    if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                    elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                        if trading_signal < -self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 1

                # clear position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if (j not in epsilon_idx) :
                            self.portfolio_weights_epsilon[j] = 0

                portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
                portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank
                if np.linalg.norm(portfolio_weights_R_rank_all, 1)>0: portfolio_weights_R_rank_all /= np.linalg.norm(portfolio_weights_R_rank_all, ord=1)

                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))

            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist}

        if method == "Avellaneda_Lee":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):
                result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
                result = self.trading_signal.trading_signal(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
                cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue

                # update new position
                for j in range(len(epsilon_idx)):
                    #if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]) or 252/kappa[j] > 30:
                    #if (np.isnan(kappa[j])) or 252/kappa[j] > 30:
                    if np.isnan(kappa[j]):
                        continue
                    trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                    if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                            self.active_portfolio_size += 1
                        if trading_signal < -self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 1
                            self.active_portfolio_size += 1

                # clear position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if j not in epsilon_idx:
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1
                        if j in epsilon_idx:
                            idx = np.where(epsilon_idx==j)[0]
                            #if (R_sq[idx]<self.config["R2_filter"]) or (252/kappa[idx] > 30):
                            #if 252/kappa[idx] > 30:
                            #    self.portfolio_weights_epsilon[j] = 0
                            #    self.active_portfolio_size -= 1

                portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
                portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank
                if np.linalg.norm(portfolio_weights_R_rank_all, 1)>0: portfolio_weights_R_rank_all /= np.linalg.norm(portfolio_weights_R_rank_all, ord=1)

                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))

            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist}

        if method == "empirical_mean_variance":
            t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
            for t_idx in tqdm.tqdm(t_idx_list):

                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]; Phi = result["Phi"]
                cumulative_epsilon = np.cumsum(epsilon, axis=1)
                mu = np.nanmean(cumulative_epsilon, axis=1); sigma = np.nanstd(cumulative_epsilon, axis=1)
                cummulative_epsilon_end = np.cumsum(epsilon, axis=1)[:, -1]
                if np.isnan(epsilon_idx[0]):
                    epsilon_idx_hist.append(epsilon_idx)
                    portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                    portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                    continue

                # update new position
                for j in range(len(epsilon_idx)):
                    if np.isnan(cummulative_epsilon_end[j]) or np.isnan(mu[j]) or np.isnan(sigma[j]):
                        continue
                    trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                    if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                        if trading_signal > -self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                        if trading_signal < self.config["threshold_close"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                            self.active_portfolio_size -= 1
                    else:
                        if trading_signal > self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                            self.active_portfolio_size += 1
                        if trading_signal < -self.config["threshold_open"]:
                            self.portfolio_weights_epsilon[epsilon_idx[j]] = 1
                            self.active_portfolio_size += 1

                # clear position for epsilon portfolios out of track
                for j in range(len(self.portfolio_weights_epsilon)):
                    if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                        if j not in epsilon_idx:
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1

                portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
                if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
                portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank
                if np.linalg.norm(portfolio_weights_R_rank_all, 1)>0: portfolio_weights_R_rank_all /= np.linalg.norm(portfolio_weights_R_rank_all, ord=1)

                epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))

            return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist}

    def portfolio_weights_PCA_rank_permutation(self, t_start, t_end,  method="Avellaneda_Lee"):
        t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
        equity_idx_hist = []; portfolio_weights_R_name_hist = []
        for t_idx in tqdm.tqdm(t_idx_list):
            result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
            result = self.trading_signal.trading_signal(self.time_axis[t_idx])
            epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
            cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
            if np.isnan(epsilon_idx[0]):
                epsilon_idx_hist.append(epsilon_idx)
                portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                portfolio_weights_R_name_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                continue

            # update new position
            for j in range(len(epsilon_idx)):
                if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]):
                    continue
                trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                    if trading_signal > -self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                        self.active_portfolio_size -= 1
                elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                    if trading_signal < self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                        self.active_portfolio_size -= 1
                else:
                    if trading_signal > self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                        self.active_portfolio_size += 1
                    if trading_signal < -self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 1
                        self.active_portfolio_size += 1
            
            # clear position for epsilon portfolios out of track
            for j in range(len(self.portfolio_weights_epsilon)):
                if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                    if j not in epsilon_idx:
                        self.portfolio_weights_epsilon[j] = 0
                        self.active_portfolio_size -= 1
                    if j in epsilon_idx:
                        idx = np.where(epsilon_idx==j)[0]
                        if (R_sq[idx]<self.config["R2_filter"]) or (252/kappa[idx] > 30):
                            self.portfolio_weights_epsilon[j] = 0
                            self.active_portfolio_size -= 1

            portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
            if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
            portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank

            # transform from rank space to name space
            equity_idx = self.equity_data.equity_idx_by_rank[epsilon_idx, t_idx].astype(int)
            portfolio_weights_R_name_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_name_all[equity_idx] = portfolio_weights_R_rank

            if np.linalg.norm(portfolio_weights_R_name_all, 1)>0: portfolio_weights_R_name_all /= np.linalg.norm(portfolio_weights_R_name_all, ord=1)
            epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))
            equity_idx_hist.append(list(copy.deepcopy(equity_idx))), portfolio_weights_R_name_hist.append(list(copy.deepcopy(portfolio_weights_R_name_all)))

        return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist,
                "equity_idx": equity_idx_hist, "portfolio_weights_R_name": portfolio_weights_R_name_hist}
    
    def portfolio_weights_PCA_rank_theta_transform(self, t_start, t_end):
        t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        epsilon_idx_hist = []; portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
        equity_idx_hist = []; portfolio_weights_R_name_hist = []
        for t_idx in tqdm.tqdm(t_idx_list):
            result = self.factor.residual_return(self.time_axis[t_idx]); Phi = result["Phi"]
            result = self.trading_signal.trading_signal(self.time_axis[t_idx])
            epsilon_idx = result["epsilon_idx"]; kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
            cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
            if np.isnan(epsilon_idx[0]):
                epsilon_idx_hist.append(epsilon_idx)
                portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon)))
                portfolio_weights_R_rank_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                portfolio_weights_R_name_hist.append(list(np.zeros(len(self.equity_data.equity_idx_list))))
                continue

            kappa_cache = np.zeros((len(self.equity_data.equity_idx_list), self.config["mean_reversion_speed_filter_lookback_window"])); kappa_cache[:] = np.nan
            for t_idx_lookback in np.arange(0, self.config["mean_reversion_speed_filter_lookback_window"], 1):
                result_lookback = self.trading_signal.trading_signal(self.time_axis[t_idx-t_idx_lookback])
                valid_idx = np.where(result_lookback["R_sq"]>self.config["R2_filter"])[0]
                kappa_cache[result_lookback["epsilon_idx"][valid_idx], -(t_idx_lookback+1)] = result_lookback["kappa"][valid_idx]
            kappa_lookback = np.nanmean(kappa_cache, axis=1)
            tau_lookback = 252/kappa_lookback
            sort_idx = np.argsort(tau_lookback)
            tau_lookback_rank = np.zeros(len(tau_lookback)); tau_lookback_rank[:] = np.nan; tau_lookback_rank[sort_idx] = np.arange(len(tau_lookback))

            # update new position
            for j in range(len(epsilon_idx)):
                if (np.isnan(kappa[j])) or (R_sq[j]<self.config["R2_filter"]) or (tau_lookback_rank[epsilon_idx[j]]>len(epsilon_idx)*self.config["mean_reversion_speed_filter_quantile"]):
                    continue
                trading_signal = (cummulative_epsilon_end[j]-mu[j])/sigma[j]
                if self.portfolio_weights_epsilon[epsilon_idx[j]] > 0:
                    if trading_signal > -self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                elif self.portfolio_weights_epsilon[epsilon_idx[j]] < 0:
                    if trading_signal < self.config["threshold_close"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 0
                else:
                    if trading_signal > self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = -1
                    if trading_signal < -self.config["threshold_open"]:
                        self.portfolio_weights_epsilon[epsilon_idx[j]] = 1

            # clear position for epsilon portfolios out of track
            for j in range(len(self.portfolio_weights_epsilon)):
                if np.abs(self.portfolio_weights_epsilon[j]) > 1e-8:
                    if j not in epsilon_idx:
                        self.portfolio_weights_epsilon[j] = 0

            portfolio_weights_R_rank = Phi.T.dot(self.portfolio_weights_epsilon[epsilon_idx]).flatten()
            if np.linalg.norm(portfolio_weights_R_rank, 1)>0: portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, ord=1)
            portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx] = portfolio_weights_R_rank

            # transform from rank space to name space
            portfolio_weights_R_name_all = np.zeros(len(self.equity_data.equity_idx_list))
            theta_lookback = 30
            result = self.equity_data.occupation_rate_by_rank(epsilon_idx, self.time_axis[t_idx-theta_lookback+1], self.time_axis[t_idx+1])
            equity_idx = np.array(result["equity_idx"]); theta = result["occupation_rate"]
            portfolio_weights_R_name_all[equity_idx] = theta.T.dot(portfolio_weights_R_rank)

            if np.linalg.norm(portfolio_weights_R_name_all, 1)>0: portfolio_weights_R_name_all /= np.linalg.norm(portfolio_weights_R_name_all, ord=1)
            epsilon_idx_hist.append(epsilon_idx); portfolio_weights_epsilon_hist.append(list(copy.deepcopy(self.portfolio_weights_epsilon))); portfolio_weights_R_rank_hist.append(list(copy.deepcopy(portfolio_weights_R_rank_all)))
            equity_idx_hist.append(list(copy.deepcopy(equity_idx))), portfolio_weights_R_name_hist.append(list(copy.deepcopy(portfolio_weights_R_name_all)))

        return {"time": [self.time_axis[t_idx] for t_idx in t_idx_list], "epsilon_idx": epsilon_idx_hist, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist,
                "equity_idx": equity_idx_hist, "portfolio_weights_R_name": portfolio_weights_R_name_hist}

    def train(self, train_t_start=None, train_t_end=None, valid_t_start=None, valid_t_end=None):
        pass


#%%
'''
portfolio_weights_feed_forward_network_config = {"PnL_evaluation_window_length": 24,
                                                 "risk_aversion_factor": 2,
                                                 "train_t_start": datetime.datetime(1991, 7, 1),
                                                 "train_t_end": datetime.datetime(1991, 11, 1),
                                                 "valid_t_start": datetime.datetime(1991, 11, 1),
                                                 "valid_t_end": datetime.datetime(1991, 12, 15),
                                                 "epoch_max": 20,
                                                 "temporal_batch_num": 3,
                                                 "learning_rate": 1e-3,
                                                 "drop_out_rate": 0.25,
                                                 }
'''

class portfolio_weights_feed_forward_network:
    def __init__(self, equity_data, factor, trading_signal, portfolio_weights_feed_forward_network_config):
        self.equity_data = equity_data; self.factor = factor; self.trading_signal = trading_signal
        self.config = portfolio_weights_feed_forward_network_config
        self.time_axis = self.equity_data.time_axis
        f = open(os.path.join(os.path.dirname(__file__), "portfolio_weights_feed_forward_network.pkl"), "wb"); pickle.dump(self.config, f); f.close()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"

    def portfolio_weights(self, t_start, t_end):
        self.ffn.eval()
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        result = self.factor.residual_return_batch(self.time_axis[t_idx[0]], self.time_axis[t_idx[-1]])
        time = result["time"]; Phi = result["Phi"]
        result = self.trading_signal.trading_signal_batch(self.time_axis[t_idx[0]], self.time_axis[t_idx[-1]])
        equity_idx = result["equity_idx"]; epsilon_idx = result["epsilon_idx"]; fft = [torch.from_numpy(j).to(torch.float32).to(self.device) for j in result["fft"]]
        portfolio_weights_epsilon_hist = []; portfolio_weights_R_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.ffn.forward(fft[j]).cpu().detach().numpy().flatten(); portfolio_weights_epsilon /= np.linalg.norm(portfolio_weights_epsilon, 1)
            portfolio_weights_R = Phi[j].T.dot(portfolio_weights_epsilon); portfolio_weights_R /= np.linalg.norm(portfolio_weights_R, 1)
            portfolio_weights_epsilon_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_epsilon_all[epsilon_idx[j]] = portfolio_weights_epsilon
            portfolio_weights_R_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_all[equity_idx[j]] = portfolio_weights_R
            portfolio_weights_epsilon_hist.append(portfolio_weights_epsilon_all)
            portfolio_weights_R_hist.append(portfolio_weights_R_all)

        return {"time": [self.time_axis[j] for j in t_idx], "equity_idx": equity_idx, "epsilon_idx": epsilon_idx, 
                "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R": portfolio_weights_R_hist}

    def train(self, train_t_start=None, train_t_end=None, valid_t_start=None, valid_t_end=None):
        if train_t_start == None:
            train_t_start = self.config["train_t_start"]; train_t_end = self.config["train_t_end"]; valid_t_start = self.config["valid_t_start"]; valid_t_end = self.config["valid_t_end"]
        t_idx = np.arange(np.searchsorted(self.time_axis, train_t_start), np.searchsorted(self.time_axis, train_t_end)+1, 1)
        print("Training neural network from %s to %s." % (datetime.datetime.strftime(self.time_axis[t_idx[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx[-1]], "%Y-%m-%d")))

        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/FFT_feed_forward_network/", self.factor.config["type"], "/portfolio_weights_feed_forward_network_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
        self.ffn = neural_network.feed_forward_neural_network(self.factor.config["residual_return_evaluation_window_length"], self.config["drop_out_rate"]).to(self.device)
        self.optimizer = optim.Adam(self.ffn.parameters(), lr=self.config["learning_rate"])

        if os.path.exists(network_file_name):
            checkpoint = torch.load(network_file_name)
            self.ffn.load_state_dict(checkpoint["model_state_dict"])
            self.ffn.eval()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            target_in_sample_hist = checkpoint["in-sample target"]
            target_out_of_sample_hist = checkpoint["out-of-sample target"]
            portfolio_weights_epsilon_long_proportiion = checkpoint["portfolio_weights_epsilon_long_proportion"]
        else:
            self._initialize_ffn()
            self.epoch = 0
            target_in_sample_hist = []; target_out_of_sample_hist = []; portfolio_weights_epsilon_long_proportiion = []

        for epoch in range(self.config["epoch_max"]-self.epoch):
            self.ffn.train()
            target_epoch_hist = []
            temporal_batch_num = self.config["temporal_batch_num"]; temporal_batch_size = int(np.ceil(len(t_idx)/temporal_batch_num))
            temporal_batch_pt = 0
            for count in tqdm.tqdm(range(len(t_idx))):
                if count%temporal_batch_size == 0:
                    t_idx_sub = t_idx[(temporal_batch_pt*temporal_batch_size):min(((temporal_batch_pt+1)*temporal_batch_size), len(t_idx))]
                    temporal_batch_pt += 1
                    tqdm.tqdm.write("Training sample from {} to {}.".format(datetime.datetime.strftime(self.time_axis[t_idx_sub[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx_sub[-1]], "%Y-%m-%d")))
                target = self._target_func(self.time_axis[np.random.choice(t_idx_sub)])
                self.optimizer.zero_grad()
                target.backward()
                self.optimizer.step()
                target_epoch_hist.append(-float(target.cpu().detach().numpy()))

            #result = self.portfolio_weights(train_t_start, train_t_end)
            #time = result["time"]; portfolio_weights_R = result["portfolio_weights_R"]
            #return_hist = utils.evaluate_PnL_vanilla(self.equity_data, time, portfolio_weights_R)
            #target_in_sample = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))
            target_in_sample = np.mean(target_epoch_hist)

            result = self.portfolio_weights(valid_t_start, valid_t_end)
            time = result["time"]; portfolio_weights_R = result["portfolio_weights_R"]; portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
            return_hist = utils.evaluate_PnL_vanilla(self.equity_data, time, portfolio_weights_R)
            target_out_of_sample = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))

            target_in_sample_hist.append(target_in_sample); target_out_of_sample_hist.append(target_out_of_sample)
            portfolio_weights_epsilon_long_proportiion.append(np.mean([np.sum(np.maximum(j, 0))/np.linalg.norm(j, 1) for j in portfolio_weights_epsilon]))
            tqdm.tqdm.write("Epoch: %d, epoch train target: %f, in-sample target: %f, out-of-sample target: %f, long proportion of epsilon-weights: %f" % (self.epoch+epoch, np.mean(target_epoch_hist), target_in_sample, target_out_of_sample, portfolio_weights_epsilon_long_proportiion[-1]))

            torch.save({"epoch": self.epoch+epoch+1, "model_state_dict": self.ffn.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                        'in-sample target': target_in_sample_hist, 'out-of-sample target': target_out_of_sample_hist, "portfolio_weights_epsilon_long_proportion": portfolio_weights_epsilon_long_proportiion},
                        network_file_name)
            gc.collect()
        print("Training neural network complete.")

    def _initialize_ffn(self):
        def init_weights(m):
            if type(m) == nn.Linear: torch.abs(torch.nn.init.xavier_uniform_(m.weight))

        self.ffn.apply(init_weights)
        self.epoch = 0

    def _target_func(self, t_eval):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_eval)-self.config["PnL_evaluation_window_length"]+1, np.searchsorted(self.time_axis, t_eval)+1, 1)
        result = self.factor.residual_return_batch(self.time_axis[t_idx[0]], t_eval)
        Phi = [torch.from_numpy(j).to(torch.float32).to(self.device) for j in result["Phi"]]
        result = self.trading_signal.trading_signal_batch(self.time_axis[t_idx[0]], t_eval)
        time = result["time"]; equity_idx = result["equity_idx"]
        fft = [torch.from_numpy(j).to(torch.float32).to(self.device) for j in result["fft"]]
        return_hist = [torch.from_numpy(self.equity_data.return_[equity_idx[j], t_idx[j]+1]).to(torch.float32).to(self.device) for j in range(len(t_idx))]
        portfolio_return_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.ffn.forward(fft[j])
            portfolio_weights_epsilon = portfolio_weights_epsilon/torch.norm(portfolio_weights_epsilon, p=1)
            portfolio_weights_R = torch.mm(torch.transpose(Phi[j],0,1), portfolio_weights_epsilon)
            portfolio_weights_R = portfolio_weights_R/torch.norm(portfolio_weights_R, p=1)
            portfolio_return = torch.sum(torch.mul(return_hist[j], portfolio_weights_R.view(-1)))
            portfolio_return_hist.append(portfolio_return.unsqueeze(0))

        portfolio_return_hist = torch.cat(portfolio_return_hist)
        target = -(torch.mean(portfolio_return_hist) - self.config["risk_aversion_factor"]*torch.var(portfolio_return_hist))

        return target

#%%
'''
portfolio_weights_CNN_transformer_config = {"PnL_evaluation_window_length": 24,
                                                 "risk_aversion_factor": 2,
                                                 "train_t_start": datetime.datetime(1991, 7, 1),
                                                 "train_t_end": datetime.datetime(1991, 11, 1),
                                                 "valid_t_start": datetime.datetime(1991, 11, 1),
                                                 "valid_t_end": datetime.datetime(1991, 12, 15),
                                                 "epoch_max": 20,
                                                 "learning_rate": 1e-3,
                                                 "CNN_input_channels": 1,
                                                 "CNN_output_channels": 8,
                                                 "CNN_kernel_size": 2,
                                                 "CNN_drop_out_rate": 0.25,
                                                 "transformer_input_channels": 8,
                                                 "transformer_hidden_channels": 16,
                                                 "transformer_output_channels": 16,
                                                 "transformer_head_num": 4,
                                                 "transformer_drop_out_rate": 0.25,
                                                 }
'''
class portfolio_weights_CNN_transformer:
    def __init__(self, equity_data, equity_data_high_freq, factor, trading_signal_OU_process, portfolio_weights_CNN_transformer_config):
        self.equity_data = equity_data; self.equity_data_high_freq = equity_data_high_freq; self.factor = factor; self.trading_signal_OU_process = trading_signal_OU_process
        self.config = portfolio_weights_CNN_transformer_config

        self.time_axis = self.equity_data.time_axis
        self.time_axis_daily_high_freq = self.equity_data_high_freq.time_axis_daily
        self.factor._initialize_residual_return_all()
        f = open(os.path.join(os.path.dirname(__file__), "../neural_network/CNN_transformer/{}/portfolio_weights_CNN_transformer_config.pkl".format(self.factor.config["type"])), "wb"); pickle.dump(self.config, f); f.close()
        self.device = torch.device(self._get_free_gpu() if torch.cuda.is_available() else "cpu")
        #self.device = "cpu"
        self.network_file_name_prev = None

    def portfolio_weights(self, t_start, t_end):
        match self.factor.config["type"]:
            case "name":
                return self.portfolio_weights_PCA_name(t_start, t_end)
            case "rank_hybrid_Atlas":
                return self.portfolio_weights_PCA_rank_hybrid_Atlas(t_start, t_end)
            case "rank_hybrid_Atlas_high_freq":
                return self.portfolio_weights_PCA_rank_hybrid_Atlas_high_freq(t_start, t_end)
    
    def train(self, t_start, t_end, valid_t_start, valid_t_end):
        match self.factor.config["type"]:
            case "name":
                return self.train_PCA_name(t_start, t_end, valid_t_start, valid_t_end)
            case "rank_hybrid_Atlas":
                return self.train_PCA_rank_hybrid_Atlas(t_start, t_end, valid_t_start, valid_t_end)
            case "rank_hybrid_Atlas_high_freq":
                return self.train_PCA_rank_hybrid_Atlas_high_freq(t_start, t_end, valid_t_start, valid_t_end)

    def portfolio_weights_PCA_name(self, t_start, t_end):
        self.network.eval()
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        result = self.factor.residual_return_batch(self.time_axis[t_idx[0]], self.time_axis[t_idx[-1]])
        time = result["time"]; epsilon_idx = result["epsilon_idx"]; Phi = result["Phi"]
        cumulative_epsilon = [torch.from_numpy(np.cumsum(result["epsilon"][j], axis=1)).to(torch.float32).to(self.device) for j in range(len(t_idx))]
        portfolio_weights_epsilon_hist = []; portfolio_weights_R_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.network.forward(cumulative_epsilon[j].unsqueeze(1)).cpu().detach().numpy().flatten(); portfolio_weights_epsilon /= np.linalg.norm(portfolio_weights_epsilon, 1)
            portfolio_weights_R = Phi[j].T.dot(portfolio_weights_epsilon); portfolio_weights_R /= np.linalg.norm(portfolio_weights_R, 1)
            portfolio_weights_epsilon_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_epsilon_all[epsilon_idx[j]] = portfolio_weights_epsilon
            portfolio_weights_R_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_all[epsilon_idx[j]] = portfolio_weights_R
            portfolio_weights_epsilon_hist.append(portfolio_weights_epsilon_all)
            portfolio_weights_R_hist.append(portfolio_weights_R_all)
        return {"time": [self.time_axis[j] for j in t_idx], "epsilon_idx": epsilon_idx, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, 
                "equity_idx": epsilon_idx, "portfolio_weights_R": portfolio_weights_R_hist}
    
    def portfolio_weights_PCA_rank_hybrid_Atlas(self, t_start, t_end):
        self.network.eval()
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        result = self.factor.residual_return_batch(self.time_axis[t_idx[0]], self.time_axis[t_idx[-1]])
        time = result["time"]; epsilon_idx = result["epsilon_idx"]; Phi = result["Phi"]
        cumulative_epsilon = [torch.from_numpy(np.cumsum(result["epsilon"][j], axis=1)).to(torch.float32).to(self.device) for j in range(len(t_idx))]
        portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.network.forward(cumulative_epsilon[j].unsqueeze(1)).cpu().detach().numpy().flatten(); portfolio_weights_epsilon /= np.linalg.norm(portfolio_weights_epsilon, 1)
            portfolio_weights_R_rank = Phi[j].T.dot(portfolio_weights_epsilon); portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, 1)
            portfolio_weights_epsilon_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_epsilon_all[epsilon_idx[j]] = portfolio_weights_epsilon
            portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx[j]] = portfolio_weights_R_rank
            portfolio_weights_epsilon_hist.append(portfolio_weights_epsilon_all)
            portfolio_weights_R_rank_hist.append(portfolio_weights_R_rank_all)
        return {"time": [self.time_axis[j] for j in t_idx], "epsilon_idx": epsilon_idx, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist}

    def portfolio_weights_PCA_rank_hybrid_Atlas_high_freq(self, t_start, t_end):
        self.network.eval()
        t_idx = np.arange(np.searchsorted(self.time_axis_daily_high_freq, t_start), np.searchsorted(self.time_axis_daily_high_freq, t_end)+1, 1)
        result = self.factor.residual_return_batch(self.time_axis_daily_high_freq[t_idx[0]], self.time_axis_daily_high_freq[t_idx[-1]])
        time = result["time"]; epsilon_idx = result["epsilon_idx"]; Phi = result["Phi"]
        cumulative_epsilon = [torch.from_numpy(np.cumsum(result["epsilon"][j], axis=1)).to(torch.float32).to(self.device) for j in range(len(t_idx))]
        portfolio_weights_epsilon_hist = []; portfolio_weights_R_rank_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.network.forward(cumulative_epsilon[j].unsqueeze(1)).cpu().detach().numpy().flatten(); portfolio_weights_epsilon /= np.linalg.norm(portfolio_weights_epsilon, 1)
            portfolio_weights_R_rank = Phi[j].T.dot(portfolio_weights_epsilon); portfolio_weights_R_rank /= np.linalg.norm(portfolio_weights_R_rank, 1)
            portfolio_weights_epsilon_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_epsilon_all[epsilon_idx[j]] = portfolio_weights_epsilon
            portfolio_weights_R_rank_all = np.zeros(len(self.equity_data.equity_idx_list)); portfolio_weights_R_rank_all[epsilon_idx[j]] = portfolio_weights_R_rank
            portfolio_weights_epsilon_hist.append(portfolio_weights_epsilon_all)
            portfolio_weights_R_rank_hist.append(portfolio_weights_R_rank_all)
        return {"time": [self.time_axis_daily_high_freq[j] for j in t_idx], "epsilon_idx": epsilon_idx, "portfolio_weights_epsilon": portfolio_weights_epsilon_hist, "portfolio_weights_R_rank": portfolio_weights_R_rank_hist}

    def train_PCA_name(self, train_t_start=None, train_t_end=None, valid_t_start=None, valid_t_end=None):
        if train_t_start == None:
            train_t_start = self.config["train_t_start"]; train_t_end = self.config["train_t_end"]; valid_t_start = self.config["valid_t_start"]; valid_t_end = self.config["valid_t_end"]
        t_idx = np.arange(np.searchsorted(self.time_axis, train_t_start), np.searchsorted(self.time_axis, train_t_end)+1, 1)
        print("Start training neural network from %s to %s." % (datetime.datetime.strftime(self.time_axis[t_idx[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx[-1]], "%Y-%m-%d")))
        self.network = neural_network.CNN_transformer(self.config["CNN_input_channels"], self.config["CNN_output_channels"], self.config["CNN_kernel_size"], self.config["CNN_drop_out_rate"],
                                                          self.config["transformer_input_channels"], self.config["transformer_hidden_channels"], self.config["transformer_output_channels"], self.config["transformer_head_num"], self.config["transformer_drop_out_rate"])
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])

        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/CNN_transformer/", self.factor.config["type"], "/neural_network_CNN_transformer_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
        if os.path.exists(os.path.join(os.path.dirname(__file__), network_file_name)):
            checkpoint = torch.load(network_file_name, map_location=self.device)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            self.network.eval()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            target_in_sample_hist = checkpoint["in-sample target"]
            target_out_of_sample_hist = checkpoint["out-of-sample target"]
        else:
            if self.config["is_initialize_from_pretrain"]:
                if self.network_file_name_prev:
                    checkpoint = torch.load(self.network_file_name_prev, map_location=self.device)
                else:
                    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/CNN_transformer/", self.factor.config["type"], "/neural_network_CNN_transformer_initial_parameter.pt"])), map_location=self.device)
                self.network.load_state_dict(checkpoint["model_state_dict"])
                self.network.eval()
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = 0
            target_in_sample_hist = []; target_out_of_sample_hist = []

        self.train_data_cache = {}
        for epoch in range(self.config["epoch_max"]-self.epoch):
            self.network.train()
            target_epoch_hist = []
            temporal_batch_num = self.config["temporal_batch_num"]; temporal_batch_size = int(np.ceil(len(t_idx)/temporal_batch_num))
            temporal_batch_pt = 0
            for count in tqdm.tqdm(range(len(t_idx))):
                if count % temporal_batch_size == 0:
                    t_idx_sub = t_idx[(temporal_batch_pt*temporal_batch_size):min(((temporal_batch_pt+1)*temporal_batch_size), len(t_idx))]
                    temporal_batch_pt = (temporal_batch_pt+1)%temporal_batch_num
                    tqdm.tqdm.write("Training sample from {} to {}.".format(datetime.datetime.strftime(self.time_axis[t_idx_sub[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx_sub[-1]], "%Y-%m-%d")))
                target = self._target_func_PCA_name_R_space(self.time_axis[np.random.choice(t_idx_sub)])
                self.optimizer.zero_grad()
                target.backward()
                self.optimizer.step()
                target_epoch_hist.append(-float(target.cpu().detach().numpy()))

            #result = self.portfolio_weights(train_t_start, train_t_end)
            #time = result["time"]; portfolio_weights_R = result["portfolio_weights_R"]
            #return_hist = utils.evaluate_PnL_vanilla(self.equity_data, time, portfolio_weights_R)
            #target_in_sample_v2 = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))
            target_in_sample = np.mean(target_epoch_hist)

            result = self.portfolio_weights(valid_t_start, valid_t_end)
            time = result["time"]; epsilon_idx = result["epsilon_idx"]; portfolio_weights_R = result["portfolio_weights_R"]
            result = utils.evaluate_PnL_name_space(self.equity_data, time, epsilon_idx, portfolio_weights_R, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, is_vanilla=True)
            return_hist = result["return_hist"][1:]
            target_out_of_sample = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))

            target_in_sample_hist.append(target_in_sample); target_out_of_sample_hist.append(target_out_of_sample)
            print("Epoch: %d, in-sample target: %f, out-of-sample target: %f" % (self.epoch+epoch+1, target_in_sample, target_out_of_sample))

            torch.save({"epoch": self.epoch+epoch+1, "model_state_dict": self.network.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                        'in-sample target': copy.deepcopy(target_in_sample_hist), 'out-of-sample target': target_out_of_sample_hist},
                        network_file_name)
        self.network_file_name_prev = copy.deepcopy(network_file_name)
        del self.train_data_cache
        print("Training neural network complete.")

    def _target_func_PCA_name_R_space(self, t_eval):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_eval)-self.config["PnL_evaluation_window_length"]+1, np.searchsorted(self.time_axis, t_eval)+1, 1)
        Phi = []; cumulative_epsilon = []; return_hist = []
        for j in range(len(t_idx)):
            if not (self.time_axis[t_idx[j]] in self.train_data_cache):
                result = self.factor.residual_return(self.time_axis[t_idx[j]])
                Phi_temp = torch.from_numpy(result["Phi"]).to(torch.float32).to(self.device)
                cumulative_epsilon_temp = torch.from_numpy(np.cumsum(result["epsilon"], axis=1)).to(torch.float32).to(self.device)
                return_hist_temp = torch.from_numpy(self.equity_data.return_[result["epsilon_idx"], t_idx[j]+1]).to(torch.float32).to(self.device)
                self.train_data_cache[self.time_axis[t_idx[j]]] = {"Phi": Phi_temp, "cumulative_epsilon": cumulative_epsilon_temp, "return_hist": return_hist_temp}
            Phi.append(self.train_data_cache[self.time_axis[t_idx[j]]]["Phi"])
            cumulative_epsilon.append(self.train_data_cache[self.time_axis[t_idx[j]]]["cumulative_epsilon"])
            return_hist.append(self.train_data_cache[self.time_axis[t_idx[j]]]["return_hist"])

        portfolio_return_hist = []; transaction_cost_aversion_hist = []; dollar_neutrality_aversion_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.network.forward(cumulative_epsilon[j].unsqueeze(1))
            portfolio_weights_R = torch.mm(Phi[j].T, portfolio_weights_epsilon)
            portfolio_weights_R = portfolio_weights_R/torch.norm(portfolio_weights_R, p=1)
            portfolio_return = torch.sum(torch.mul(return_hist[j], portfolio_weights_R.view(-1)))
            portfolio_return_hist.append(portfolio_return.unsqueeze(0))
            if self.config["transaction_cost_aversion_factor"]!=None:
                if j == 0:
                    portfolio_weights_R_prev = copy.deepcopy(portfolio_weights_R)
                else:
                    transaction_cost_aversion = torch.norm(portfolio_weights_R.view(-1)-portfolio_weights_R_prev.view(-1), p=1)
                    transaction_cost_aversion = transaction_cost_aversion/torch.norm(transaction_cost_aversion, p=1)
                    portfolio_weights_R_prev = copy.deepcopy(portfolio_weights_R)
                    transaction_cost_aversion_hist.append(transaction_cost_aversion.unsqueeze(0))
            if self.config["dollar_neutrality_aversion_factor"]!=None:
                dollar_neutrality_aversion = torch.abs(torch.sum(portfolio_weights_R.view(-1)))
                dollar_neutrality_aversion_hist.append(dollar_neutrality_aversion.unsqueeze(0))

        portfolio_return_hist = torch.cat(portfolio_return_hist)

        target = torch.mean(portfolio_return_hist) - self.config["risk_aversion_factor"]*torch.var(portfolio_return_hist)
        if self.config["transaction_cost_aversion_factor"]!=None:
            transaction_cost_aversion_hist = torch.cat(transaction_cost_aversion_hist)
            target = target - self.config["transaction_cost_aversion_factor"]*torch.mean(transaction_cost_aversion_hist)
        if self.config["dollar_neutrality_aversion_factor"]!=None:
            dollar_neutrality_aversion_hist = torch.cat(dollar_neutrality_aversion_hist)
            target = target - self.config["dollar_neutrality_aversion_factor"]*torch.mean(dollar_neutrality_aversion_hist)
        return -target

    def train_PCA_rank_hybrid_Atlas(self, train_t_start=None, train_t_end=None, valid_t_start=None, valid_t_end=None):
        if train_t_start == None:
            train_t_start = self.config["train_t_start"]; train_t_end = self.config["train_t_end"]; valid_t_start = self.config["valid_t_start"]; valid_t_end = self.config["valid_t_end"]
        t_idx = np.arange(np.searchsorted(self.time_axis, train_t_start), np.searchsorted(self.time_axis, train_t_end)+1, 1)
        print("Start training neural network from %s to %s." % (datetime.datetime.strftime(self.time_axis[t_idx[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx[-1]], "%Y-%m-%d")))
        self.network = neural_network.CNN_transformer(self.config["CNN_input_channels"], self.config["CNN_output_channels"], self.config["CNN_kernel_size"], self.config["CNN_drop_out_rate"],
                                                          self.config["transformer_input_channels"], self.config["transformer_hidden_channels"], self.config["transformer_output_channels"], self.config["transformer_head_num"], self.config["transformer_drop_out_rate"])
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])

        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/CNN_transformer/", self.factor.config["type"], "/neural_network_CNN_transformer_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
        if os.path.exists(os.path.join(os.path.dirname(__file__), network_file_name)):
            checkpoint = torch.load(network_file_name, map_location=self.device)
            self.network.load_state_dict(checkpoint["model_state_dict"])
            self.network.eval()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            target_in_sample_hist = checkpoint["in-sample target"]
            target_out_of_sample_hist = checkpoint["out-of-sample target"]
        else:
            if self.config["is_initialize_from_pretrain"]:
                if self.network_file_name_prev:
                    checkpoint = torch.load(self.network_file_name_prev, map_location=self.device)
                else:
                    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/CNN_transformer/", self.factor.config["type"], "/neural_network_CNN_transformer_initial_parameter.pt"])), map_location=self.device)
                self.network.load_state_dict(checkpoint["model_state_dict"])
                self.network.eval()
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = 0
            target_in_sample_hist = []; target_out_of_sample_hist = []
        
        self.train_data_cache = {}
        for epoch in range(self.config["epoch_max"]-self.epoch):
            self.network.train()
            target_epoch_hist = []
            temporal_batch_num = self.config["temporal_batch_num"]; temporal_batch_size = int(np.ceil(len(t_idx)/temporal_batch_num))
            temporal_batch_pt = 0
            for count in tqdm.tqdm(range(len(t_idx))):
                if count % temporal_batch_size == 0:
                    t_idx_sub = t_idx[(temporal_batch_pt*temporal_batch_size):min(((temporal_batch_pt+1)*temporal_batch_size), len(t_idx))]
                    temporal_batch_pt = (temporal_batch_pt+1)%temporal_batch_num
                    tqdm.tqdm.write("Training sample from {} to {}.".format(datetime.datetime.strftime(self.time_axis[t_idx_sub[0]], "%Y-%m-%d"), datetime.datetime.strftime(self.time_axis[t_idx_sub[-1]], "%Y-%m-%d")))
                target = self._target_func_PCA_rank_hybrid_Atlas_R_space(self.time_axis[np.random.choice(t_idx_sub)])
                self.optimizer.zero_grad()
                target.backward()
                self.optimizer.step()
                target_epoch_hist.append(-float(target.cpu().detach().numpy()))

            #result = self.portfolio_weights(train_t_start, train_t_end)
            #time = result["time"]; portfolio_weights_R = result["portfolio_weights_R"]
            #return_hist = utils.evaluate_PnL_vanilla(self.equity_data, time, portfolio_weights_R)
            #target_in_sample_v2 = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))
            target_in_sample = np.mean(target_epoch_hist)

            result = self.portfolio_weights(valid_t_start, valid_t_end)
            time = result["time"]; epsilon_idx = result["epsilon_idx"]; portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
            result = utils.evaluate_PnL_R_rank_space(self.equity_data, time, epsilon_idx, portfolio_weights_R_rank, leverage=1, transaction_cost_factor=0.0005, shorting_cost_factor=0.0001, mode="hybrid-Atlas")
            return_hist = result["return_hist"][1:]
            target_out_of_sample = float(np.mean(return_hist) - self.config["risk_aversion_factor"]*np.var(return_hist))

            target_in_sample_hist.append(target_in_sample); target_out_of_sample_hist.append(target_out_of_sample)
            print("Epoch: %d, in-sample target: %f, out-of-sample target: %f" % (self.epoch+epoch+1, target_in_sample, target_out_of_sample))

            torch.save({"epoch": self.epoch+epoch+1, "model_state_dict": self.network.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                        'in-sample target': copy.deepcopy(target_in_sample_hist), 'out-of-sample target': target_out_of_sample_hist},
                        network_file_name)
        self.network_file_name_prev = copy.deepcopy(network_file_name)
        del self.train_data_cache
        print("Training neural network complete.")

    def _target_func_PCA_rank_hybrid_Atlas_R_space(self, t_eval):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_eval)-self.config["PnL_evaluation_window_length"]+1, np.searchsorted(self.time_axis, t_eval)+1, 1)
        Phi = []; cumulative_epsilon = []; return_hist = []
        for j in range(len(t_idx)):
            if not (self.time_axis[t_idx[j]] in self.train_data_cache):
                result = self.factor.residual_return(self.time_axis[t_idx[j]])
                Phi_temp = torch.from_numpy(result["Phi"]).to(torch.float32).to(self.device)
                cumulative_epsilon_temp = torch.from_numpy(np.cumsum(result["epsilon"], axis=1)).to(torch.float32).to(self.device)
                equity_idx_now = self.equity_data.equity_idx_by_rank[result["epsilon_idx"], t_idx[j]+1].astype(int)
                equity_idx_prev = self.equity_data.equity_idx_by_rank[result["epsilon_idx"], t_idx[j]].astype(int)
                capitalization_now = self.equity_data.capitalization[equity_idx_now, t_idx[j]+1]
                capitalization_prev = self.equity_data.capitalization[equity_idx_prev, t_idx[j]]
                return_hybrid_Atlas = torch.from_numpy(capitalization_now/capitalization_prev - 1).to(torch.float32).to(self.device)
                self.train_data_cache[self.time_axis[t_idx[j]]] = {"Phi": Phi_temp, "cumulative_epsilon": cumulative_epsilon_temp, "return_hist": return_hybrid_Atlas}
            Phi.append(self.train_data_cache[self.time_axis[t_idx[j]]]["Phi"])
            cumulative_epsilon.append(self.train_data_cache[self.time_axis[t_idx[j]]]["cumulative_epsilon"])
            return_hist.append(self.train_data_cache[self.time_axis[t_idx[j]]]["return_hist"])
        
        portfolio_return_hist = []; portfolio_weights_R_hist = []; rank_variation_aversion_hist = []; dollar_neutrality_aversion_hist = []
        for j in range(len(t_idx)):
            portfolio_weights_epsilon = self.network.forward(cumulative_epsilon[j].unsqueeze(1))
            portfolio_weights_R = torch.mm(Phi[j].T, portfolio_weights_epsilon)
            portfolio_weights_R = portfolio_weights_R/torch.norm(portfolio_weights_R, p=1)
            portfolio_return = torch.sum(torch.mul(return_hist[j], portfolio_weights_R.view(-1)))
            portfolio_return_hist.append(portfolio_return.unsqueeze(0))
            if self.config["transaction_cost_aversion_factor"]!=None:
                portfolio_weights_R_hist.append(portfolio_weights_R)
            if self.config["rank_variation_aversion_factor"]!=None:
                N = portfolio_weights_R.shape[0]
                rank_variation_aversion = torch.norm(2*portfolio_weights_R[1:(N-1), :]-portfolio_weights_R[0:(N-2), :]-portfolio_weights_R[2:N], p=1)/(N-2)
                rank_variation_aversion_hist.append(rank_variation_aversion.unsqueeze(0))
            if self.config["dollar_neutrality_aversion_factor"]!=None:
                dollar_neutrality_aversion = torch.abs(torch.sum(portfolio_weights_R.view(-1)))
                dollar_neutrality_aversion_hist.append(dollar_neutrality_aversion.unsqueeze(0))

        portfolio_return_hist = torch.cat(portfolio_return_hist)
        target = torch.mean(portfolio_return_hist) - self.config["risk_aversion_factor"]*torch.var(portfolio_return_hist)
        if self.config["transaction_cost_aversion_factor"]!=None:
            transaction_cost_aversion_hist = []
            for j in np.arange(1, len(portfolio_weights_R_hist), 1):
                transaction_cost_aversion = torch.norm(portfolio_weights_R_hist[j].view(-1)-portfolio_weights_R_hist[j-1].view(-1), p=1)
                transaction_cost_aversion_hist.append(transaction_cost_aversion.unsqueeze(0))
            transaction_cost_aversion_hist = torch.cat(transaction_cost_aversion_hist)
            target = target - self.config["transaction_cost_aversion_factor"]*torch.mean(transaction_cost_aversion_hist)
        if self.config["rank_variation_aversion_factor"]!=None:
            rank_variation_aversion_hist = torch.cat(rank_variation_aversion_hist)
            target = target - self.config["rank_variation_aversion_factor"]*torch.mean(rank_variation_aversion_hist)
        if self.config["dollar_neutrality_aversion_factor"]!=None:
            dollar_neutrality_aversion_hist = torch.cat(dollar_neutrality_aversion_hist)
            target = target - self.config["dollar_neutrality_aversion_factor"]*torch.mean(dollar_neutrality_aversion_hist)
        return -target

    def train_PCA_rank_hybrid_Atlas_high_freq(self, train_t_start=None, train_t_end=None, valid_t_start=None, valid_t_end=None):
        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../neural_network/CNN_transformer/rank_hybrid_Atlas/neural_network_CNN_transformer_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
        if os.path.exists(os.path.join(os.path.dirname(__file__), network_file_name)):
            checkpoint = torch.load(network_file_name, map_location=self.device)
            epoch = checkpoint["epoch"]
            if epoch < self.config["epoch_max"]-1:
                raise ValueError("Training neural network for PCA-rank-hybrid-Atlas has not completed yet. Please first complete the training using PCA_hybrid_Atlas before evaluating portfolio weights for PCA_hybrid_Atlas_high_freq.")
            self.factor.config["type"] = "rank_hybrid_Atlas" # temporarily change the factor type to rank_hybrid_Atlas to load neural network in the rank_hybrid_Atlas folder and train the neural network based on PCA factors by PCA-rank-hybrid-Atlas
            self.train_PCA_rank_hybrid_Atlas(train_t_start, train_t_end, valid_t_start, valid_t_end)
            self.factor.config["type"] = "rank_hybrid_Atlas_high_freq" # change the factor type back to rank_hybrid_Atlas_high_freq
        else:
            raise ValueError("Training neural network for PCA-rank-hybrid-Atlas has not completed yet. Please first complete the training using PCA_hybrid_Atlas before evaluating portfolio weights for PCA_hybrid_Atlas_high_freq.")

    def _get_free_gpu(self):
        pynvml.nvmlInit()
        def get_memory_free_MiB(gpu_index):
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.free // 1024 ** 2

        device_status = []
        #print(f"NVIDIA Driver version - {pynvml.nvmlSystemGetDriverVersion()}")
        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            device_status.append(get_memory_free_MiB(i))
            #print(f"Device {i} {pynvml.nvmlDeviceGetName(handle)} - free memory: {get_memory_free_MiB(i)} MiB")
        cuda = "cuda:{}".format(np.argmax(device_status))
        print("Use GPU: ", cuda, " with free memory: ", np.max(device_status), " MiB")
        return cuda
        

