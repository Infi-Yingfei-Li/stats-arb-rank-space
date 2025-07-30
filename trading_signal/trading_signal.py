#%%
import os, sys, copy, h5py, datetime, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import data.data as data
import market_decomposition.market_factor_classic as factor

#%%
'''
trading_signal_OU_process_config = {"max_cache_len": 10}
'''
class trading_signal_OU_process:
    def __init__(self, equity_data, equity_data_high_freq, factor, trading_signal_OU_process_config):
        self.equity_data = equity_data; self.equity_data_high_freq = equity_data_high_freq; self.factor = factor
        self.config = trading_signal_OU_process_config
        self.time_axis = copy.deepcopy(self.equity_data.time_axis)
        self.time_axis_high_freq_daily = copy.deepcopy(self.equity_data_high_freq.time_axis_daily)
        match self.factor.config["type"]:
            case "name":
                self._initialize_trading_signal_PCA_name()
            case "rank_hybrid_Atlas":
                self._initialize_trading_signal_PCA_rank_hybrid_Atlas()
            case "rank_hybrid_Atlas_high_freq":
                self._initialize_trading_signal_PCA_rank_hybrid_Atlas_high_freq()
            case "rank_permutation":
                self._initialize_trading_signal_PCA_rank_permutation()
            case "rank_theta_transform":
                self._initialize_trading_signal_PCA_rank_theta_transform()

        self.cache_len = 0
        self.cache_idle_time = np.zeros(len(self.time_axis)); self.cache_idle_time[:] = np.nan
        self.cache = [None for _ in range(len(self.time_axis))]
        print("Initialize trading signal from OU process complete.")

    def trading_signal(self, t_eval):
        t_idx = np.searchsorted(self.time_axis, t_eval)
        self.cache_idle_time += 1
        if self.cache[t_idx] is None:
            if self.cache_len >= self.config["max_cache_len"]:
                idx2remove = np.nanargmax(self.cache_idle_time)
                self.cache[idx2remove] = None; self.cache_idle_time[idx2remove] = np.nan
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "epsilon_idx": result["epsilon_idx"], "kappa": result["kappa"], \
                                     "mu": result["mu"], "sigma": result["sigma"], "cummulative_epsilon_end": result["cummulative_epsilon_end"], "R_sq": result["R_sq"]}
                self.cache_idle_time[t_idx] = 0
            else:
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "epsilon_idx": result["epsilon_idx"], "kappa": result["kappa"], \
                                     "mu": result["mu"], "sigma": result["sigma"], "cummulative_epsilon_end": result["cummulative_epsilon_end"], "R_sq": result["R_sq"]}
                self.cache_len += 1; self.cache_idle_time[t_idx] = 0

        return self.cache[t_idx]

    def _initialize_trading_signal_PCA_name(self):
        self.save_file_name = os.path.join(os.path.dirname(__file__), "OU_process/PCA_factor_name/trading_signal_OU_process_timestr.npz")
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from OU process.")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(epsilon_idx[0]):
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), kappa=np.array([np.nan]), mu=np.array([np.nan]),\
                                         sigma=np.array([np.nan]), cummulative_epsilon_end=np.array([np.nan]), R_sq=np.array([np.nan]))
                    t_idx += 1; continue
                
                print(self.time_axis[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                kappa = []; mu = []; sigma = []; cummulative_epsilon_end = []; R_sq = []
                for j in range(len(epsilon_idx)):
                    X = copy.deepcopy(epsilon_cummulative[j, 0:(epsilon_cummulative.shape[1]-1)]).reshape((-1, 1))
                    Y = copy.deepcopy(epsilon_cummulative[j, 1:(epsilon_cummulative.shape[1])]).reshape((-1, 1))
                    reg = LinearRegression().fit(X, Y)
                    a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
                    if b > 0 and b < 1:
                        kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                        cummulative_epsilon_end.append(epsilon_cummulative[j, -1]); R_sq.append(reg.score(X, Y))
                    else:
                        kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); cummulative_epsilon_end.append(epsilon_cummulative[j,-1]); R_sq.append(np.nan)
                np.savez_compressed(save_file_name, epsilon_idx=np.array(epsilon_idx), kappa=np.array(kappa), mu=np.array(mu),\
                                      sigma=np.array(sigma), cummulative_epsilon_end=np.array(cummulative_epsilon_end), R_sq=np.array(R_sq))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

    def _initialize_trading_signal_PCA_rank_hybrid_Atlas(self):
        self.save_file_name = os.path.join(os.path.dirname(__file__), "OU_process/PCA_factor_rank_hybrid_Atlas/trading_signal_OU_process_timestr.npz")
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from OU process.")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(epsilon_idx[0]):
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), kappa=np.array([np.nan]), mu=np.array([np.nan]),\
                                         sigma=np.array([np.nan]), cummulative_epsilon_end=np.array([np.nan]), R_sq=np.array([np.nan]))
                    t_idx += 1; continue
                
                print(self.time_axis[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                kappa = []; mu = []; sigma = []; cummulative_epsilon_end = []; R_sq = []
                for j in range(len(epsilon_idx)):
                    X = copy.deepcopy(epsilon_cummulative[j, 0:(epsilon_cummulative.shape[1]-1)]).reshape((-1, 1))
                    Y = copy.deepcopy(epsilon_cummulative[j, 1:(epsilon_cummulative.shape[1])]).reshape((-1, 1))
                    reg = LinearRegression().fit(X, Y)
                    a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
                    if b > 0 and b < 1:
                        kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                        cummulative_epsilon_end.append(epsilon_cummulative[j, -1]); R_sq.append(reg.score(X, Y))
                    else:
                        kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); cummulative_epsilon_end.append(epsilon_cummulative[j,-1]); R_sq.append(np.nan)
                np.savez_compressed(save_file_name, epsilon_idx=np.array(epsilon_idx), kappa=np.array(kappa), mu=np.array(mu),\
                                      sigma=np.array(sigma), cummulative_epsilon_end=np.array(cummulative_epsilon_end), R_sq=np.array(R_sq))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)
    
    def _initialize_trading_signal_PCA_rank_hybrid_Atlas_high_freq(self):
        self.save_file_name = os.path.join(os.path.dirname(__file__), "OU_process/PCA_factor_rank_hybrid_Atlas_high_freq/trading_signal_OU_process_timestr.npz")
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from OU process")
            t_idx = 0
            while t_idx < len(self.time_axis_high_freq_daily):
                time_str = datetime.datetime.strftime(self.time_axis_high_freq_daily[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis_high_freq_daily[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(epsilon_idx[0]):
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), kappa=np.array([np.nan]), mu=np.array([np.nan]),\
                                         sigma=np.array([np.nan]), cummulative_epsilon_end=np.array([np.nan]), R_sq=np.array([np.nan]))
                    t_idx += 1; continue
                
                print(self.time_axis_high_freq_daily[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                kappa = []; mu = []; sigma = []; cummulative_epsilon_end = []; R_sq = []
                for j in range(len(epsilon_idx)):
                    X = copy.deepcopy(epsilon_cummulative[j, 0:(epsilon_cummulative.shape[1]-1)]).reshape((-1, 1))
                    Y = copy.deepcopy(epsilon_cummulative[j, 1:(epsilon_cummulative.shape[1])]).reshape((-1, 1))
                    reg = LinearRegression().fit(X, Y)
                    a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
                    if b > 0 and b < 1:
                        kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                        cummulative_epsilon_end.append(epsilon_cummulative[j, -1]); R_sq.append(reg.score(X, Y))
                    else:
                        kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); cummulative_epsilon_end.append(epsilon_cummulative[j,-1]); R_sq.append(np.nan)
                np.savez_compressed(save_file_name, epsilon_idx=np.array(epsilon_idx), kappa=np.array(kappa), mu=np.array(mu),\
                                        sigma=np.array(sigma), cummulative_epsilon_end=np.array(cummulative_epsilon_end), R_sq=np.array(R_sq))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

    def _initialize_trading_signal_PCA_rank_permutation(self):
        self.save_file_name = os.path.join(os.path.dirname(__file__), "OU_process/PCA_factor_rank_permutation/trading_signal_OU_process_timestr.npz")
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from OU process.")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(epsilon_idx[0]):
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), kappa=np.array([np.nan]), mu=np.array([np.nan]),\
                                         sigma=np.array([np.nan]), cummulative_epsilon_end=np.array([np.nan]), R_sq=np.array([np.nan]))
                    t_idx += 1; continue

                print(self.time_axis[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                kappa = []; mu = []; sigma = []; cummulative_epsilon_end = []; R_sq = []
                for j in range(len(epsilon_idx)):
                    X = copy.deepcopy(epsilon_cummulative[j, 0:(epsilon_cummulative.shape[1]-1)]).reshape((-1, 1))
                    Y = copy.deepcopy(epsilon_cummulative[j, 1:(epsilon_cummulative.shape[1])]).reshape((-1, 1))
                    reg = LinearRegression().fit(X, Y)
                    a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
                    if b > 0 and b < 1:
                        kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                        cummulative_epsilon_end.append(epsilon_cummulative[j, -1]); R_sq.append(reg.score(X, Y))
                    else:
                        kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); cummulative_epsilon_end.append(epsilon_cummulative[j,-1]); R_sq.append(np.nan)
                np.savez_compressed(save_file_name, epsilon_idx=np.array(epsilon_idx), kappa=np.array(kappa), mu=np.array(mu),\
                                      sigma=np.array(sigma), cummulative_epsilon_end=np.array(cummulative_epsilon_end), R_sq=np.array(R_sq))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

    def _initialize_trading_signal_PCA_rank_theta_transform(self):
        self.save_file_name = os.path.join(os.path.dirname(__file__), "OU_process/PCA_factor_rank_theta_transform/trading_signal_OU_process_timestr.npz")
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from OU process.")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name.replace("timestr", time_str)):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis[t_idx])
                epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(epsilon_idx[0]):
                    np.savez_compressed(save_file_name, equity_idx=np.array([np.nan]), epsilon_idx=np.array([np.nan]), kappa=np.array([np.nan]), mu=np.array([np.nan]),\
                                         sigma=np.array([np.nan]), cummulative_epsilon_end=np.array([np.nan]), R_sq=np.array([np.nan]))
                    t_idx += 1; continue
                
                print(self.time_axis[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                kappa = []; mu = []; sigma = []; cummulative_epsilon_end = []; R_sq = []
                for j in range(len(epsilon_idx)):
                    X = copy.deepcopy(epsilon_cummulative[j, 0:(epsilon_cummulative.shape[1]-1)]).reshape((-1, 1))
                    Y = copy.deepcopy(epsilon_cummulative[j, 1:(epsilon_cummulative.shape[1])]).reshape((-1, 1))
                    reg = LinearRegression().fit(X, Y)
                    a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
                    if b > 0 and b < 1:
                        kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                        cummulative_epsilon_end.append(epsilon_cummulative[j, -1]); R_sq.append(reg.score(X, Y))
                    else:
                        kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); cummulative_epsilon_end.append(epsilon_cummulative[j,-1]); R_sq.append(np.nan)
                np.savez_compressed(save_file_name, epsilon_idx=np.array(epsilon_idx), kappa=np.array(kappa), mu=np.array(mu),\
                                      sigma=np.array(sigma), cummulative_epsilon_end=np.array(cummulative_epsilon_end), R_sq=np.array(R_sq))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)


#%%
'''
trading_signal_fast_fourier_transform_config = {"max_cache_len": 10}
'''
class trading_signal_fast_fourier_transform:
    def __init__(self, equity_data, factor, trading_signal_fast_fourier_transform_config):
        self.equity_data = equity_data; self.factor = factor
        self.config = trading_signal_fast_fourier_transform_config
        self.time_axis = self.equity_data.time_axis
        if self.factor.config["quick_test"]:
            match self.factor.config["type"]:
                case "name":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "QT_fast_fourier_transform/PCA_factor_name/trading_signal_fast_fourier_transform_timestr.npz")
                case "rank_theta_transform":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "QT_fast_fourier_transform/PCA_factor_rank_theta_transform/trading_signal_fast_fourier_transform_timestr.npz")
                case "rank_hybrid_Atlas":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "QT_fast_fourier_transform/PCA_factor_rank_hybrid_Atlas/trading_signal_fast_fourier_transform_timestr.npz")
        else:
            match self.factor.config["type"]:
                case "name":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "fast_fourier_transform/PCA_factor_name/trading_signal_fast_fourier_transform_timestr.npz")
                case "rank_theta_transform":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "fast_fourier_transform/PCA_factor_rank_theta_transform/trading_signal_fast_fourier_transform_timestr.npz")
                case "rank_hybrid_Atlas":
                    self.save_file_name = os.path.join(os.path.dirname(__file__), "fast_fourier_transform/PCA_factor_rank_hybrid_Atlas/trading_signal_fast_fourier_transform_timestr.npz")

        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing trading signal from fast fourier transform.")
            t_idx = 0
            while t_idx < len(self.equity_data.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                result = self.factor.residual_return(self.time_axis[t_idx])
                equity_idx = result["equity_idx"]; epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
                if np.isnan(equity_idx[0]):
                    np.savez_compressed(save_file_name, equity_idx=np.array([np.nan]), epsilon_idx=np.array([np.nan]), fft=np.array([np.nan]))
                    t_idx += 1; continue
                
                print(self.time_axis[t_idx])
                epsilon_cummulative = np.cumsum(epsilon, axis=1)
                fft = []; T = epsilon_cummulative.shape[1]
                for j in range(len(epsilon_idx)):
                    fft_complex = np.fft.fft(epsilon_cummulative[j, :])
                    if T % 2 == 0:
                        fft_trig = [fft_complex[0]] + [(fft_complex[j]+fft_complex[T-j]) for j in np.arange(1, T/2, 1).astype(np.int16)]\
                        + [1j*(fft_complex[j]-fft_complex[T-j]) for j in np.arange(1, T/2, 1).astype(np.int16)] + [fft_complex[int(T/2)]]
                    else:
                        fft_trig = [fft_complex[0]] + [(fft_complex[j]+fft_complex[T-j]) for j in np.arange(1, (T+1)/2, 1).astype(np.int16)]\
                        + [1j*(fft_complex[j]-fft_complex[T-j]) for j in np.arange(1, (T+1)/2, 1).astype(np.int16)]
                    fft.append(fft_trig)
                fft = np.absolute(np.array(fft))/T
                np.savez_compressed(save_file_name, equity_idx=equity_idx, epsilon_idx=epsilon_idx, fft=fft)
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

        self.cache_len = 0
        self.cache_idle_time = np.zeros(len(self.time_axis)); self.cache_idle_time[:] = np.nan
        self.cache = [None for _ in range(len(self.time_axis))]
        print("Initialize trading signal from fast fourier transform complete.")

    def trading_signal(self, t_eval):
        t_idx = np.searchsorted(self.time_axis, t_eval)
        self.cache_idle_time += 1
        if self.cache[t_idx] is None:
            if self.cache_len >= self.config["max_cache_len"]:
                idx2remove = np.nanargmax(self.cache_idle_time)
                self.cache[idx2remove] = None; self.cache_idle_time[idx2remove] = np.nan
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "equity_idx": result["equity_idx"], "epsilon_idx": result["epsilon_idx"], "fft": result["fft"]}
                self.cache_idle_time[t_idx] = 0
            else:
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "equity_idx": result["equity_idx"], "epsilon_idx": result["epsilon_idx"], "fft": result["fft"]}
                self.cache_len += 1; self.cache_idle_time[t_idx] = 0

        return self.cache[t_idx]
    
    def trading_signal_batch(self, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        time_hist = []; equity_idx_hist = []; epsilon_idx_hist = []; fft_hist = []
        for j in t_idx:
            result = copy.deepcopy(self.trading_signal(self.time_axis[j]))
            time_hist.append(result["time"]); equity_idx_hist.append(result["equity_idx"]); epsilon_idx_hist.append(result["epsilon_idx"]); fft_hist.append(result["fft"])
        return {"time": time_hist, "equity_idx": equity_idx_hist, "epsilon_idx": epsilon_idx_hist, "fft": fft_hist}

