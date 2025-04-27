#%%
import os, sys, copy, h5py, datetime, pickle, tqdm, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import data.data as data

#%% PCA factor
'''
PCA_factor_config = {"factor_evaluation_window_length": 60, #252
                "loading_evaluation_window_length": 24, #60
                "residual_return_evaluation_window_length": 24, #30
                "rank_min": 0,
                "rank_max": 499,
                "factor_number": 3,
                "type": "name", 
                "max_cache_len": 1000,
                "quick_test": True}
'''

class PCA_factor:
    def __init__(self, equity_data, equity_data_high_freq, PCA_factor_config):
        self.equity_data = equity_data; self.equity_data_high_freq = equity_data_high_freq
        self.config = PCA_factor_config
        self.time_axis = self.equity_data.time_axis
        self.time_axis_high_freq_daily = self.equity_data_high_freq.time_axis_daily
        self.rank_min = self.config["rank_min"]; self.rank_max = self.config["rank_max"]
        self.factor_number = self.config["factor_number"] if self.config["type"]=="name" else 1
        match self.config["type"]:
            case "name":
                self.save_file_name = os.path.join(os.path.dirname(__file__), "PCA_factor_name/PCA_factor_name_timestr.npz")
                self._initialize_PCA_factor_name()
            case "rank_hybrid_Atlas":
                self.save_file_name = os.path.join(os.path.dirname(__file__), "PCA_factor_rank_hybrid_Atlas/PCA_factor_rank_hybrid_Atlas_timestr.npz")
                self._initialize_PCA_factor_rank_hybrid_Atlas()
            case "rank_hybrid_Atlas_high_freq":
                self.save_file_name = os.path.join(os.path.dirname(__file__), "PCA_factor_rank_hybrid_Atlas_high_freq/PCA_factor_rank_hybrid_Atlas_high_freq_timestr.npz")
                self._initialize_PCA_factor_rank_hybrid_Atlas_high_freq()
            case "rank_permutation":
                self.save_file_name = os.path.join(os.path.dirname(__file__), "PCA_factor_rank_permutation/PCA_factor_rank_permutation_timestr.npz")
                self._intialized_PCA_factor_rank_permutation()
            case "rank_theta_transform":
                self.save_file_name = os.path.join(os.path.dirname(__file__), "PCA_factor_rank_theta_transform/PCA_factor_rank_theta_transform_timestr.npz")
                self._initialize_PCA_factor_rank_theta_transform()
        
        self.cache_len = 0
        self.cache_idle_time = np.zeros(len(self.time_axis)); self.cache_idle_time[:] = np.nan
        self.cache = [None for _ in range(len(self.time_axis))]
        print("Initialize PCA factor {} complete.".format(self.config["type"]))
    
    def residual_return(self, t_eval):
        t_idx = np.searchsorted(self.time_axis, t_eval)
        self.cache_idle_time += 1
        if self.cache[t_idx] is None:
            if self.cache_len >= self.config["max_cache_len"]:
                idx2remove = np.nanargmax(self.cache_idle_time)
                self.cache[idx2remove] = None; self.cache_idle_time[idx2remove] = np.nan
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "epsilon_idx": result["epsilon_idx"], "epsilon": result["epsilon"], "Phi": result["Phi"]}
                self.cache_idle_time[t_idx] = 0
            else:
                file_name = self.save_file_name.replace("timestr", datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d"))
                result = np.load(file_name)
                self.cache[t_idx] = {"time": self.time_axis[t_idx], "epsilon_idx": result["epsilon_idx"], "epsilon": result["epsilon"], "Phi": result["Phi"]}
                self.cache_len += 1; self.cache_idle_time[t_idx] = 0
        return self.cache[t_idx]
    
    def residual_return_batch(self, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        time_hist = []; epsilon_idx_hist = []; Phi_hist = []; epsilon_hist = []
        for j in t_idx:
            result = copy.deepcopy(self.residual_return(self.time_axis[j]))
            time_hist.append(result["time"]); epsilon_idx_hist.append(result["epsilon_idx"])
            epsilon_hist.append(result["epsilon"]); Phi_hist.append(result["Phi"])
        return {"time": time_hist, "epsilon_idx": epsilon_idx_hist, "epsilon": epsilon_hist, "Phi": Phi_hist}
    
    def _initialize_PCA_factor_name(self):
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing PCA factor in name space.")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                if t_idx < self.config["residual_return_evaluation_window_length"]+self.config["factor_evaluation_window_length"]-2:
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]),Phi=np.array([np.nan]), epsilon=np.array([np.nan]), eigenvalues=np.array([np.nan]))
                    t_idx += 1; continue
                print(self.time_axis[t_idx])
                equity_idx_valid = [k for k in self.equity_data.equity_idx_list if (self.equity_data.rank[k,t_idx]>=self.rank_min and self.equity_data.rank[k,t_idx]<=self.rank_max)]
                equity_idx_valid = [k for k in equity_idx_valid if not any(np.isnan(self.equity_data.return_[k,(t_idx-self.config["residual_return_evaluation_window_length"]-self.config["factor_evaluation_window_length"]+2):(t_idx+2)]))]
                epsilon_ar = np.zeros((len(equity_idx_valid), self.config["residual_return_evaluation_window_length"])); epsilon_ar[:] = np.nan
                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, t_idx+1, 1):
                    R = copy.deepcopy(self.equity_data.return_[equity_idx_valid, :][:, (t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]) - self.equity_data.risk_free_rate[(t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]
                    U, S, V_T = np.linalg.svd(R, full_matrices=True)
                    F = np.diag(S[0:self.factor_number]).dot(V_T[0:self.factor_number, :])
                    omega = np.linalg.lstsq(R.T, F.T, rcond=None)[0].T
                    F = copy.deepcopy(F[:, (-self.config["loading_evaluation_window_length"]):])
                    R = copy.deepcopy(R[:, (-self.config["loading_evaluation_window_length"]):])
                    beta = np.linalg.lstsq(F.T, R.T, rcond=None)[0].T
                    Phi = np.identity(R.shape[0]) - beta.dot(omega); epsilon = Phi.dot(R)
                    epsilon_ar[:, t_epsilon_idx-(t_idx-self.config["residual_return_evaluation_window_length"]+1)] = copy.deepcopy(epsilon[:,-1])

                np.savez_compressed(save_file_name, epsilon_idx=equity_idx_valid, Phi=Phi, epsilon=epsilon_ar, eigenvalues=np.power(S, 2))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

    def _initialize_PCA_factor_rank_hybrid_Atlas(self):
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing PCA factor in rank space (hybrid Atlas).")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                if t_idx < self.config["residual_return_evaluation_window_length"]+self.config["factor_evaluation_window_length"]-1:
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), Phi=np.array([np.nan]), epsilon=np.array([np.nan]), eigenvalues=np.array([np.nan]))
                    t_idx += 1; continue
                print(self.time_axis[t_idx])

                rank_idx = np.arange(self.rank_min, self.rank_max+1, 1)
                rank_idx = [j for j in rank_idx if not any(np.isnan(self.equity_data.equity_idx_by_rank[j, (t_idx-self.config["residual_return_evaluation_window_length"]-self.config["factor_evaluation_window_length"]+1):(t_idx+2)]))]
                epsilon_rank_ar = np.zeros((len(rank_idx), self.config["residual_return_evaluation_window_length"])); epsilon_rank_ar[:] = np.nan
                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, t_idx+1, 1):
                    result = self.equity_data.return_by_rank_func(rank_idx, self.time_axis[t_epsilon_idx-self.config["factor_evaluation_window_length"]+1], self.time_axis[t_epsilon_idx], mode="hybrid-Atlas")
                    R = result["return_by_rank"] - self.equity_data.risk_free_rate[(t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]
                    U, S, V_T = np.linalg.svd(R, full_matrices=True)
                    F = np.diag(S[0:self.factor_number]).dot(V_T[0:self.factor_number, :]).reshape((self.factor_number, -1))
                    omega = np.linalg.lstsq(R.T, F.T, rcond=None)[0].T
                    F = copy.deepcopy(F[:, (-self.config["loading_evaluation_window_length"]):])
                    R = copy.deepcopy(R[:, (-self.config["loading_evaluation_window_length"]):])
                    beta = np.linalg.lstsq(F.T, R.T, rcond=None)[0].T
                    Phi = np.identity(R.shape[0]) - beta.dot(omega); epsilon = Phi.dot(R)
                    epsilon_rank_ar[:, t_epsilon_idx-(t_idx-self.config["residual_return_evaluation_window_length"]+1)] = copy.deepcopy(epsilon[:,-1])

                np.savez_compressed(save_file_name, epsilon_idx=rank_idx, Phi=Phi, epsilon=epsilon_rank_ar, eigenvalues=np.power(S, 2))
                t_idx += 1

            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)
        else:
            pass

    def _initialize_PCA_factor_rank_hybrid_Atlas_high_freq(self):
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing PCA factor in rank space (hybrid Atlas, high frequency).")
            t_idx = 0
            while t_idx < len(self.time_axis_high_freq_daily):
                time_str = datetime.datetime.strftime(self.time_axis_high_freq_daily[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                if t_idx < self.config["residual_return_evaluation_window_length"]+self.config["factor_evaluation_window_length"]-1:
                    np.savez_compressed(save_file_name, epsilon_idx=np.array([np.nan]), Phi=np.array([np.nan]), epsilon=np.array([np.nan]), eigenvalues=np.array([np.nan]))
                    t_idx += 1; continue
                print(self.time_axis_high_freq_daily[t_idx])

                rank_idx = np.arange(self.rank_min, self.rank_max+1, 1)
                rank_idx = [j for j in rank_idx if not any(np.isnan(self.equity_data_high_freq.equity_idx_by_rank_daily[j, (t_idx-self.config["residual_return_evaluation_window_length"]-self.config["factor_evaluation_window_length"]+1):(t_idx+2)]))]
                #if t_idx < len(self.time_axis_high_freq_daily)-1:
                #    rank_idx = [j for j in rank_idx if not np.isnan(self.equity_data_high_freq.capitalization_daily[self.equity_data_high_freq.equity_idx_by_rank_daily[j, t_idx].astype(int), t_idx+1])]
                epsilon_rank_ar = np.zeros((len(rank_idx), self.config["residual_return_evaluation_window_length"])); epsilon_rank_ar[:] = np.nan
                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, t_idx+1, 1):
                    result = self.equity_data_high_freq.daily_return_by_rank_func(rank_idx, self.time_axis_high_freq_daily[t_epsilon_idx-self.config["factor_evaluation_window_length"]+1], self.time_axis_high_freq_daily[t_epsilon_idx], mode="hybrid-Atlas")
                    r_f_idx = np.arange(np.searchsorted(self.time_axis, self.time_axis_high_freq_daily[t_epsilon_idx-self.config["factor_evaluation_window_length"]+1]), np.searchsorted(self.time_axis, self.time_axis_high_freq_daily[t_epsilon_idx])+1, 1)
                    R = result["return_by_rank"] - self.equity_data.risk_free_rate[r_f_idx]
                    U, S, V_T = np.linalg.svd(R, full_matrices=True)
                    F = np.diag(S[0:self.factor_number]).dot(V_T[0:self.factor_number, :]).reshape((self.factor_number, -1))
                    omega = np.linalg.lstsq(R.T, F.T, rcond=None)[0].T
                    F = copy.deepcopy(F[:, (-self.config["loading_evaluation_window_length"]):])
                    R = copy.deepcopy(R[:, (-self.config["loading_evaluation_window_length"]):])
                    beta = np.linalg.lstsq(F.T, R.T, rcond=None)[0].T
                    Phi = np.identity(R.shape[0]) - beta.dot(omega); epsilon = Phi.dot(R)
                    epsilon_rank_ar[:, t_epsilon_idx-(t_idx-self.config["residual_return_evaluation_window_length"]+1)] = copy.deepcopy(epsilon[:,-1])

                np.savez_compressed(save_file_name, epsilon_idx=rank_idx, Phi=Phi, epsilon=epsilon_rank_ar, eigenvalues=np.power(S, 2))
                t_idx += 1

            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)
        else:
            pass

    def _intialized_PCA_factor_rank_permutation(self):
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing PCA factor in rank space (permutation).")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                if t_idx < self.config["residual_return_evaluation_window_length"]+self.config["factor_evaluation_window_length"]-1:
                    np.savez_compressed(save_file_name, rank_idx=np.array([np.nan]), epsilon_idx=np.array([np.nan]), Phi=np.array([np.nan]), epsilon=np.array([np.nan]), eigenvalues=np.array([np.nan]))
                    t_idx += 1; continue
                print(self.time_axis[t_idx])

                rank_idx = np.arange(self.rank_min, self.rank_max+1, 1)
                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, min(t_idx+2, len(self.time_axis)), 1):
                    result = self.equity_data.return_by_rank_func(rank_idx, self.time_axis[t_epsilon_idx-self.config["factor_evaluation_window_length"]+1], self.time_axis[t_epsilon_idx], mode="permutation")
                    rank_idx = np.intersect1d(rank_idx, result["rank_idx"])

                epsilon_ar = np.zeros((len(rank_idx), self.config["residual_return_evaluation_window_length"])); epsilon_ar[:] = np.nan
                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, t_idx+1, 1):
                    result = self.equity_data.return_by_rank_func(rank_idx, self.time_axis[t_epsilon_idx-self.config["factor_evaluation_window_length"]+1], self.time_axis[t_epsilon_idx], mode="permutation")
                    R = result["return_by_rank"] - self.equity_data.risk_free_rate[(t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]
                    U, S, V_T = np.linalg.svd(R, full_matrices=True)
                    F = np.diag(S[0:self.factor_number]).dot(V_T[0:self.factor_number, :]).reshape((self.factor_number, -1))
                    omega = np.linalg.lstsq(R.T, F.T, rcond=None)[0].T
                    F = copy.deepcopy(F[:, (-self.config["loading_evaluation_window_length"]):])
                    R = copy.deepcopy(R[:, (-self.config["loading_evaluation_window_length"]):])
                    beta = np.linalg.lstsq(F.T, R.T, rcond=None)[0].T
                    Phi = np.identity(R.shape[0]) - beta.dot(omega); epsilon = Phi.dot(R)
                    epsilon_ar[:, t_epsilon_idx-(t_idx-self.config["residual_return_evaluation_window_length"]+1)] = copy.deepcopy(epsilon[:,-1])

                np.savez_compressed(save_file_name, epsilon_idx=rank_idx, Phi=Phi, epsilon=epsilon_ar, eigenvalues=np.power(S, 2))
                t_idx += 1
            
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)

    def _initialize_PCA_factor_rank_theta_transform(self):
        if not os.path.exists(self.save_file_name.replace("timestr", "complete")):
            print("Initializing PCA factor in rank space (theta transform).")
            t_idx = 0
            while t_idx < len(self.time_axis):
                time_str = datetime.datetime.strftime(self.time_axis[t_idx], "%Y%m%d")
                save_file_name = copy.deepcopy(self.save_file_name.replace("timestr", time_str))
                if os.path.exists(save_file_name):
                    t_idx += 1; continue
                if t_idx < self.config["residual_return_evaluation_window_length"]+self.config["factor_evaluation_window_length"]-2:
                    np.savez_compressed(save_file_name, equity_idx=np.array([np.nan]), epsilon_idx=np.array([np.nan]), Phi=np.array([np.nan]), epsilon=np.array([np.nan]), eigenvalues=np.array([np.nan]), leakage=np.array([np.nan]))
                    t_idx += 1; continue
                print(self.time_axis[t_idx])

                rank_idx = np.arange(self.rank_min, self.rank_max+1, 1)
                result = self.equity_data.occupation_rate_by_rank(rank_idx, self.time_axis[t_idx-self.config["residual_return_evaluation_window_length"]-self.config["factor_evaluation_window_length"]+2], self.time_axis[t_idx], mode="rank", is_parallel=True)
                equity_idx = result["equity_idx"]; rank_idx = result["rank_idx"]; occupation_time = result["occupation_time"]
                idx_valid_equity_axis = [j for j in range(len(equity_idx)) if not any(np.isnan(self.equity_data.return_[equity_idx[j], (t_idx-self.config["residual_return_evaluation_window_length"]-self.config["factor_evaluation_window_length"]+2):(t_idx+2)]))]
                equity_idx = [equity_idx[j] for j in idx_valid_equity_axis]; occupation_time = occupation_time[:, idx_valid_equity_axis]
                idx_valid_rank_axis = [j for j in range(len(rank_idx)) if np.sum(occupation_time[j, :])>0]
                rank_idx = [rank_idx[j] for j in idx_valid_rank_axis]; occupation_time = occupation_time[idx_valid_rank_axis, :]
                theta = occupation_time/np.sum(occupation_time, axis=1, keepdims=True)
                epsilon_ar = np.zeros((len(rank_idx), self.config["residual_return_evaluation_window_length"])); epsilon_ar[:] = np.nan

                for t_epsilon_idx in np.arange(t_idx-self.config["residual_return_evaluation_window_length"]+1, t_idx+1, 1):
                    R = copy.deepcopy(self.equity_data.return_[equity_idx, :][:, (t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]) - self.equity_data.risk_free_rate[(t_epsilon_idx-self.config["factor_evaluation_window_length"]+1):(t_epsilon_idx+1)]
                    theta_R = theta.dot(R)
                    U, S, V_T = np.linalg.svd(theta_R, full_matrices=True)
                    F = np.diag(S[0:self.factor_number]).dot(V_T[0:self.factor_number, :]).reshape((self.factor_number, -1))
                    omega = np.linalg.lstsq(theta_R.T, F.T, rcond=None)[0].T
                    F = copy.deepcopy(F[:, (-self.config["loading_evaluation_window_length"]):])
                    theta_R = copy.deepcopy(theta_R[:, (-self.config["loading_evaluation_window_length"]):])
                    theta_beta = np.linalg.lstsq(F.T, theta_R.T, rcond=None)[0].T
                    Phi = (np.identity(theta_R.shape[0]) - theta_beta.dot(omega)); epsilon = Phi.dot(theta_R)
                    epsilon_ar[:, t_epsilon_idx-(t_idx-self.config["residual_return_evaluation_window_length"]+1)] = copy.deepcopy(epsilon[:,-1])

                np.savez_compressed(save_file_name, epsilon_idx=rank_idx, equity_idx=equity_idx, Phi=Phi, epsilon=epsilon_ar, theta = theta, eigenvalues=np.power(S, 2))
                t_idx += 1
            np.savez_compressed(self.save_file_name.replace("timestr", "complete"), indicator=1)
        else:
            pass

    def _initialize_residual_return_all(self):
        file_name = self.save_file_name.replace("timestr", "residual_return_all")
        if not os.path.exists(file_name):
            if self.config["type"] in ["name", "rank_hybrid_Atlas", "rank_permutation", "rank_theta_transform"]:
                self.epsilon_all = np.zeros((len(self.equity_data.equity_idx_list), len(self.time_axis))); self.epsilon_all[:] = np.nan
                for t_idx in tqdm.tqdm(range(len(self.time_axis))):
                    result = self.residual_return(self.time_axis[t_idx])
                    if np.isnan(result["epsilon_idx"][0]):
                        continue
                    self.epsilon_all[result["epsilon_idx"], t_idx] = result["epsilon"][:, -1]
                np.savez(file_name, epsilon=self.epsilon_all)
            if self.config["type"] == "name_Yeo_GP":
                t_start = datetime.datetime(2000, 1, 1); t_end = datetime.datetime(2014, 12, 31)
                t_idx_list = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
                self.epsilon_all = np.zeros((len(self.equity_data.equity_idx_list), len(t_idx_list))); self.epsilon_all[:] = np.nan
                for t_idx in tqdm.tqdm(t_idx_list):
                    result = self.residual_return(self.time_axis[t_idx])
                    if np.isnan(result["epsilon_idx"][0]):
                        continue
                    self.epsilon_all[result["epsilon_idx"], t_idx-t_idx_list[0]] = result["epsilon"][:, -1]
                np.savez(file_name, epsilon=self.epsilon_all)
            if self.config["type"] == "rank_hybrid_Atlas_high_freq":
                self.epsilon_all = np.zeros((len(self.equity_data.equity_idx_list), len(self.time_axis_high_freq_daily))); self.epsilon_all[:] = np.nan
                for t_idx in tqdm.tqdm(range(len(self.time_axis_high_freq_daily))):
                    result = self.residual_return(self.time_axis_high_freq_daily[t_idx])
                    if np.isnan(result["epsilon_idx"][0]):
                        continue
                    if self.config["type"] == "name":
                        self.epsilon_all[result["epsilon_idx"], t_idx] = result["epsilon"][:, -1]
                    if self.config["type"] == "rank_hybrid_Atlas":
                        self.epsilon_all[result["epsilon_idx"], t_idx] = result["epsilon"][:, -1]
                    if self.config["type"] == "rank_hybrid_Atlas_high_freq":
                        self.epsilon_all[result["epsilon_idx"], t_idx] = result["epsilon"][:, -1]
                    if self.config["type"] == "rank_permutation":
                        self.epsilon_all[result["epsilon_idx"], t_idx] = result["epsilon"][:, -1]
                np.savez(file_name, epsilon=self.epsilon_all)

        else:
            result = np.load(file_name)
            self.epsilon_all = result["epsilon"]

    def _clear_cache(self):
        del self.cache_idle_time, self.cache
        self.cache_len = 0
        self.cache_idle_time = np.zeros(len(self.time_axis)); self.cache_idle_time[:] = np.nan
        self.cache = [None for _ in range(len(self.time_axis))]
        gc.collect()

