#%%
import os, sys, copy, h5py, datetime, tqdm, gc, pickle
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_dict = {"PCA_name": color_cycle[0], "PCA_rank_permutation": color_cycle[1], "PCA_rank_hybrid_Atlas": color_cycle[2], 
              "PCA_rank_hybrid_Atlas_high_freq": color_cycle[3], "PCA_rank_theta": color_cycle[4]}

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

rank_min = 0; rank_max = 499
# time interval for market decomposition
#t_eval_start = datetime.datetime(2004,1,1); t_eval_end = datetime.datetime(2015,12,31)
t_eval_start = datetime.datetime(2004,1,1); t_eval_end = datetime.datetime(2004,1,31)
# time interval for backtest
#t_backtest_start = datetime.datetime(2005,1,1); t_backtest_end = datetime.datetime(2014,12,31)
#t_backtest_start = datetime.datetime(2005,1,1); t_backtest_end = datetime.datetime(2004,12,31)

factor_num = 5
eta = 0.7
threshold_open = 1.25; threshold_close = 0.75

#%% select equities for backtest
t_idx_list = np.arange(np.searchsorted(equity_data_.time_axis, datetime.datetime(2000,1,1)), np.searchsorted(equity_data_.time_axis, datetime.datetime(2015,12,31)), 1)
equity_idx = equity_data_.equity_idx_by_rank[rank_min:rank_max, t_idx_list[0]].astype(int)
for t_idx in t_idx_list:
    equity_idx = np.union1d(equity_idx, equity_data_.equity_idx_by_rank[rank_min:rank_max, t_idx].astype(int))
valid_idx = [j for j in range(len(equity_idx)) if ~np.isnan(equity_data_.return_[equity_idx[j], t_idx_list]).any()]
equity_idx = equity_idx[valid_idx]

t_idx_list = np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end), 1)

#%% market decomposition
residual_all = np.zeros((len(equity_idx), len(t_idx_list))); residual_all[:] = np.nan
factor_update_count = 0
lookback_window = 60

def market_decomposition(t_eval):
    t_idx = np.searchsorted(equity_data_.time_axis, t_eval)
    r = equity_data_.return_[equity_idx, :][:, (t_idx-lookback_window+1):(t_idx+1)]
    if factor_update_count == 0:
        R = (r-np.mean(r, axis=1, keepdims=True))/np.std(r, axis=1, ddof=1, keepdims=True)
        U, S, V = np.linalg.svd(R, full_matrices=True)
        q = U[:, 0:factor_num].reshape((-1, factor_num))
        q = np.diag(1/np.std(r, axis=1, ddof=1)).dot(q)
        F = q.T.dot(r)
    L = np.linalg.lstsq(F.T, r.T, rcond=None)[0].T
    U = r - L.dot(F)
    residual_all[:, t_idx-t_idx_list[0]] = U[:, -1]
    return {"L": L, "F": F, "U": U}

if os.path.exists(os.path.join(os.path.dirname(__file__), "residual_all.npz")):
    result = np.load(os.path.join(os.path.dirname(__file__), "residual_all.npz"))
    residual_all = result["residual_all"]
else:
    for t_idx in tqdm.tqdm(t_idx_list):
        t_eval = equity_data_.time_axis[t_idx]
        t_idx = np.searchsorted(equity_data_.time_axis, t_eval)
        r = equity_data_.return_[equity_idx, :][:, (t_idx-lookback_window+1):(t_idx+1)]
        if factor_update_count == 0:
            R = (r-np.mean(r, axis=1, keepdims=True))/np.std(r, axis=1, ddof=1, keepdims=True)
            U, S, V = np.linalg.svd(R, full_matrices=True)
            q = U[:, 0:factor_num].reshape((-1, factor_num))
            q = np.diag(1/np.std(r, axis=1, ddof=1)).dot(q)
            F = q.T.dot(r)
        L = np.linalg.lstsq(F.T, r.T, rcond=None)[0].T
        U = r - L.dot(F)
        residual_all[:, t_idx-t_idx_list[0]] = U[:, -1]
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "residual_all.npz"), residual_all=residual_all)

#%% generate trading signal
if os.path.exists(os.path.join(os.path.dirname(__file__), "trading_signal.npz")):
    result = np.load(os.path.join(os.path.dirname(__file__), "trading_signal.npz"))
    kappa_all = result["kappa_all"]; tau_all = result["tau_all"]; mu_all = result["mu_all"]; sigma_all = result["sigma_all"]; X_end_all = result["X_end_all"]; R_sq_all = result["R_sq_all"]
else:
    kappa_all = np.zeros((len(equity_idx), len(t_idx_list))); kappa_all[:] = np.nan
    tau_all = np.zeros((len(equity_idx), len(t_idx_list))); tau_all[:] = np.nan
    mu_all = np.zeros((len(equity_idx), len(t_idx_list))); mu_all[:] = np.nan
    sigma_all = np.zeros((len(equity_idx), len(t_idx_list))); sigma_all[:] = np.nan
    X_end_all = np.zeros((len(equity_idx), len(t_idx_list))); X_end_all[:] = np.nan
    R_sq_all = np.zeros((len(equity_idx), len(t_idx_list))); R_sq_all[:] = np.nan

    for t_idx in tqdm.tqdm(t_idx_list):
        #residual = residual_all[:, (t_idx-t_idx_list[0]-lookback_window+1):(t_idx-t_idx_list[0]+1)]
        result = market_decomposition(equity_data_.time_axis[t_idx])
        residual = result["U"]
        cumulative_residual = np.cumsum(residual, axis=1)
        kappa = []; mu = []; sigma = []; X_end = []; R_sq = []
        for j in range(residual.shape[0]):
            X = copy.deepcopy(cumulative_residual[j, 0:(cumulative_residual.shape[1]-1)]).reshape((-1, 1))
            Y = copy.deepcopy(cumulative_residual[j, 1:(cumulative_residual.shape[1])]).reshape((-1, 1))
            reg = LinearRegression().fit(X, Y)
            a = reg.intercept_[0]; b = reg.coef_[0,0]; error_var = np.var(Y-reg.predict(X))
            if b > 0 and b < 1:
                kappa.append(-np.log(b)*252); mu.append(a/(1-b)); sigma.append(np.sqrt(error_var/(1-np.power(b,2))))
                X_end.append(cumulative_residual[j, -1]); R_sq.append(reg.score(X, Y))
            else:
                kappa.append(np.nan); mu.append(np.nan); sigma.append(np.nan); X_end.append(np.nan); R_sq.append(np.nan)
        kappa_all[:, t_idx-t_idx_list[0]] = np.array(kappa)
        tau_all[:, t_idx-t_idx_list[0]] = 252/np.array(kappa)
        mu_all[:, t_idx-t_idx_list[0]] = np.array(mu)
        sigma_all[:, t_idx-t_idx_list[0]] = np.array(sigma)
        X_end_all[:, t_idx-t_idx_list[0]] = np.array(X_end)
        R_sq_all[:, t_idx-t_idx_list[0]] = np.array(R_sq)
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "trading_signal.npz"), kappa_all=kappa_all, tau_all=tau_all, mu_all=mu_all, sigma_all=sigma_all, X_end_all=X_end_all, R_sq_all=R_sq_all)


#%% calculate portfolio weights
portfolio_weights_all = np.zeros((len(equity_idx), len(t_idx_list))); portfolio_weights_all[:] = np.nan
portfolio_weights_prev = np.zeros(len(equity_idx)); portfolio_weights = np.zeros(len(equity_idx))

time_hist = []
asset = 1; asset_hist = []

position_binary = np.zeros((len(equity_idx), len(t_idx_list)))
portfolio_size = 0

optimization_hist = []; q_hist = np.zeros((len(equity_idx), len(t_idx_list))); q_hist[:] = np.nan
for j in tqdm.tqdm(range(len(t_idx_list))):
    # during market open, the portfolio weights evolve
    r_ = equity_data_.return_[equity_idx, t_idx_list[j]]
    portfolio_weights = np.multiply(portfolio_weights, 1+r_)
    portfolio_weights_prev = copy.deepcopy(portfolio_weights)

    # after the market close, determine portfolio weights
    time_hist.append(equity_data_.time_axis[t_idx_list[j]])
    asset += np.sum(portfolio_weights); asset_hist.append(asset)
    
    result = market_decomposition(equity_data_.time_axis[t_idx_list[j]])
    L = result["L"]
    kappa = kappa_all[:, j]; tau = tau_all[:, j]; mu = mu_all[:, j]; sigma = sigma_all[:, j]; X_end = X_end_all[:, j]; R_sq = R_sq_all[:, j]
    tau_avg = np.nanmean(tau_all[:, max(0, j-5):(j+1)], axis=1)
    sort_idx = np.argsort(tau_avg)

    position_binary[:, j] = copy.deepcopy(position_binary[:, j-1])
    new_position_cache = []
    for k in range(len(equity_idx)):
        if ~np.isnan(kappa[sort_idx[k]]) and R_sq[sort_idx[k]]>0.7:
            signal = (X_end[sort_idx[k]]-mu[sort_idx[k]])/sigma[sort_idx[k]]
            if position_binary[sort_idx[k], j-1] > 0:
                if signal > -threshold_close:
                    position_binary[sort_idx[k], j] = 0
                    portfolio_size -= 1
            elif position_binary[sort_idx[k], j-1] < 0:
                if signal < threshold_close:
                    position_binary[sort_idx[k], j] = 0
                    portfolio_size -= 1
            else:
                if signal > threshold_open:
                    new_position_cache.append([sort_idx[k], -1])
                if signal < -threshold_open:
                    new_position_cache.append([sort_idx[k], 1])
        else:
            if np.abs(position_binary[sort_idx[k], j-1]) > 0:
                position_binary[sort_idx[k], j] = 0
                portfolio_size -= 1

    for k in new_position_cache:
        if portfolio_size < 75:
            position_binary[k[0], j] = k[1]
            portfolio_size += 1

    if j == 0:
        idx_change = np.where(np.abs(position_binary[:, j])>0)[0]
        idx_unchange = np.where(np.abs(position_binary[:, j])==0)[0]
    else:
        idx_change = np.where(np.abs(position_binary[:, j]-position_binary[:, j-1])>0)[0]
        idx_unchange = np.where(np.abs(position_binary[:, j]-position_binary[:, j-1])==0)[0]
    
    if len(idx_change) > 0:
        # target
        L = L[idx_change, :]

        # constraint 1: compliance with signal indication
        A = np.zeros((len(idx_change), len(idx_change)))
        for k in range(len(idx_change)):
            A[k, k] = -1 if position_binary[idx_change[k], j] == 1 else 0
            A[k, k] = 1 if position_binary[idx_change[k], j] == -1 else 0

        # constraint 2: compliance with dollar neutrality
        #A_eq = np.ones((1, len(idx_change)))
        #q_tilde = portfolio_weights_prev[idx_unchange]
        #b_eq = -np.sum(q_tilde)
        A_tilde = np.zeros((len(idx_change), len(idx_change)))
        for k in range(len(idx_change)):
            if position_binary[idx_change[k], j] == 0:
                A_tilde[k, k] = 1
        A_eq = np.vstack((A_tilde, np.ones((1, len(idx_change)))))
        q_tilde = portfolio_weights_prev[idx_unchange]
        b_eq = np.zeros((len(idx_change)+1, 1)); b_eq[-1, 0] = -np.sum(q_tilde)
        
        # constraint 3: compliance with leverage limit    
        d = asset - np.linalg.norm(q_tilde, ord=1)

        q = cp.Variable((len(idx_change), 1))
        prob = cp.Problem(cp.Minimize(cp.norm(L.T@q, 1)),
                            [A@q <= 0, A_eq@q == b_eq, cp.norm(q, 1) <= d])
        prob.solve(solver="SCS")
        q = q.value.flatten()

    portfolio_weights = np.zeros(len(equity_idx))
    portfolio_weights[idx_change] = q
    portfolio_weights[idx_unchange] = portfolio_weights_prev[idx_unchange]
    portfolio_weights_all[:, j] = portfolio_weights
    asset -= np.sum(portfolio_weights) + 0.0005*np.linalg.norm(portfolio_weights-portfolio_weights_prev, ord=1)

    q_hist[idx_change, j] = q
    optimization_hist.append([np.linalg.norm(L.T@q, 1), np.sum(A@q), np.linalg.norm(A_eq@q-b_eq, 1), np.linalg.norm(q, 1)-d])

optimization_hist = np.array(optimization_hist).T

#%%
plt.figure(figsize=(12, 6))
plt.plot(time_hist, asset_hist)

#%%
plt.figure(figsize=(12, 6))
plt.plot(time_hist, optimization_hist[0, :], label="market neurality")
plt.plot(time_hist, optimization_hist[1, :], label="signal compliance")
plt.plot(time_hist, optimization_hist[2, :], label="dollar neurality")
plt.plot(time_hist, optimization_hist[3, :], label="leverage limit")
plt.legend()

#%%
plt.figure(figsize=(8, 6))
plt.imshow(position_binary, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("binary position")
plt.colorbar()

#%%
plt.figure(figsize=(8, 6))
plt.imshow(portfolio_weights_all, aspect="auto", cmap="RdBu_r")
plt.title("binary position")
plt.colorbar()

#%%



#%% calculate portfolio weights
class Optimization(nn.Module):
    def __init__(self, q_len):
        super(Optimization, self).__init__()
        self.q = nn.Parameter(torch.rand((q_len, 1), dtype=torch.float32))

    def forward(self, L, A, A_eq, b_eq, d):
        L = L/np.linalg.norm(L, 'fro')
        L = torch.from_numpy(L).to(torch.float32)
        A = torch.from_numpy(A).to(torch.float32)
        A_eq = torch.from_numpy(A_eq).to(torch.float32)
        b_eq = torch.from_numpy(b_eq).to(torch.float32)

        target = torch.norm(L.T@self.q, 1)
        penalty_1 = torch.sum(torch.relu(A@self.q))
        penalty_2 = torch.norm(A_eq@self.q-b_eq, 1)
        penalty_3 = torch.abs(torch.norm(self.q, 1) - d)
        return target + 1000*(penalty_1 + penalty_2 + penalty_3)

t_backtest_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_backtest_start), np.searchsorted(equity_data_.time_axis, t_backtest_end)+1, 1)
position = np.zeros((len(equity_idx), len(t_backtest_idx)))
tau_hist = np.zeros((len(equity_idx), len(t_backtest_idx)))
time_hist = []
asset = 1; asset_hist = []
train_hist = []; penalty_hist = []
portfolio_weight = np.zeros(len(equity_idx)); portfolio_weight_hist = []
for j in tqdm.tqdm(range(len(t_backtest_idx))):
    # at the market close, determine portfolio weights and rebalance
    t_idx = t_backtest_idx[j]
    time_hist.append(equity_data_.time_axis[t_idx])
    asset += np.sum(portfolio_weight); asset_hist.append(asset)
    result = market_decomposition(equity_data_.time_axis[t_idx])
    L = result["L"]
    result = trading_signal(equity_data_.time_axis[t_idx])
    kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]; X_end = result["X_end"]; R_sq = result["R_sq"]
    tau_hist[:, j] = 252/kappa
    tau_avg = np.nanmean(tau_hist[:, max(0, j-23):j+1], axis=1)
    for k in range(len(equity_idx)):
        if ~np.isnan(kappa[k]) and tau_avg[k]>tau_interval[0] and tau_avg[k]<tau_interval[1] and R_sq[k]>0.7:
            signal = (X_end[k]-mu[k])/sigma[k]
            if position[k, j-1] > 0:
                if signal > -threshold_close:
                    position[k, j] = 0
            elif position[k, j-1] < 0:
                if signal < threshold_close:
                    position[k, j] = 0
            else:
                if signal > threshold_open:
                    position[k, j] = -1
                if signal < -threshold_open:
                    position[k, j] = 1

    idx_change = np.where(np.abs(position[:, j]-position[:, j-1])>0)[0]
    idx_unchange = np.where(np.abs(position[:, j]-position[:, j-1])==0)[0]

    A = np.zeros((len(idx_change), len(idx_change)))
    A_tilde = np.zeros((len(idx_change), len(idx_change)))
    for k in range(len(idx_change)):
        A[k, k] = -1 if position[k, j] == 1 else 1
        A_tilde[k, k] = 1 if position[k, j] == 0 else 0
    L = L[idx_change, :]
    #A_eq = np.vstack((A_tilde, np.ones((1, len(idx_change)))))
    #b_eq = np.zeros((len(idx_change)+1, 1)); b_eq[-1, 0] = -np.sum(q_tilde)
    q_tilde = portfolio_weight[idx_unchange]
    A_eq = np.ones((1, len(idx_change)))
    b_eq = np.array([np.sum(q_tilde)])
    d = asset - np.linalg.norm(q_tilde, ord=1)
    
    '''
    q = cp.Variable((len(idx_change), 1))
    prob = cp.Problem(cp.Minimize(cp.norm(L.T@q, 1)),
                      [A@q <= 0, A_eq@q == b_eq, cp.norm(q, 1) == d])
    prob.solve()
    q = cp.Variable((len(idx_change), 1))
    prob = cp.Problem(cp.Minimize(0),
                      [A@q <= 0, A_eq@q == b_eq, cp.norm(L.T@q, 1) == 0])
    prob.solve()
    '''
    model = Optimization(len(idx_change))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_hist = []
    for epoch in range(5000):
        optimizer.zero_grad()
        loss = model.forward(L, A, A_eq, b_eq, d)
        loss.backward()
        optimizer.step()
        loss_hist.append(float(loss.detach()))
    train_hist.append(loss_hist)
    q = model.q.detach().numpy()
    penalty = [np.linalg.norm(L.T@q, 1), np.sum(np.maximum(A@q, 0)), np.linalg.norm(A_eq@q-b_eq, ord=1), np.abs(np.linalg.norm(q, 1)-d)]
    penalty_hist.append(penalty)

    portfolio_weight_prev = copy.deepcopy(portfolio_weight)
    portfolio_weight = np.zeros(len(equity_idx))
    portfolio_weight[idx_change] = q.flatten()
    portfolio_weight[idx_unchange] = q_tilde.flatten()
    portfolio_weight_hist.append(copy.deepcopy(portfolio_weight))
    asset -= np.sum(portfolio_weight) + 0.0005*np.linalg.norm(portfolio_weight-portfolio_weight_prev, ord=1)

    # the portfolio weights evolve during market open
    asset *= (1+0.02/252)
    r = equity_data_.return_[equity_idx, t_idx+1]
    for k in range(len(equity_idx)):
        portfolio_weight[k] *= (1+r[k])

time_hist.append(equity_data_.time_axis[t_idx+1])
asset += np.sum(portfolio_weight); asset_hist.append(asset)

#%% save results
portfolio_weight_hist = np.array(portfolio_weight_hist).T
train_hist = np.array(train_hist).T
penalty_hist = np.array(penalty_hist).T
time_hist = np.array(time_hist)
result = {"time_hist": [j.timestamp() for j in time_hist], "asset_hist": asset_hist,
          "position": position, "tau_hist": tau_hist, "portfolio_weight_hist": portfolio_weight_hist, 
          "train_hist": train_hist, "penalty_hist": penalty_hist}

file_name = os.path.join(os.path.dirname(__file__), "result.npz")
pickle.dump(result, open(file_name, "wb"))


#%%


# %%
