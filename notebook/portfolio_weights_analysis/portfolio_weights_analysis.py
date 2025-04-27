#%%
import os, sys, copy, h5py, datetime, tqdm, gc, pickle
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.colors 
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import utils.utils as utils

#%%
QUICK_TEST = False
PCA_TYPE = "name"; factor_name = "PCA_name"
#PCA_TYPE = "rank_permutation"; factor_name = "PCA_rank_permutation"
#PCA_TYPE = "rank_theta_transform"; factor_name = "PCA_rank_theta"
#PCA_TYPE = "rank_hybrid_Atlas"; factor_name = "PCA_rank_hybrid_Atlas"

t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)
t_eval_start = datetime.datetime(2011,1,1); t_eval_end = datetime.datetime(2022,12,15)

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
                "theta_evaluation_window_length": 60,
                "rank_min": 0,
                "rank_max": 999,
                "factor_number": 3,
                "type": PCA_TYPE,
                "max_cache_len": 100,
                "quick_test": QUICK_TEST}
PCA_factor_ = factor.PCA_factor(equity_data_, equity_data_high_freq_, PCA_factor_config)

#%% portfolio weight analysis: OU process
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
match PCA_TYPE:
    case "name":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
        portfolio_weights_R = result["portfolio_weights_R"]
    case "rank_permutation":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
        portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    case "rank_theta_transform":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
        portfolio_weights_R_rank = result["portfolio_weights_R_rank"]
    case "rank_hybrid_Atlas":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
        portfolio_weights_R_rank = result["portfolio_weights_R_rank"]

def average_holding_days(portfolio_weights):
    holding_days = []
    for j in range(portfolio_weights.shape[0]):
        idx = [k for k in range(1, portfolio_weights.shape[1], 1) if portfolio_weights[j, k] != portfolio_weights[j, k-1]]
        if len(idx) > 2: holding_days.append(np.mean(np.diff(idx)))
    return np.mean(holding_days)

colors = ['green', 'white', 'red']
boundaries = [-1.5, -0.5, 0.5, 1.5]
colormap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(boundaries, colormap.N, clip=True)

if PCA_TYPE == "name":
    average_holding_days_list = []
    plt.figure(figsize=(18, 18))
    year_list = np.arange(1991, 2022, 1)
    for year in tqdm.tqdm(year_list):
        t_idx = np.arange(np.searchsorted(time, datetime.datetime(year, 1, 1)), np.searchsorted(time, datetime.datetime(year, 12, 31))+1, 1)
        portfolio_weights_epsilon_sub = np.zeros((len(equity_data_.equity_idx_list), len(t_idx)))
        for j in t_idx:
            portfolio_weights_epsilon_sub[:, j-t_idx[0]] = portfolio_weights_epsilon[:, j]
        idx_equity_axis = np.unique(equity_data_.equity_idx_by_rank[0:(PCA_factor_.config["rank_max"]+1), :][:, t_idx].flatten()).astype(int)
        portfolio_weights_epsilon_sub = portfolio_weights_epsilon_sub[idx_equity_axis, :]
        average_holding_days_list.append(average_holding_days(portfolio_weights_epsilon_sub))

        plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+1)
        plt.imshow(portfolio_weights_epsilon_sub, aspect="auto", cmap=colormap, norm=norm, interpolation="none")
        plt.colorbar()
        plt.title("year: {}".format(year))
        plt.xlabel("time (days)"); plt.ylabel("equity index")
        plt.xticks([0, len(t_idx)//2, len(t_idx)-1], [time[j].strftime("%Y/%m/%d") for j in t_idx[[0, len(t_idx)//2, len(t_idx)-1]]])

    plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+2)
    plt.plot(year_list, average_holding_days_list, marker="o")
    plt.xlabel("year"); plt.ylabel("avg. hold. days")
    plt.suptitle("portfolio weights: PCA {}".format(factor_name))
    plt.tight_layout()

if PCA_TYPE in ["rank_permutation", "rank_theta_transform", "rank_hybrid_Atlas"]:
    average_holding_days_list = []
    plt.figure(figsize=(18, 18))
    year_list = np.arange(1991, 2022, 1)
    for year in tqdm.tqdm(year_list):
        t_idx = np.arange(np.searchsorted(time, datetime.datetime(year, 1, 1)), np.searchsorted(time, datetime.datetime(year, 12, 31))+1, 1)
        idx_epsilon_axis = np.arange(PCA_factor_.config["rank_min"], PCA_factor_.config["rank_max"]+1, 1)
        portfolio_weights_epsilon_sub = portfolio_weights_epsilon[idx_epsilon_axis, :][:, t_idx]
        average_holding_days_list.append(average_holding_days(portfolio_weights_epsilon_sub))

        plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+1)
        plt.imshow(portfolio_weights_epsilon_sub, aspect="auto", cmap=colormap, norm=norm, interpolation="none")
        plt.colorbar()
        plt.title("year: {}".format(year))
        plt.xlabel("time (days)"); plt.ylabel("rank index")
        plt.xticks([0, len(t_idx)//2, len(t_idx)-1], [time[j].strftime("%Y/%m/%d") for j in t_idx[[0, len(t_idx)//2, len(t_idx)-1]]])

    plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+2)
    plt.plot(year_list, average_holding_days_list, marker="o")
    plt.xlabel("year"); plt.ylabel("avg. hold. days")
    plt.suptitle("portfolio weights: PCA {}".format(factor_name))
    plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_epsilon_map_{}.png".format(factor_name)), dpi=300)

#%% portfolio weight analysis: CNN+transformer

file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
match PCA_TYPE:
    case "name":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
    case "rank_hybrid_Atlas_high_freq":
        time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
        portfolio_weights_epsilon = result["portfolio_weights_epsilon"]

def average_holding_days(portfolio_weights):
    holding_days = []
    for j in range(portfolio_weights.shape[0]):
        idx = [k for k in range(1, portfolio_weights.shape[1], 1) if portfolio_weights[j, k] != portfolio_weights[j, k-1]]
        if len(idx) > 2: holding_days.append(np.mean(np.diff(idx)))
    return np.mean(holding_days)

colors = ['green', 'white', 'red']
boundaries = [-1.5, -0.5, 0.5, 1.5]
colormap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(boundaries, colormap.N, clip=True)

if PCA_TYPE == "name":
    average_holding_days_list = []
    plt.figure(figsize=(18, 18))
    year_list = np.arange(t_eval_start.year, t_eval_end.year, 1)
    for year in tqdm.tqdm(year_list):
        t_idx = np.array([j for j in range(len(time)) if (time[j].year == year)])
        portfolio_weights_epsilon_sub = np.zeros((len(equity_data_.equity_idx_list), len(t_idx)))
        for j in t_idx:
            portfolio_weights_epsilon_sub[:, j-t_idx[0]] = portfolio_weights_epsilon[:, j]
        t_idx_global = np.arange(np.searchsorted(equity_data_.time_axis, time[t_idx[0]]), np.searchsorted(equity_data_.time_axis, time[t_idx[-1]])+1, 1)
        idx_equity_axis = np.unique(equity_data_.equity_idx_by_rank[0:(PCA_factor_.config["rank_max"]+1), :][:, t_idx_global].flatten()).astype(int)
        sort_idx = np.argsort(np.nanmax(equity_data_.capitalization[idx_equity_axis, t_idx_global], axis=1))[::-1]
        idx_equity_axis = idx_equity_axis[sort_idx]
        portfolio_weights_epsilon_sub = portfolio_weights_epsilon_sub[idx_equity_axis, :]
        average_holding_days_list.append(average_holding_days(portfolio_weights_epsilon_sub))

        plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+1)
        plt.imshow(portfolio_weights_epsilon_sub, aspect="auto", cmap="RdBu", interpolation="none")
        plt.colorbar()
        plt.title("year: {}".format(year))
        plt.xlabel("time (days)"); plt.ylabel("equity index")
        plt.xticks([0, len(t_idx)//2, len(t_idx)-1], [time[j].strftime("%Y/%m/%d") for j in t_idx[[0, len(t_idx)//2, len(t_idx)-1]]])

    plt.subplot(np.ceil(len(year_list)/4).astype(int), 4, np.searchsorted(year_list, year)+2)
    plt.plot(year_list, average_holding_days_list, marker="o")
    plt.xlabel("year"); plt.ylabel("avg. hold. days")
    plt.suptitle("portfolio weights: PCA {}".format(factor_name))
    plt.tight_layout()

#%% summary plot: dollar neurality and active positions

eval_t_start = datetime.datetime(2006,1,1); eval_t_end = datetime.datetime(2022,12,15)
plt.figure(figsize=(18, 10))
plt.subplot(4,2,1)
PCA_TYPE = "name"; factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
t_idx = np.arange(np.searchsorted(time, eval_t_start), np.searchsorted(time, eval_t_end)+1, 1)
time = [time[j] for j in t_idx]
portfolio_weights_epsilon = result["portfolio_weights_epsilon"][:, t_idx]
portfolio_weights_epsilon = portfolio_weights_epsilon/np.sum(np.abs(portfolio_weights_epsilon), axis=0, keepdims=True)
portfolio_weights_R = result["portfolio_weights_R"][:, t_idx]
portfolio_weights_R = portfolio_weights_R/np.sum(np.abs(portfolio_weights_R), axis=0, keepdims=True)
positive_position_epsilon = []; negative_position_epsilon = []
for j in range(portfolio_weights_epsilon.shape[1]):
    pos_idx = np.where(portfolio_weights_epsilon[:, j] > 0)[0]
    neg_idx = np.where(portfolio_weights_epsilon[:, j] < 0)[0]
    positive_position_epsilon.append([np.sum(portfolio_weights_epsilon[pos_idx, j]), np.std(portfolio_weights_epsilon[pos_idx, j])])
    negative_position_epsilon.append([np.sum(portfolio_weights_epsilon[neg_idx, j]), np.std(portfolio_weights_epsilon[neg_idx, j])])
positive_position_epsilon = np.array(positive_position_epsilon).T; negative_position_epsilon = np.array(negative_position_epsilon).T
positive_position_R = np.sum(np.maximum(portfolio_weights_R, 0), axis=0)
negative_position_R = np.sum(np.minimum(portfolio_weights_R, 0), axis=0)
dollar_neutrality = (positive_position_R+negative_position_R)/(np.abs(positive_position_R)+np.abs(negative_position_R))
plt.plot(time, positive_position_epsilon[0, :])
#plt.fill_between(time, positive_position_epsilon[0, :]-positive_position_epsilon[1, :], positive_position_epsilon[0, :]+positive_position_epsilon[1, :], alpha=0.3)
plt.plot(time, negative_position_epsilon[0, :])
#plt.fill_between(time, negative_position_epsilon[0, :]-negative_position_epsilon[1, :], negative_position_epsilon[0, :]+negative_position_epsilon[1, :], alpha=0.3)
plt.ylim(-1, 1)
plt.tick_params(axis='both', direction='in')
plt.xlabel("time"); plt.ylabel(r"$\sum_{w_{i,t}^\epsilon > 0} w_{i,t}^\epsilon$ or $\sum_{w_{i,t}^\epsilon < 0} w_{i,t}^\epsilon$")
plt.legend(title="parametric model in name space", loc="upper right")
plt.subplot(4,2,2)
plt.plot(time, dollar_neutrality)
plt.xlabel("time"); plt.ylabel("dollar neutrality")
plt.ylim(-0.5, 0.5)
plt.tick_params(axis='both', direction='in')
plt.legend(title="parametric model in name space", loc="upper right")

plt.subplot(4,2,3)
PCA_TYPE = "name"; factor_name = "PCA_name"
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
t_idx = np.arange(np.searchsorted(time, eval_t_start), np.searchsorted(time, eval_t_end)+1, 1)
time = [time[j] for j in t_idx]
portfolio_weights_epsilon = result["portfolio_weights_epsilon"][:, t_idx]
portfolio_weights_epsilon = portfolio_weights_epsilon/np.sum(np.abs(portfolio_weights_epsilon), axis=0, keepdims=True)
portfolio_weights_R = result["portfolio_weights_R"][:, t_idx]
portfolio_weights_R = portfolio_weights_R/np.sum(np.abs(portfolio_weights_R), axis=0, keepdims=True)
positive_position_epsilon = []; negative_position_epsilon = []
for j in range(portfolio_weights_epsilon.shape[1]):
    pos_idx = np.where(portfolio_weights_epsilon[:, j] > 0)[0]
    neg_idx = np.where(portfolio_weights_epsilon[:, j] < 0)[0]
    positive_position_epsilon.append([np.sum(portfolio_weights_epsilon[pos_idx, j]), np.std(portfolio_weights_epsilon[pos_idx, j])])
    negative_position_epsilon.append([np.sum(portfolio_weights_epsilon[neg_idx, j]), np.std(portfolio_weights_epsilon[neg_idx, j])])
positive_position_epsilon = np.array(positive_position_epsilon).T; negative_position_epsilon = np.array(negative_position_epsilon).T
positive_position_R = np.sum(np.maximum(portfolio_weights_R, 0), axis=0)
negative_position_R = np.sum(np.minimum(portfolio_weights_R, 0), axis=0)
dollar_neutrality = (positive_position_R+negative_position_R)/(np.abs(positive_position_R)+np.abs(negative_position_R))
plt.plot(time, positive_position_epsilon[0, :])
#plt.fill_between(time, positive_position_epsilon[0, :]-positive_position_epsilon[1, :], positive_position_epsilon[0, :]+positive_position_epsilon[1, :], alpha=0.3)
plt.plot(time, negative_position_epsilon[0, :])
#plt.fill_between(time, negative_position_epsilon[0, :]-negative_position_epsilon[1, :], negative_position_epsilon[0, :]+negative_position_epsilon[1, :], alpha=0.3)
plt.xlabel("time"); plt.ylabel(r"$\sum_{w_{i,t}^\epsilon > 0} w_{i,t}^\epsilon$ or $\sum_{w_{i,t}^\epsilon < 0} w_{i,t}^\epsilon$")
plt.ylim(-1, 1)
plt.tick_params(axis='both', direction='in')
plt.legend(title="neural networks in name space", loc="upper right")
plt.subplot(4,2,4)
plt.plot(time, dollar_neutrality)
plt.xlabel("time"); plt.ylabel("dollar neutrality")
plt.ylim(-0.5, 0.5)
plt.tick_params(axis='both', direction='in')
plt.legend(title="neural networks in name space", loc="upper right")

plt.subplot(4,2,5)
PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
t_idx = np.arange(np.searchsorted(time, eval_t_start), np.searchsorted(time, eval_t_end)+1, 1)
time = [time[j] for j in t_idx]
portfolio_weights_epsilon = result["portfolio_weights_epsilon"][:, t_idx]
portfolio_weights_epsilon = portfolio_weights_epsilon/np.sum(np.abs(portfolio_weights_epsilon), axis=0, keepdims=True)
portfolio_weights_R = result["portfolio_weights_R_rank"][:, t_idx]
portfolio_weights_R = portfolio_weights_R/np.sum(np.abs(portfolio_weights_R), axis=0, keepdims=True)
positive_position_epsilon = []; negative_position_epsilon = []
for j in range(portfolio_weights_epsilon.shape[1]):
    pos_idx = np.where(portfolio_weights_epsilon[:, j] > 0)[0]
    neg_idx = np.where(portfolio_weights_epsilon[:, j] < 0)[0]
    positive_position_epsilon.append([np.sum(portfolio_weights_epsilon[pos_idx, j]), np.std(portfolio_weights_epsilon[pos_idx, j])])
    negative_position_epsilon.append([np.sum(portfolio_weights_epsilon[neg_idx, j]), np.std(portfolio_weights_epsilon[neg_idx, j])])
positive_position_epsilon = np.array(positive_position_epsilon).T; negative_position_epsilon = np.array(negative_position_epsilon).T
positive_position_R = np.sum(np.maximum(portfolio_weights_R, 0), axis=0)
negative_position_R = np.sum(np.minimum(portfolio_weights_R, 0), axis=0)
dollar_neutrality = (positive_position_R+negative_position_R)/(np.abs(positive_position_R)+np.abs(negative_position_R))
plt.plot(time, positive_position_epsilon[0, :])
#plt.fill_between(time, positive_position_epsilon[0, :]-positive_position_epsilon[1, :], positive_position_epsilon[0, :]+positive_position_epsilon[1, :], alpha=0.3)
plt.plot(time, negative_position_epsilon[0, :])
#plt.fill_between(time, negative_position_epsilon[0, :]-negative_position_epsilon[1, :], negative_position_epsilon[0, :]+negative_position_epsilon[1, :], alpha=0.3)
plt.xlabel("time"); plt.ylabel(r"$\sum_{w_{i,t}^\epsilon > 0} w_{i,t}^\epsilon$ or $\sum_{w_{i,t}^\epsilon < 0} w_{i,t}^\epsilon$")
plt.legend(title="parametric model in rank space", loc="upper right")
plt.ylim(-1, 1)
plt.tick_params(axis='both', direction='in')
plt.legend(title="parametric model in rank space", loc="upper right")
plt.subplot(4,2,6)
plt.plot(time, dollar_neutrality)
plt.xlabel("time"); plt.ylabel("dollar neutrality")
plt.ylim(-0.5, 0.5)
plt.tick_params(axis='both', direction='in')
plt.legend(title="parametric model in rank space", loc="upper right")

plt.subplot(4,2,7)
PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
result = np.load(file_name, allow_pickle=True)
time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
t_idx = np.arange(np.searchsorted(time, eval_t_start), np.searchsorted(time, eval_t_end)+1, 1)
time = [time[j] for j in t_idx]
portfolio_weights_epsilon = result["portfolio_weights_epsilon"][:, t_idx]
portfolio_weights_epsilon = portfolio_weights_epsilon/np.sum(np.abs(portfolio_weights_epsilon), axis=0, keepdims=True)
portfolio_weights_R = result["portfolio_weights_R_rank"][:, t_idx]
portfolio_weights_R = portfolio_weights_R/np.sum(np.abs(portfolio_weights_R), axis=0, keepdims=True)
positive_position_epsilon = []; negative_position_epsilon = []
for j in range(portfolio_weights_epsilon.shape[1]):
    pos_idx = np.where(portfolio_weights_epsilon[:, j] > 0)[0]
    neg_idx = np.where(portfolio_weights_epsilon[:, j] < 0)[0]
    positive_position_epsilon.append([np.sum(portfolio_weights_epsilon[pos_idx, j]), np.std(portfolio_weights_epsilon[pos_idx, j])])
    negative_position_epsilon.append([np.sum(portfolio_weights_epsilon[neg_idx, j]), np.std(portfolio_weights_epsilon[neg_idx, j])])
positive_position_epsilon = np.array(positive_position_epsilon).T; negative_position_epsilon = np.array(negative_position_epsilon).T
positive_position_R = np.sum(np.maximum(portfolio_weights_R, 0), axis=0)
negative_position_R = np.sum(np.minimum(portfolio_weights_R, 0), axis=0)
dollar_neutrality = (positive_position_R+negative_position_R)/(np.abs(positive_position_R)+np.abs(negative_position_R))
plt.plot(time, positive_position_epsilon[0, :])
#plt.fill_between(time, positive_position_epsilon[0, :]-positive_position_epsilon[1, :], positive_position_epsilon[0, :]+positive_position_epsilon[1, :], alpha=0.3)
plt.plot(time, negative_position_epsilon[0, :])
#plt.fill_between(time, negative_position_epsilon[0, :]-negative_position_epsilon[1, :], negative_position_epsilon[0, :]+negative_position_epsilon[1, :], alpha=0.3)
plt.xlabel("time"); plt.ylabel(r"$\sum_{w_{i,t}^\epsilon > 0} w_{i,t}^\epsilon$ or $\sum_{w_{i,t}^\epsilon < 0} w_{i,t}^\epsilon$")
plt.ylim(-1, 1)
plt.tick_params(axis='both', direction='in')
plt.legend(title="neural networks in rank space", loc="upper right")
plt.subplot(4,2,8)
plt.plot(time, dollar_neutrality)
plt.xlabel("time"); plt.ylabel("dollar neutrality")
plt.ylim(-0.5, 0.5)
plt.tick_params(axis='both', direction='in')
plt.legend(title="neural networks in rank space", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "dollar_neutrality.pdf"), dpi=300)



#%%


