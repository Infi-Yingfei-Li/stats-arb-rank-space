#%%
import os, sys, copy, h5py, datetime, tqdm
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional
torch.set_printoptions(precision=7)

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import neural_network.neural_network as neural_network
import portfolio_weights.portfolio_weights as portfolio_weights
import utils.utils as utils

#%%
QUICK_TEST = False
PCA_TYPE = "name"; factor_name = "PCA_name"
#PCA_TYPE = "rank_hybrid_Atlas_high_freq"; factor_name = "PCA_rank_hybrid_Atlas_high_freq"

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_dict = {"PCA_name": color_cycle[0], "PCA_rank_permutation": color_cycle[1], "PCA_rank_hybrid_Atlas": color_cycle[2], 
              "PCA_rank_hybrid_Atlas_high_freq": color_cycle[3], "PCA_rank_theta": color_cycle[4]}

network_train_time_interval = []
if PCA_TYPE == "name":
    network_folder = os.path.join(os.path.dirname(__file__), "../../neural_network/CNN_transformer/name")
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    network_folder = os.path.join(os.path.dirname(__file__), "../../neural_network/CNN_transformer/rank_hybrid_Atlas")
for file in os.listdir(network_folder):
    if file.endswith(".pt") and (file.split("_")[-2]!="initial"):
        time_start = file.split("_")[-2]
        time_end = file.split("_")[-1].split(".")[0]
        network_train_time_interval.append([datetime.datetime.strptime(time_start, "%Y%m%d"), datetime.datetime.strptime(time_end, "%Y%m%d")])

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
#PCA_factor_._initialize_residual_return_all()

trading_signal_OU_process_config = {"max_cache_len": 100}
trading_signal_OU_process_ = trading_signal.trading_signal_OU_process(equity_data_, equity_data_high_freq_, PCA_factor_, trading_signal_OU_process_config)

#%% training log
target_in_sample_hist_all = []; target_out_of_sample_hist_all = []
plt.figure(figsize=(18, 18))
for j in range(len(network_train_time_interval)):
    train_t_start = network_train_time_interval[j][0]; train_t_end = network_train_time_interval[j][1]
    portfolio_weights_CNN_transformer_config = {"PnL_evaluation_window_length": 24,
                                                "risk_aversion_factor": 2,
                                                "tau_aversion_factor": None,
                                                "transaction_cost_aversion_factor": None,
                                                "dollar_neutrality_aversion_factor": 1,
                                                "train_t_start": datetime.datetime(1991, 7, 1),
                                                "train_t_end": datetime.datetime(1991, 10, 31),
                                                "valid_t_start": datetime.datetime(1991, 11, 1),
                                                "valid_t_end": datetime.datetime(1991, 12, 15),
                                                "epoch_max": 50,
                                                "learning_rate": 1e-3,
                                                "CNN_input_channels": 1,
                                                "CNN_output_channels": 8,
                                                "CNN_kernel_size": 2,
                                                "CNN_drop_out_rate": 0.25,
                                                "transformer_input_channels": 8,
                                                "transformer_hidden_channels": 2*8,
                                                "transformer_output_channels": 8,
                                                "transformer_head_num": 4,
                                                "transformer_drop_out_rate": 0.25,
                                                "temporal_batch_num":4,
                                                "is_initialize_from_pretrain": False
                                                }
    network_config = portfolio_weights_CNN_transformer_config

    network = neural_network.CNN_transformer(network_config["CNN_input_channels"], network_config["CNN_output_channels"], network_config["CNN_kernel_size"], network_config["CNN_drop_out_rate"],
                                                        network_config["transformer_input_channels"], network_config["transformer_hidden_channels"], network_config["transformer_output_channels"], network_config["transformer_head_num"], network_config["transformer_drop_out_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=network_config["learning_rate"])

    if PCA_TYPE == "name":
        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../../neural_network/CNN_transformer/name/neural_network_CNN_transformer_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
    if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
        network_file_name = os.path.join(os.path.dirname(__file__), ''.join(["../../neural_network/CNN_transformer/rank_hybrid_Atlas/neural_network_CNN_transformer_", datetime.datetime.strftime(train_t_start, "%Y%m%d"), "_", datetime.datetime.strftime(train_t_end, "%Y%m%d"), ".pt"]))
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), network_file_name), map_location=device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    target_in_sample_hist = checkpoint["in-sample target"]
    target_out_of_sample_hist = checkpoint["out-of-sample target"]
    target_in_sample_hist_all.append(target_in_sample_hist)
    target_out_of_sample_hist_all.append(target_out_of_sample_hist)

    plt.subplot(max(np.ceil(len(network_train_time_interval)/3).astype(int), 1), 3, j+1)
    plt.plot(target_in_sample_hist, label="in-sample target")
    plt.plot(target_out_of_sample_hist, label="validation target")
    plt.title("{}-{}".format(datetime.datetime.strftime(train_t_start, "%Y/%m/%d"), datetime.datetime.strftime(train_t_end, "%Y/%m/%d")))
    plt.xlabel("epoch"); plt.ylabel("target")
    plt.hlines(0, 0, len(target_in_sample_hist), color="red", linestyle="--")
    plt.legend()

plt.tight_layout()
plt.suptitle("{}: train log".format(PCA_TYPE))
plt.savefig(os.path.join(os.path.dirname(__file__), "train_log_{}.pdf".format(PCA_TYPE)))

#%% training log average
target_in_sample_hist_all = np.array(target_in_sample_hist_all)
target_out_of_sample_hist_all = np.array(target_out_of_sample_hist_all)
initial_in_sample = np.random.uniform(-0.001, 0.001, target_in_sample_hist_all.shape[0])
initial_out_of_sample = np.random.uniform(-0.001, 0.001, target_out_of_sample_hist_all.shape[0])
target_in_sample_hist_all = np.concatenate([initial_in_sample.reshape(-1, 1), target_in_sample_hist_all], axis=1)
target_out_of_sample_hist_all = np.concatenate([initial_out_of_sample.reshape(-1, 1), target_out_of_sample_hist_all], axis=1)

plt.plot(np.arange(0, target_in_sample_hist_all.shape[1], 1), np.mean(target_in_sample_hist_all, axis=0), label="in-sample target", color=color_cycle[0])
for j in range(target_in_sample_hist_all.shape[0]):
    plt.plot(np.arange(0, target_in_sample_hist_all.shape[1], 1), target_in_sample_hist_all[j, :], color=color_cycle[0], alpha=0.1)
#plt.fill_between(np.arange(0, target_in_sample_hist_all.shape[1], 1), np.mean(target_in_sample_hist_all, axis=0)-np.std(target_in_sample_hist_all, axis=0), np.mean(target_in_sample_hist_all, axis=0)+np.std(target_in_sample_hist_all, axis=0), alpha=0.3)
plt.plot(np.arange(0, target_out_of_sample_hist_all.shape[1], 1), np.mean(target_out_of_sample_hist_all, axis=0), label="validation target", color=color_cycle[1])
for j in range(target_out_of_sample_hist_all.shape[0]):
    plt.plot(np.arange(0, target_out_of_sample_hist_all.shape[1], 1), target_out_of_sample_hist_all[j, :], color=color_cycle[1], alpha=0.1)
#plt.fill_between(np.arange(0, target_out_of_sample_hist_all.shape[1], 1), np.mean(target_out_of_sample_hist_all, axis=0)-np.std(target_out_of_sample_hist_all, axis=0), np.mean(target_out_of_sample_hist_all, axis=0)+np.std(target_out_of_sample_hist_all, axis=0), alpha=0.3)
plt.xlabel("epoch"); plt.ylabel("target")
plt.ylim([-0.001, 0.007])
plt.hlines(0, 0, target_in_sample_hist_all.shape[1], color="red", linestyle="--")
plt.legend(loc="upper left")
plt.savefig(os.path.join(os.path.dirname(__file__), "train_log_average_{}.pdf".format(PCA_TYPE)))


#%% portfolio weights epsilon
eval_t_start = datetime.datetime(2006,1,1); eval_t_end = datetime.datetime(2022,12,15)
file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-CNN_transformer.npz".format(factor_name))
result = np.load(file_name)
time = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
portfolio_weights_epsilon = result["portfolio_weights_epsilon"]
t_idx = [j for j in range(len(time)) if time[j]>=eval_t_start and time[j]<=eval_t_end]
time = [time[j] for j in t_idx]
portfolio_weights_epsilon = portfolio_weights_epsilon[:, t_idx]

file_name = os.path.join(os.path.dirname(__file__), "../../results/portfolio_performance_{}-OU_process-OU_process.npz").format(factor_name)
result = np.load(file_name)
time_OU = [datetime.datetime.fromtimestamp(j) for j in result["time"]]
portfolio_weights_epsilon_OU = result["portfolio_weights_epsilon"]
for j in range(portfolio_weights_epsilon_OU.shape[1]):
    portfolio_weights_epsilon_OU[:, j] /= np.linalg.norm(portfolio_weights_epsilon_OU[:, j], ord=1)
t_idx = [j for j in range(len(time_OU)) if time_OU[j]>=eval_t_start and time_OU[j]<=eval_t_end]
time_OU = [time_OU[j] for j in t_idx]
portfolio_weights_epsilon_OU = portfolio_weights_epsilon_OU[:, t_idx]


#%% portfolio weights in residual space by year
def average_holding_days(portfolio_weights):
    holding_days = []
    for j in range(portfolio_weights.shape[0]):
        idx = [k for k in range(1, portfolio_weights.shape[1], 1) if np.sign(portfolio_weights[j, k]) != np.sign(portfolio_weights[j, k-1])]
        if len(idx) > 2: holding_days.append(np.mean(np.diff(idx)))
    return np.mean(holding_days)

color_max = 0.02 if PCA_TYPE=="rank_hybrid_Atlas_high_freq" else 0.005
average_holding_time = []
average_holding_time_OU = []

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    plt.figure(figsize=(10, 18))
    for year in np.arange(eval_t_start.year, eval_t_end.year+1, 1):
        plt.subplot(eval_t_end.year-eval_t_start.year+1, 2, 2*(year-eval_t_start.year)+1)
        t_idx = [j for j in range(len(time)) if time[j].year == year]
        plt.imshow(portfolio_weights_epsilon[0:500, :][:, t_idx], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
        plt.xticks([0, len(t_idx)-1], [datetime.datetime.strftime(time[t_idx[0]], "%Y/%m/%d"), datetime.datetime.strftime(time[t_idx[-1]], "%Y/%m/%d")])
        plt.xlabel("time"); plt.ylabel("rank")
        plt.colorbar()
        average_holding_time.append(average_holding_days(portfolio_weights_epsilon[0:500, :][:, t_idx]))

        plt.subplot(eval_t_end.year-eval_t_start.year+1, 2, 2*(year-eval_t_start.year)+2)
        t_idx = [j for j in range(len(time)) if time_OU[j].year == year]
        plt.imshow(portfolio_weights_epsilon_OU[0:500, :][:, t_idx], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
        plt.xticks([0, len(t_idx)-1], [datetime.datetime.strftime(time_OU[t_idx[0]], "%Y/%m/%d"), datetime.datetime.strftime(time_OU[t_idx[-1]], "%Y/%m/%d")])
        plt.xlabel("time"); plt.ylabel("rank")
        plt.colorbar()
        average_holding_time_OU.append(average_holding_days(portfolio_weights_epsilon_OU[0:500, :][:, t_idx]))

    plt.suptitle(r"$w_t^{\epsilon}$ by left col: neural network; right col: OU process")
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_epsilon_{}.png".format(PCA_TYPE)), dpi=300)

if PCA_TYPE == "name":
    plt.figure(figsize=(10, 18))
    t_idx = np.arange(np.searchsorted(equity_data_.time_axis, eval_t_start), np.searchsorted(equity_data_.time_axis, eval_t_end)+1, 1)
    equity_idx = np.unique(equity_data_.equity_idx_by_rank[0:500, :][:, t_idx].flatten()).astype(int)
    for year in np.arange(eval_t_start.year, eval_t_end.year+1, 1):
        plt.subplot(eval_t_end.year-eval_t_start.year+1, 2, 2*(year-eval_t_start.year)+1)
        t_idx = [j for j in range(len(time)) if time[j].year == year]
        plt.imshow(portfolio_weights_epsilon[equity_idx, :][:, t_idx], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
        plt.xticks([0, len(t_idx)-1], [datetime.datetime.strftime(time[t_idx[0]], "%Y/%m/%d"), datetime.datetime.strftime(time[t_idx[-1]], "%Y/%m/%d")])
        plt.xlabel("time"); plt.ylabel("rank")
        plt.colorbar()
        average_holding_time.append(average_holding_days(portfolio_weights_epsilon[equity_idx, :][:, t_idx]))

        plt.subplot(eval_t_end.year-eval_t_start.year+1, 2, 2*(year-eval_t_start.year)+2)
        t_idx = [j for j in range(len(time)) if time_OU[j].year == year]
        plt.imshow(portfolio_weights_epsilon_OU[equity_idx, :][:, t_idx], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
        plt.xticks([0, len(t_idx)-1], [datetime.datetime.strftime(time_OU[t_idx[0]], "%Y/%m/%d"), datetime.datetime.strftime(time_OU[t_idx[-1]], "%Y/%m/%d")])
        plt.xlabel("time"); plt.ylabel("rank")
        plt.colorbar()
        average_holding_time_OU.append(average_holding_days(portfolio_weights_epsilon_OU[equity_idx, :][:, t_idx]))

    plt.suptitle(r"$w_t^{\epsilon}$ by left col: neural network; right col: OU process")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_epsilon_by_year_{}.png".format(PCA_TYPE)), dpi=300)


#%% average holding time
plt.figure(figsize=(7,4))
plt.plot(np.arange(eval_t_start.year, eval_t_end.year+1, 1), average_holding_time_OU, "o-",label="OU process")
plt.plot(np.arange(eval_t_start.year, eval_t_end.year+1, 1), average_holding_time, "o-", label="neural network")
plt.ylabel("average holding time (days)")
plt.ylim([0, 50])
plt.legend()
plt.savefig(os.path.join(os.path.dirname(__file__), "average_holding_time_{}.pdf".format(PCA_TYPE)), dpi=300)

#%% portfolio weights in residual space all time
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(portfolio_weights_epsilon[0:500, :], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
    x_tick_label = [datetime.datetime.strftime(datetime.datetime(j, 1, 1), "%Y/%m/%d") for j in np.arange(2006, 2022, 5)]
    x_tick_idx = [np.searchsorted(time, datetime.datetime.strptime(t, "%Y/%m/%d")) for t in x_tick_label]
    plt.xticks(x_tick_idx, x_tick_label)
    plt.xlabel("time"); plt.ylabel("rank")
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(portfolio_weights_epsilon_OU[0:500, :], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
    plt.xticks(x_tick_idx, x_tick_label)
    plt.xlabel("time"); plt.ylabel("rank")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_epsilon_{}.pdf".format(PCA_TYPE)), dpi=300)

if PCA_TYPE == "name":
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 1, 1)
    equity_idx = np.unique(equity_data_.equity_idx_by_rank[0:500, :].flatten()).astype(int)
    plt.imshow(portfolio_weights_epsilon[equity_idx, :], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
    x_tick_label = [datetime.datetime.strftime(datetime.datetime(j, 1, 1), "%Y/%m/%d") for j in np.arange(2006, 2022, 5)]
    x_tick_idx = [np.searchsorted(time, datetime.datetime.strptime(t, "%Y/%m/%d")) for t in x_tick_label]
    plt.xticks(x_tick_idx, x_tick_label)
    plt.xlabel("time"); plt.ylabel("rank")
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(portfolio_weights_epsilon_OU[equity_idx, :], cmap="RdBu_r", aspect="auto", vmin=-color_max, vmax=color_max)
    plt.xticks(x_tick_idx, x_tick_label)
    plt.xlabel("time"); plt.ylabel("rank")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "portfolio_weights_epsilon_{}.pdf".format(PCA_TYPE)), dpi=300)


#%%
signal = []
for j in tqdm.tqdm(range(len(time))):
    result = PCA_factor_.residual_return(time[j])
    epsilon_idx = result["epsilon_idx"]; epsilon = result["epsilon"]
    epsilon = epsilon / np.std(epsilon, axis=1, keepdims=True)
    cummulative_epsilon_norm = np.cumsum(epsilon, axis=1)
    result = trading_signal_OU_process_.trading_signal(time[j])
    kappa = result["kappa"]; mu = result["mu"]; sigma = result["sigma"]
    cummulative_epsilon_end = result["cummulative_epsilon_end"]; R_sq = result["R_sq"]
    signal.append({"time": time[j], "epsilon_idx": epsilon_idx, "cummulative_epsilon_norm": cummulative_epsilon_norm[:, -1],
                   "kappa": kappa, "signal_OU": (cummulative_epsilon_end-mu)/sigma, "R_sq": R_sq})


#%% correlate neural network portfolio weights with OU process (by year)
color_max = 0.01
transparant_threshold = 0.002
transparancy = 0.001
year_start = eval_t_start.year; year_end = eval_t_end.year

plt.figure(figsize=(3*(year_end-year_start+1), 9))
for year in np.arange(year_start, year_end+1, 1):
    plt.subplot(3, year_end-year_start+1, year-year_start+1)
    t_idx = [j for j in range(len(time)) if time[j].year == year]
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(252/signal[j]["kappa"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.001)
        plt.scatter(252/signal[j]["kappa"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 60)); plt.ylim((-5, 5))
    plt.xlabel(r"$\tau$ (days)"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")
    plt.legend(title="{}".format(year), loc="upper right")

    plt.subplot(3, year_end-year_start+1, (year_end-year_start+1)+year-year_start+1)
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(signal[j]["R_sq"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
        plt.scatter(signal[j]["R_sq"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 1)); plt.ylim((-5, 5))
    plt.xlabel(r"$R^2$"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")
    plt.legend(title="{}".format(year), loc="upper right")
    
    plt.subplot(3, year_end-year_start+1, 2*(year_end-year_start+1)+year-year_start+1)
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(signal[j]["R_sq"][zero_idx], 252/signal[j]["kappa"][zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
        plt.scatter(signal[j]["R_sq"][non_zero_idx], 252/signal[j]["kappa"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 1)); plt.ylim((0, 60))
    plt.xlabel(r"$R^2$"); plt.ylabel(r"$\tau$ (days)")
    plt.legend(title="{}".format(year), loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_NN_portfolio_weights_vs_OU_signal_by_year.png".format(factor_name)), dpi=300)

#%% correlate neural network portfolio weights with OU process (all time)
color_max = 0.01
transparant_threshold = 0.002
transparancy = 0.001
year_start = eval_t_start.year; year_end = eval_t_end.year

plt.figure(figsize=(9, 3))
plt.subplot(1,3,1)
for j in range(len(time)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(252/signal[j]["kappa"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(252/signal[j]["kappa"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 60)); plt.ylim((-5, 5))
plt.xlabel(r"$\tau$ (days)"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")

plt.subplot(1,3,2)
for j in range(len(time)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(signal[j]["R_sq"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(signal[j]["R_sq"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 1)); plt.ylim((-5, 5))
plt.xlabel(r"$R^2$"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")

plt.subplot(1,3,3)
for j in range(len(t_idx)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(signal[j]["R_sq"][zero_idx], 252/signal[j]["kappa"][zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(signal[j]["R_sq"][non_zero_idx], 252/signal[j]["kappa"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 1)); plt.ylim((0, 60))
plt.xlabel(r"$R^2$"); plt.ylabel(r"$\tau$ (days)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_NN_portfolio_weights_vs_OU_signal.png".format(factor_name)), dpi=300)

#%%
plt.figure(figsize=(9, 3))
plt.subplot(1,3,1)
plt.xlim((0, 60)); plt.ylim((-5, 5))
plt.xlabel(r"$\tau$ (days)", fontsize=14)
plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$", fontsize=14)

plt.subplot(1,3,2)
plt.xlim((0, 1)); plt.ylim((-5, 5))
plt.xlabel(r"$R^2$", fontsize=14)
plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$", fontsize=14)

plt.subplot(1,3,3)
plt.xlim((0, 1)); plt.ylim((0, 60))
plt.xlabel(r"$R^2$", fontsize=14)
plt.ylabel(r"$\tau$ (days)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_NN_portfolio_weights_vs_OU_signal.pdf".format(factor_name)), dpi=300)

plt.figure(figsize=(4, 0.5))
# create color bar
ar = np.linspace(-color_max, color_max, 1000).reshape(1, -1)
plt.imshow(ar, cmap="RdBu_r", aspect="auto")
plt.xticks([0, ar.shape[1]//2, ar.shape[1]-1], [-color_max, 0, color_max])
plt.yticks([])
plt.title(r"$w_t^{\epsilon}$")
plt.savefig(os.path.join(os.path.dirname(__file__), "color_bar.pdf"), dpi=300)


#%% correlate OU portfolio weights with OU process (by year)
color_max = 0.02
transparant_threshold = 0.002
transparancy = 0.001
year_start = eval_t_start.year; year_end = eval_t_end.year

plt.figure(figsize=(3*(year_end-year_start+1), 9))
for year in np.arange(year_start, year_end+1, 1):
    plt.subplot(3, year_end-year_start+1, year-year_start+1)
    t_idx = [j for j in range(len(time)) if time[j].year == year]
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(252/signal[j]["kappa"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
        plt.scatter(252/signal[j]["kappa"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 60)); plt.ylim((-5, 5))
    plt.xlabel(r"$\tau$ (days)"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")
    plt.legend(title="{}".format(year), loc="upper right")

    plt.subplot(3, year_end-year_start+1, (year_end-year_start+1)+year-year_start+1)
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(signal[j]["R_sq"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
        plt.scatter(signal[j]["R_sq"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 1)); plt.ylim((-5, 5))
    plt.xlabel(r"$R^2$"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")
    plt.legend(title="{}".format(year), loc="upper right")
    
    plt.subplot(3, year_end-year_start+1, 2*(year_end-year_start+1)+year-year_start+1)
    for j in range(len(t_idx)):
        epsilon_idx = signal[j]["epsilon_idx"]
        non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
        zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
        plt.scatter(signal[j]["R_sq"][zero_idx], 252/signal[j]["kappa"][zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
        plt.scatter(signal[j]["R_sq"][non_zero_idx], 252/signal[j]["kappa"][non_zero_idx], s=10,
                    c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
    plt.xlim((0, 1)); plt.ylim((0, 60))
    plt.xlabel(r"$R^2$"); plt.ylabel(r"$\tau$ (days)")
    plt.legend(title="{}".format(year), loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_OU_portfolio_weights_vs_OU_signal_by_year.png".format(factor_name)), dpi=300)

#%% correlate OU portfolio weights with OU process (all time)
color_max = 0.02
transparant_threshold = 0.002
transparancy = 0.001
year_start = eval_t_start.year; year_end = eval_t_end.year

plt.figure(figsize=(9, 3))
plt.subplot(1,3,1)
for j in range(len(time)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(252/signal[j]["kappa"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(252/signal[j]["kappa"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 60)); plt.ylim((-5, 5))
plt.xlabel(r"$\tau$ (days)"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")

plt.subplot(1,3,2)
for j in range(len(time)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(signal[j]["R_sq"][zero_idx], signal[j]["signal_OU"][zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(signal[j]["R_sq"][non_zero_idx], signal[j]["signal_OU"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 1)); plt.ylim((-5, 5))
plt.xlabel(r"$R^2$"); plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$")

plt.subplot(1,3,3)
for j in range(len(t_idx)):
    epsilon_idx = signal[j]["epsilon_idx"]
    non_zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])>=transparant_threshold)[0]
    zero_idx = np.where(np.abs(portfolio_weights_epsilon_OU[epsilon_idx, j])<transparant_threshold)[0]
    plt.scatter(signal[j]["R_sq"][zero_idx], 252/signal[j]["kappa"][zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=transparancy)
    plt.scatter(signal[j]["R_sq"][non_zero_idx], 252/signal[j]["kappa"][non_zero_idx], s=10,
                c=portfolio_weights_epsilon_OU[epsilon_idx, j][non_zero_idx], cmap="RdBu_r", vmin=-color_max, vmax=color_max, alpha=0.5)
plt.xlim((0, 1)); plt.ylim((0, 60))
plt.xlabel(r"$R^2$"); plt.ylabel(r"$\tau$ (days)")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_OU_portfolio_weights_vs_OU_signal.png".format(factor_name)), dpi=300)

#%%
plt.figure(figsize=(9, 3))
plt.subplot(1,3,1)
plt.xlim((0, 60)); plt.ylim((-5, 5))
plt.xlabel(r"$\tau$ (days)", fontsize=14)
plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$", fontsize=14)
plt.tick_params(direction='in')

plt.subplot(1,3,2)
plt.xlim((0, 1)); plt.ylim((-5, 5))
plt.xlabel(r"$R^2$", fontsize=14)
plt.ylabel(r"$\frac{x_t-\mu}{\sigma}$", fontsize=14)

plt.subplot(1,3,3)
plt.xlim((0, 1)); plt.ylim((0, 60))
plt.xlabel(r"$R^2$", fontsize=14)
plt.ylabel(r"$\tau$ (days)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_OU_portfolio_weights_vs_OU_signal.pdf".format(factor_name)), dpi=300)

#%%
plt.figure(figsize=(4, 0.5))
# create color bar
ar = np.linspace(-color_max, color_max, 1000).reshape(1, -1)
plt.imshow(ar, cmap="RdBu_r", aspect="auto", alpha=0.5)
plt.xticks([0, ar.shape[1]//2, ar.shape[1]-1], [-color_max, 0, color_max])
plt.yticks([])
plt.title(r"$w_t^{\epsilon}$", fontsize=14)
plt.savefig(os.path.join(os.path.dirname(__file__), "color_bar.pdf"), dpi=300)


#%%





# %%
