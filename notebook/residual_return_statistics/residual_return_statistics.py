#%%
import os, sys, copy, h5py, datetime, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import utils.utils as utils

#%%
rank_max = 499
#PCA_TYPE = "name"
PCA_TYPE = "rank_permutation"
#PCA_TYPE = "rank_hybrid_Atlas"
#PCA_TYPE = "rank_hybrid_Atlas_high_freq"
#PCA_TYPE = "rank_theta_transform"

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    t_eval_start = datetime.datetime(2005,7,1); t_eval_end = datetime.datetime(2022,12,15)
else:
    t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)

equity_idx_PERMNO_ticker = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_idx_PERMNO_ticker.csv"))

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
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    time_tick_label = ["2005/7/1", "2010/1/1", "2015/1/1", "2020/1/1", "2022/12/15"]
    time_tick_idx = [np.searchsorted(equity_data_high_freq_.time_axis_daily, datetime.datetime.strptime(t, "%Y/%m/%d")) for t in time_tick_label]
    time_tick_idx = np.array(time_tick_idx) - time_tick_idx[0]
else:
    time_tick_label = ["1991/1/1", "1996/1/1", "2001/1/1", "2006/1/1", "2011/1/1", "2016/1/1", "2022/12/15"]
    time_tick_idx = [np.searchsorted(equity_data_.time_axis, datetime.datetime.strptime(t, "%Y/%m/%d")) for t in time_tick_label]
    time_tick_idx = np.array(time_tick_idx) - time_tick_idx[0]

color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
match PCA_TYPE:
    case "name":
        plot_color = color_cycle[0]
    case "rank_permutation":
        plot_color = color_cycle[2]
    case "rank_hybrid_Atlas":
        plot_color = color_cycle[2]
    case "rank_hybrid_Atlas_high_freq":
        plot_color = color_cycle[3]
    case "rank_theta_transform":
        plot_color = color_cycle[4]

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    three_year_interval_list = [(datetime.datetime(2005, 7, 1), datetime.datetime(2008, 12, 31)),
                           (datetime.datetime(2009, 1, 1), datetime.datetime(2011, 12, 31)),
                            (datetime.datetime(2012, 1, 1), datetime.datetime(2014, 12, 31)),
                            (datetime.datetime(2015, 1, 1), datetime.datetime(2017, 12, 31)),
                            (datetime.datetime(2018, 1, 1), datetime.datetime(2020, 12, 31)),
                            (datetime.datetime(2021, 1, 1), datetime.datetime(2022, 12, 1))]
    three_year_interval_str = ["20050701_20081231", "20090101_20111231", "20120101_20141231", "20150101_20171231", "20180101_20201231", "20210101_20221201"]
    three_year_interval_legend_str = ["2005-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020", "2021-2022"]

else:
    five_year_interval_list = [(datetime.datetime(1991, 1, 1), datetime.datetime(1995, 12, 31)),
                        (datetime.datetime(1996, 1, 1), datetime.datetime(2000, 12, 31)),
                        (datetime.datetime(2001, 1, 1), datetime.datetime(2005, 12, 31)),
                        (datetime.datetime(2006, 1, 1), datetime.datetime(2010, 12, 31)),
                        (datetime.datetime(2011, 1, 1), datetime.datetime(2015, 12, 31)),
                        (datetime.datetime(2016, 1, 1), datetime.datetime(2022, 12, 1))]
    five_year_interval_list_str = ["19910101_19951231", "19960101_20001231", "20010101_20051231", "20060101_20101231", "20110101_20151231", "20160101_20221231"]
    five_year_interval_legend_str = ["1991-1995", "1996-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2022"]

PCA_factor_config = {"factor_evaluation_window_length": 252,
                "loading_evaluation_window_length": 60, 
                "residual_return_evaluation_window_length": 60,
                "rank_min": 0,
                "rank_max": 999,
                "factor_number": 3,
                "type": PCA_TYPE,
                "max_cache_len": 500}

PCA_factor_ = factor.PCA_factor(equity_data_, equity_data_high_freq_, PCA_factor_config)
_ = PCA_factor_._initialize_residual_return_all()

trading_signal_OU_process_config = {"max_cache_len": 100}
trading_signal_OU_ = trading_signal.trading_signal_OU_process(equity_data_, equity_data_high_freq_, PCA_factor_, trading_signal_OU_process_config)

portfolio_performance_config = {"transaction_cost_factor": 0.0005,
                                "shorting_cost_factor": 0.0001}
portfolio_performance_ = utils.portfolio_performance(equity_data_, equity_data_high_freq_, portfolio_performance_config)

#%% distribution in capitalization
plt.figure(figsize=(9, 4))
for k, t_interval in enumerate(five_year_interval_list):
    t_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_interval[0]), np.searchsorted(equity_data_.time_axis, t_interval[1])+1, 1)
    capitalization = np.zeros((500, len(t_idx))); capitalization[:] = np.nan
    for j in range(len(t_idx)):
        t = equity_data_.time_axis[t_idx[j]]
        equity_idx = equity_data_.equity_idx_by_rank[0:500, t_idx[j]].astype(int)
        capitalization[:, j] = equity_data_.capitalization[equity_idx, t_idx[j]]
    avg = np.nanmean(capitalization, axis=1)
    plt.plot(np.arange(1, 501, 1), avg/np.sum(avg), label=five_year_interval_legend_str[k])

plt.xscale("log"); plt.yscale("log")
plt.xlabel("rank"); plt.ylabel("fraction of market capitalization")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "capitalization_distribution.pdf"), dpi=300)


#%% sample residual return trajectory
if PCA_TYPE == "name":
    t_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end)+1, 1)
    time = [t for t in equity_data_.time_axis if t>=t_eval_start and t<=t_eval_end]
    epsilon_all = PCA_factor_.epsilon_all[:, t_idx]
    epsilon_idx = np.arange(0, len(equity_data_.equity_idx_list), 1)
    capitalization_max = np.nanmax(equity_data_.capitalization, axis=1)
    idx_sort = np.argsort(capitalization_max)[::-1][0:rank_max]
    plt.figure(figsize=(18, 10))
    plt.suptitle("residual return trajectory (PCA name, cumulative all time)")
    for j in np.arange(0, len(idx_sort), len(idx_sort)//10):
        plt.subplot(5, 2, j//(len(idx_sort)//10)+1)
        ticker = equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==epsilon_idx[idx_sort[j]]].iloc[-1, 2]
        plt.plot(time, np.nancumsum(epsilon_all[idx_sort[j], :]), label="ticker: {}, rank: {}".format(ticker, j))
        ticker = equity_idx_PERMNO_ticker[equity_idx_PERMNO_ticker["equity_idx"]==epsilon_idx[idx_sort[j+len(idx_sort)//20]]].iloc[-1, 2]
        plt.plot(time, np.nancumsum(epsilon_all[idx_sort[j+len(idx_sort)//20], :]), label="ticker: {}, rank: {}".format(ticker, j+len(idx_sort)//20))
        plt.hlines(0, np.min(time), np.max(time), color='red', linestyle='--', alpha=0.5)
        plt.legend(loc="lower left")
        plt.xlim(datetime.datetime(1991,1,1), datetime.datetime(2022,12,31))
    plt.tight_layout()

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    t_idx = np.arange(np.searchsorted(equity_data_high_freq_.time_axis_daily, t_eval_start), np.searchsorted(equity_data_high_freq_.time_axis_daily, t_eval_end)+1, 1)
    time = [t for t in equity_data_high_freq_.time_axis_daily if t>=t_eval_start and t<=t_eval_end]
    epsilon_all = PCA_factor_.epsilon_all[:, t_idx]
    epsilon_idx = np.arange(0, rank_max, 1)
    plt.figure(figsize=(18, 10))
    plt.suptitle("residual return trajectory (PCA {}, cumulative all time)".format(PCA_TYPE))
    for j in np.arange(0, rank_max, rank_max//10):
        plt.subplot(5, 2, j//(rank_max//10)+1)
        plt.plot(time, np.nancumsum(epsilon_all[j, :]), label="rank: {}".format(j))
        plt.plot(time, np.nancumsum(epsilon_all[j+rank_max//20, :]), label="rank: {}".format(j+rank_max//20))
        plt.legend(loc="lower left")
        plt.xlim(datetime.datetime(1991,1,1), datetime.datetime(2022,12,31))
    plt.tight_layout()

if PCA_TYPE in ["rank_permutation", "rank_hybrid_Atlas", "rank_theta_transform"]:
    t_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end)+1, 1)
    time = [t for t in equity_data_.time_axis if t>=t_eval_start and t<=t_eval_end]
    epsilon_all = PCA_factor_.epsilon_all[:, t_idx]
    epsilon_idx = np.arange(0, rank_max, 1)
    plt.figure(figsize=(18, 10))
    plt.suptitle("residual return trajectory (PCA {}, cumulative all time)".format(PCA_TYPE))
    for j in np.arange(0, rank_max, rank_max//10):
        plt.subplot(5, 2, j//(rank_max//10)+1)
        plt.plot(time, np.nancumsum(epsilon_all[j, :]), label="rank: {}".format(j))
        plt.plot(time, np.nancumsum(epsilon_all[j+rank_max//20, :]), label="rank: {}".format(j+rank_max//20))
        plt.legend(loc="lower left")
        plt.xlim(datetime.datetime(1991,1,1), datetime.datetime(2022,12,31))
    plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(__file__), "".join(["residual_return_trajectory_sample_", PCA_TYPE, ".png"])), dpi=300)

#%% residual return trajectory (cumulative all time)
plt.figure(figsize = (9, 5))
if PCA_TYPE == "name":
    plt.title("residual return trajectory (PCA name, cumulative all time)")
    plt.imshow(np.nancumsum(epsilon_all, axis=1), vmin=-0.1, vmax=0.1, aspect='auto', cmap="RdBu_r")
    plt.ylabel("epsilon index (sort by average-capitalization)"); plt.xlabel("time")
    plt.xticks(ticks=time_tick_idx, labels=time_tick_label); plt.colorbar(); plt.tight_layout()

if PCA_TYPE in ["rank_hybrid_Atlas", "rank_hybrid_Atlas_high_freq", "rank_permutation", "rank_theta_transform"]:
    plt.imshow(np.nancumsum(epsilon_all[0:rank_max], axis=1), vmin=-0.5, vmax=0.5, aspect='auto', cmap="RdBu_r")
    plt.ylabel("epsilon index"); plt.xlabel("time")
    plt.xticks(ticks=time_tick_idx, labels=time_tick_label); plt.colorbar(); plt.tight_layout()
    plt.title("residual return trajectory (PCA {}, cumulative all time)".format(PCA_TYPE))

plt.savefig(os.path.join(os.path.dirname(__file__), "".join(["residual_return_trajectory_", PCA_TYPE, ".png"])), dpi=300)


#%% stability of residual return
# check if the residual return epsilon_t evaluated at time t to t+residual_return_evaluation_window_length is stable
'''
relative_volatility = []
time = [t for t in equity_data_.time_axis if t>=t_eval_start and t<=t_eval_end]
epsilon_hist = []
t = np.random.choice(time, 1)[0]
t_idx = np.searchsorted(equity_data_.time_axis, t)
result = PCA_factor_.residual_return(t)
epsilon_idx = np.random.choice(result["epsilon_idx"], 1)[0]
idx = np.searchsorted(result["epsilon_idx"], epsilon_idx)
epsilon = result["epsilon"][idx, -1]
epsilon_hist.append(epsilon)
print("time: {}, epsilon_idx: {}, epsilon: {}".format(t, epsilon_idx, epsilon))
for j in np.arange(1, PCA_factor_config["residual_return_evaluation_window_length"]):
    if t_idx+j >= len(equity_data_.time_axis):
        break
    result = PCA_factor_.residual_return(equity_data_.time_axis[t_idx+j])
    idx = np.searchsorted(result["epsilon_idx"], epsilon_idx)
    epsilon = result["epsilon"][idx, -(j+1)]
    print("time: {}, epsilon_idx: {}, epsilon: {}".format(equity_data_.time_axis[t_idx+j-j], epsilon_idx, epsilon))
    epsilon_hist.append(epsilon)

plt.plot(range(PCA_factor_config["residual_return_evaluation_window_length"]), epsilon_hist[::-1])
relative_volatility.append(np.std(epsilon_hist)/np.mean(np.abs(epsilon_hist)))
print("relative volatility: {:.4f}".format(np.std(epsilon_hist)/np.mean(np.abs(epsilon_hist))))

plt.figure(figsize=(10, 5))
plt.hist(relative_volatility, bins=200, density=True, range=(0, 1))
plt.yscale("log")
plt.xlabel("relative volatility"); plt.ylabel("probability density")
'''

#%% eigenvalue dynamics
time_axis = PCA_factor_.time_axis; save_file_name = PCA_factor_.save_file_name
time_hist = []; eval_hist = []
for j in tqdm.tqdm(range(len(time_axis))):
    if time_axis[j] >= t_eval_start and time_axis[j] <= t_eval_end:
        time_str = time_axis[j].strftime("%Y%m%d")
        result = np.load(save_file_name.replace("timestr", time_str))
        evals = result["eigenvalues"]
        time_hist.append(time_axis[j])
        eval_hist.append(list(evals/np.sum(evals)))
eval_hist = np.array(eval_hist).T

plt.figure(figsize=(9,5))
for j in range(5):
    plt.plot(time_hist, eval_hist[j,:], label=r"$\lambda_{}$".format(j+1))
plt.xlabel("time"); plt.ylabel(r"$\lambda/||\lambda||_1$"); plt.tight_layout(); plt.ylim(0, 1)
plt.title("PCA {}: eigenvalue dynamics".format(PCA_TYPE))
plt.legend(title=PCA_TYPE)
plt.savefig(os.path.join(os.path.dirname(__file__), "eigenvalue_dynamics_{}.pdf".format(PCA_TYPE)), dpi=300)

#%% cummulative residual return versus Brownian motion
def cummulative_residual_return_BMref(t_eval_start, t_eval_end):
    t_idx = np.arange(np.searchsorted(PCA_factor_.time_axis, t_eval_start), np.searchsorted(PCA_factor_.time_axis, t_eval_end)+1, 1)
    ref_all = []
    for j in tqdm.tqdm(t_idx):
        t = PCA_factor_.time_axis[j]
        epsilon = PCA_factor_.residual_return(t)["epsilon"]
        sigma = np.std(epsilon, ddof=1, axis=1, keepdims=True)
        epsilon = epsilon/sigma
        cumsum = np.cumsum(epsilon, axis=1)/np.sqrt(np.arange(1, epsilon.shape[1]+1))
        ref_all.extend([cumsum[k,:] for k in range(epsilon.shape[0])])

    ref_all = np.vstack(ref_all)
    #bin_edge_min = np.quantile(ref_all.flatten(), 0.01)
    #bin_edge_max = np.quantile(ref_all.flatten(), 0.99)
    bin_edge_min = -3; bin_edge_max = 3
    bin_edge = np.histogram(ref_all[:, 0], bins=200, range=(bin_edge_min, bin_edge_max), density=True)[1]
    bin_center = (bin_edge[0:-1] + bin_edge[1:])/2
    pdf = [np.histogram(ref_all[:, j], bins=200, range=(bin_edge_min, bin_edge_max), density=True)[0] for j in range(ref_all.shape[1])]
    pdf = np.vstack(pdf)
    pdf_normal = np.exp(-np.power(bin_center,2)/2)/np.sqrt(2*np.pi)
    print("pdf: ", pdf.shape, "pdf_normal: ", pdf_normal.shape)

    return {"bin_center": bin_center, "pdf": pdf, "pdf_normal": pdf_normal}

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    plt.figure(figsize=(18, 3))
    for j in range(len(three_year_interval_list)):
    #for j in [0]:
        plt.subplot(1, len(three_year_interval_list),j+1)
        result = cummulative_residual_return_BMref(three_year_interval_list[j][0], three_year_interval_list[j][1])
        plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
        #plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r")
        plt.ylabel("time (days)")
        plt.xlabel("prob. dens. dist.: {}".format(r"$\frac{\epsilon_t^L}{\sigma\sqrt{t}}-\mathcal{N}(0,1)$"))
        plt.yticks(ticks=np.arange(0, result["pdf"].shape[0]-1, 10), labels=np.arange(0, result["pdf"].shape[0]-1, 10))
        plt.xticks(ticks=[np.searchsorted(result["bin_center"], j) for j in np.arange(-3, 4, 1)], labels=np.arange(-3, 4, 1))
        if j == len(three_year_interval_list)-1: plt.colorbar()
        plt.ylim(0, result["pdf"].shape[0]-1)
        plt.title("{}-{}".format(three_year_interval_list[j][0].strftime("%Y/%m/%d"), three_year_interval_list[j][1].strftime("%Y/%m/%d")))
else:
    plt.figure(figsize=(18, 3))
    for j in range(len(five_year_interval_list)):
    #for j in [0]:
        plt.subplot(1, len(five_year_interval_list),j+1)
        result = cummulative_residual_return_BMref(five_year_interval_list[j][0], five_year_interval_list[j][1])
        plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
        #plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r")
        plt.ylabel("time (days)")
        plt.xlabel("prob. dens. dist.: {}".format(r"$\frac{\epsilon_t^L}{\sigma\sqrt{t}}-\mathcal{N}(0,1)$"))
        plt.yticks(ticks=np.arange(0, result["pdf"].shape[0]-1, 10), labels=np.arange(0, result["pdf"].shape[0]-1, 10))
        plt.xticks(ticks=[np.searchsorted(result["bin_center"], j) for j in np.arange(-3, 4, 1)], labels=np.arange(-3, 4, 1))
        #plt.tick_params(direction='in', axis='both', which='both', bottom=True, top=True, left=True, right=True)
        #if j == len(five_year_interval_list)-1: plt.colorbar()
        plt.ylim(0, result["pdf"].shape[0]-1)
        #plt.title("{}-{}".format(five_year_interval_list[j][0].strftime("%Y/%m/%d"), five_year_interval_list[j][1].strftime("%Y/%m/%d")))

#plt.suptitle("PCA {}: cummulative residual return versus Brownian motion".format(PCA_TYPE))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "cummulative_residual_return_std_norm_BM_compare_{}.png".format(PCA_TYPE)), dpi=300)
plt.savefig(os.path.join(os.path.dirname(__file__), "cummulative_residual_return_std_norm_BM_compare_{}.pdf".format(PCA_TYPE)), dpi=300)


#%% mean reverting time (counting the number of days to reach a certain threshold)
def mean_reverting_local_time(PCAf):
    tau_hist = []
    time_axis = PCAf.time_axis
    t_idx = np.arange(np.searchsorted(time_axis, t_eval_start), np.searchsorted(time_axis, t_eval_end)+1, 1)
    for j in tqdm.tqdm(t_idx):
        t = time_axis[j]
        epsilon_hist = PCAf.residual_return(t)["epsilon"]
        epsilon_hist_cum = np.cumsum(epsilon_hist, axis=1)
        tau = np.zeros(epsilon_hist.shape[1]+1)
        for k in range(epsilon_hist_cum.shape[0]):
            count = 0
            threshold = np.std(epsilon_hist[k, :])
            for l in range(epsilon_hist_cum.shape[1]):
                if np.abs(epsilon_hist_cum[k,l]) < threshold:
                    count += 1
            tau[count] += 1
        tau_hist.append(list(tau/np.sum(tau)))
    return ([time_axis[j] for j in t_idx], np.array(tau_hist).T)

time_axis, tau_hist = mean_reverting_local_time(PCA_factor_)
plt.figure(figsize=(18, 4*len(five_year_interval_list)))
for j in range(len(five_year_interval_list)):
    plt.subplot(len(five_year_interval_list)//2, 2, j+1)
    t_idx = np.arange(np.searchsorted(time_axis, five_year_interval_list[j][0]), np.searchsorted(time_axis, five_year_interval_list[j][1])+1, 1)
    plt.plot(np.arange(0, tau_hist.shape[0], 1), np.mean(tau_hist[:, t_idx], axis=1), label="{}-{}".format(five_year_interval_list[j][0].strftime("%Y/%m/%d"), five_year_interval_list[j][1].strftime("%Y/%m/%d")), color=plot_color)
    plt.fill_between(np.arange(0, tau_hist.shape[0], 1), np.mean(tau_hist[:, t_idx], axis=1)-np.std(tau_hist[:, t_idx], axis=1), np.mean(tau_hist[:, t_idx], axis=1)+np.std(tau_hist[:, t_idx], axis=1), alpha=0.3, color=plot_color)
    plt.xlabel("local time"); plt.ylabel("probability density")
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "mean_reverting_local_time_hist_{}.pdf".format(PCA_TYPE)), dpi=300)


#%% mean reverting time (fit from OU process) - probability density function of tau across different time
def OU_process_rate(trading_signal, t_start, t_end, R2_threshold=0.7):
    time_axis = trading_signal.time_axis
    t_idx = np.arange(np.searchsorted(time_axis, t_start), np.searchsorted(time_axis, t_end)+1, 1)
    T_max = PCA_factor_config["residual_return_evaluation_window_length"]; T_min = 1
    time_hist = []; kappa_hist = []; kappa_flatten = []
    tau_hist = []; tau_flatten = []
    for j in tqdm.tqdm(t_idx):
        time = time_axis[j]
        if time >= t_eval_start and time <= t_eval_end:
            result = trading_signal.trading_signal(time)
            idx_valid = [k for k in range(len(result["R_sq"])) if result["R_sq"][k] > R2_threshold]
            kappa_hist.append(list(np.array([result["kappa"][k] for k in idx_valid])/252))
            kappa_flatten.extend(list(np.array([result["kappa"][k] for k in idx_valid])/252))
            tau_hist.append(list(252/np.array([result["kappa"][k] for k in idx_valid])))
            tau_flatten.extend(list(252/np.array([result["kappa"][k] for k in idx_valid])))
            time_hist.append(time)

    kappa_max = np.quantile(kappa_flatten, 0.95)
    kappa_min = np.quantile(kappa_flatten, 0.05)
    kappa_bin_edge = np.histogram(kappa_hist[0], bins=200, range=(kappa_min, kappa_max), density=True)[1]
    kappa_bin_center = (kappa_bin_edge[0:-1] + kappa_bin_edge[1:])/2
    kappa_pdf = [list(np.histogram(np.array(k), bins=200, range=(kappa_min, kappa_max), density=True)[0]) for k in kappa_hist]

    #tau_max = np.quantile(tau_flatten, 0.99)
    #tau_min = np.quantile(tau_flatten, 0.01)
    tau_min = T_min; tau_max = T_max
    tau_bin_edge = np.histogram(tau_hist[0], bins=PCA_factor_config["residual_return_evaluation_window_length"], range=(1, PCA_factor_config["residual_return_evaluation_window_length"]), density=True)[1]
    tau_bin_center = (tau_bin_edge[0:-1] + tau_bin_edge[1:])/2
    tau_pdf = [list(np.histogram(np.array(k), bins=PCA_factor_config["residual_return_evaluation_window_length"], range=(1, PCA_factor_config["residual_return_evaluation_window_length"]), density=True)[0]) for k in tau_hist]

    return {"time": time_hist, "T_min": T_min, "T_max": T_max,
            "kappa": (kappa_bin_center, np.array(kappa_pdf).T, kappa_hist, kappa_flatten),
            "tau": (tau_bin_center, np.array(tau_pdf).T, tau_hist, tau_flatten)}

result = OU_process_rate(trading_signal_OU_, t_eval_start, t_eval_end, R2_threshold=0.7)
time_hist = result["time"]; kappa_bin_center, kappa_pdf, kappa_hist, kappa_flatten = result["kappa"]
tau_bin_center, tau_pdf, tau_hist, tau_flatten = result["tau"]

plt.figure(figsize=(18, 5))
plt.imshow(tau_pdf, cmap='gray_r', aspect='auto')
plt.colorbar()
plt.xlabel("time"); plt.ylabel("tau (days)")
plt.xticks(ticks=time_tick_idx, labels=time_tick_label)
plt.yticks(ticks=[0, tau_pdf.shape[0]-1], labels=[0, PCA_factor_config["residual_return_evaluation_window_length"]])
#plt.title("{}: probability density function of tau by OU process".format(PCA_TYPE))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "OU_process_tau_pdf_{}.png").format(PCA_TYPE), dpi=300)

#%% mean reverting time (fit from OU process) - histogram of tau
plot_color = color_cycle[2]
log = []
plt.figure(figsize=(18, 3))
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    for j in range(len(three_year_interval_list)):
        result = OU_process_rate(trading_signal_OU_, three_year_interval_list[j][0], three_year_interval_list[j][1], R2_threshold=0.7)
        time_hist = result["time"]
        kappa_bin_center, kappa_pdf, kappa_hist, kappa_flatten = result["kappa"]
        tau_bin_center, tau_pdf, tau_hist, tau_flatten = result["tau"]

        plt.subplot(1, len(five_year_interval_list), j+1)
        hist, bin_edges = np.histogram(tau_flatten, bins=PCA_factor_config["residual_return_evaluation_window_length"], range=(1, PCA_factor_config["residual_return_evaluation_window_length"]), density=True)
        bin_center = (bin_edges[0:-1] + bin_edges[1:])/2
        plt.plot(bin_center, hist, label=three_year_interval_legend_str[j], color=plot_color)
        plt.vlines(bin_center[np.argmax(hist)], 0, np.max(hist), color=plot_color, linestyle="--", label="peak: {:.2f}".format(bin_center[np.argmax(hist)]))
        plt.fill_between(bin_center, 0, hist, alpha=0.3, color=plot_color)
        plt.xlabel("tau (days)"); plt.ylabel("probability density")
        plt.ylim(0, 0.2); plt.xlim(0, PCA_factor_config["residual_return_evaluation_window_length"])
        plt.legend()
        log.append([hist, bin_center])
else:
    for j in range(len(five_year_interval_list)):
        result = OU_process_rate(trading_signal_OU_, five_year_interval_list[j][0], five_year_interval_list[j][1], R2_threshold=0.7)
        time_hist = result["time"]
        kappa_bin_center, kappa_pdf, kappa_hist, kappa_flatten = result["kappa"]
        tau_bin_center, tau_pdf, tau_hist, tau_flatten = result["tau"]

        plt.subplot(1, len(five_year_interval_list), j+1)
        hist, bin_edges = np.histogram(tau_flatten, bins=PCA_factor_config["residual_return_evaluation_window_length"], range=(1, PCA_factor_config["residual_return_evaluation_window_length"]), density=True)
        bin_center = (bin_edges[0:-1] + bin_edges[1:])/2
        plt.plot(bin_center, hist, label=five_year_interval_legend_str[j], color=plot_color)
        #plt.vlines(bin_center[np.argmax(hist)], 0, np.max(hist), color=plot_color, linestyle="--", label="peak: {:.2f}".format(bin_center[np.argmax(hist)]))
        idx = np.where(bin_center>=30)[0]
        plt.fill_between(bin_center[idx], 0, hist[idx], alpha=0.3, color=plot_color)
        plt.xlabel("tau (days)"); plt.ylabel("probability density")
        plt.ylim(1e-3, 0.5); plt.xlim(0, PCA_factor_config["residual_return_evaluation_window_length"])
        plt.yscale("log")
        plt.legend()
        log.append([hist, bin_center])
#plt.suptitle(r"{}: prob. dens. dist. of $\tau$ by OU process".format(PCA_TYPE))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "OU_process_tau_hist_{}.pdf".format(PCA_TYPE)), dpi=300)


#%% minimum value of tau
t_idx = np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end)+1, 1)
tau_min = []
for j in tqdm.tqdm(t_idx):
    result = trading_signal_OU_.trading_signal(equity_data_.time_axis[j])
    kappa = np.array(result["kappa"]); R_sq = np.array(result["R_sq"])
    valid_idx = np.where(R_sq > 0.7)[0]
    kappa = kappa[valid_idx]
    tau_min.append(np.nanmin(252/kappa))

plt.figure(figsize=(9, 3))
plt.plot([equity_data_.time_axis[j] for j in t_idx], tau_min)
plt.ylabel("tau_min (days)"); plt.xlabel("time")
plt.hlines(3, equity_data_.time_axis[t_idx[0]], equity_data_.time_axis[t_idx[-1]], color='red', linestyle='--', alpha=0.5)

#%% statistics of gap process
rank_min = 0; rank_max = 499
local_time_interval_distribution = []

if os.path.exists(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.npz")):
    result = np.load(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.npz"))
    local_time_interval_distribution = result["local_time_interval_distribution"]
    bin_center = result["bin_center"]
else:
    #t_eval = datetime.datetime(2022, 12, 1)
    for t_eval in tqdm.tqdm(equity_data_high_freq_.time_axis_daily):
        result = equity_data_high_freq_.intraday_data(t_eval)
        time = result["time"]; capitalization = result["capitalization"]; rank = result["rank"]; equity_idx_by_rank = result["equity_idx_by_rank"]
        diff = np.diff(equity_idx_by_rank[rank_min:(rank_max+1), :].astype(int), axis=1)
        local_time_interval = []
        for j in range(rank_max+1):
            index = np.where(np.abs(diff[j, :])>0)[0]
            local_time_interval.extend(list(np.diff(index)))
        hist, bin_edges = np.histogram(local_time_interval, bins=len(time)+1, density=True, range=(0, len(time)))
        bin_center = 5*((bin_edges[0:-1] + bin_edges[1:])/2)
        local_time_interval_distribution.append(hist)
    local_time_interval_distribution = np.array(local_time_interval_distribution).T
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.npz"), local_time_interval_distribution=local_time_interval_distribution, bin_center=bin_center)

plt.figure(figsize=(9, 5))
plt.imshow(local_time_interval_distribution, aspect='auto', cmap='RdBu_r')
plt.colorbar()
time_tick_idx = np.arange(0, len(equity_data_high_freq_.time_axis_daily), len(equity_data_high_freq_.time_axis_daily)//5)
time_tick_label = [equity_data_high_freq_.time_axis_daily[j].strftime("%Y/%m/%d") for j in time_tick_idx]
plt.xticks(ticks=time_tick_idx, labels=time_tick_label)
plt.yticks(ticks=np.arange(0, bin_center.shape[0], 12), labels=["{} min".format((5*j).astype(int)) for j in np.arange(0, bin_center.shape[0], 12)])
plt.savefig(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.png"), dpi=300)

year_start = equity_data_high_freq_.time_axis_daily[0].year; year_end = equity_data_high_freq_.time_axis_daily[-1].year
plt.figure(figsize=(18,18))
year_list = np.arange(year_start, year_end+1, 2)
for j in range(len(year_list)):
    year = year_list[j]
    t_idx = [j for j in range(len(equity_data_high_freq_.time_axis_daily)) if equity_data_high_freq_.time_axis_daily[j].year >= year and equity_data_high_freq_.time_axis_daily[j].year < year+5]
    plt.subplot(len(year_list), 2, j+1)
    plt.plot(bin_center, np.mean(local_time_interval_distribution[:, t_idx], axis=1), label="{}-{}".format(year, year+2))
    plt.fill_between(bin_center, np.mean(local_time_interval_distribution[:, t_idx], axis=1)-np.std(local_time_interval_distribution[:, t_idx], axis=1), np.mean(local_time_interval_distribution[:, t_idx], axis=1)+np.std(local_time_interval_distribution[:, t_idx], axis=1), alpha=0.3)
    plt.vlines(45, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='red', linestyle='--', alpha=0.5, label="45 min")

    segment = [5, 25]
    idx = [j for j in range(len(bin_center)) if bin_center[j] >= segment[0] and bin_center[j] <= segment[1]]
    X = bin_center[idx].reshape(-1, 1); Y = np.log(np.mean(local_time_interval_distribution[idx, :][:, t_idx], axis=1))
    reg = LinearRegression().fit(X, Y)
    X_continuous = np.linspace(segment[0], 45, 100).reshape(-1, 1)
    Y_continuous = np.exp(reg.predict(X_continuous))
    plt.plot(X_continuous, Y_continuous, linestyle="--", color="black", alpha=0.5)

    segment = [100, 500]
    idx = [j for j in range(len(bin_center)) if bin_center[j] >= segment[0] and bin_center[j] <= segment[1]]
    X = bin_center[idx].reshape(-1, 1); Y = np.log(np.mean(local_time_interval_distribution[idx, :][:, t_idx], axis=1))
    reg = LinearRegression().fit(X, Y)
    X_continuous = np.linspace(45, segment[1], 100).reshape(-1, 1)
    Y_continuous = np.exp(reg.predict(X_continuous))
    plt.plot(X_continuous, Y_continuous, linestyle="--", color="black", alpha=0.5)

    plt.xlabel("time (min)"); plt.ylabel("probability density") 
    plt.xlim(0, 800)
    plt.yscale("log")
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution_by_year.pdf"), dpi=300)

#%% correlation betweeen kappa and R_sq
t_eval_start = datetime.datetime(1991, 1, 1); t_eval_end = datetime.datetime(2022, 12, 15)
tau_hist = []; R_sq_hist = []
for t_idx in tqdm.tqdm(np.arange(np.searchsorted(equity_data_.time_axis, t_eval_start), np.searchsorted(equity_data_.time_axis, t_eval_end)+1, 1)):
    result = trading_signal_OU_.trading_signal(equity_data_.time_axis[t_idx])
    kappa = np.array(result["kappa"]); R_sq = np.array(result["R_sq"])
    valid_idx = np.where(~np.isnan(kappa))[0]
    kappa = kappa[valid_idx]; R_sq = R_sq[valid_idx]
    tau_hist.extend(list(252/kappa)); R_sq_hist.extend(list(R_sq))

tau_hist = np.array(tau_hist); R_sq_hist = np.array(R_sq_hist)
plt.figure(figsize=(9, 9))
idx = np.where(np.logical_and(tau_hist < 30, R_sq_hist > 0.7))[0]
plt.scatter(tau_hist[idx], R_sq_hist[idx], s=0.5, label = "prop={:.2f}".format(len(idx)/len(tau_hist)))
idx = np.where(np.logical_and(tau_hist < 30, R_sq_hist < 0.7))[0]
plt.scatter(tau_hist[idx], R_sq_hist[idx], s=0.5, label = "prop={:.2f}".format(len(idx)/len(tau_hist)))
idx = np.where(np.logical_and(tau_hist > 30, R_sq_hist > 0.7))[0]
plt.scatter(tau_hist[idx], R_sq_hist[idx], s=0.5, label = "prop={:.2f}".format(len(idx)/len(tau_hist)))
idx = np.where(np.logical_and(tau_hist > 30, R_sq_hist < 0.7))[0]
plt.scatter(tau_hist[idx], R_sq_hist[idx], s=0.5, label = "prop={:.2f}".format(len(idx)/len(tau_hist)))            
plt.ylabel("R_sq"); plt.xlabel("tau (days)")
plt.yscale("log"); plt.xscale("log")
plt.xlim(1, 1000); plt.ylim(0.1, 1.5)
plt.legend()
plt.vlines(30, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='black', linestyle='--', alpha=0.5)
plt.hlines(0.7, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], color='black', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(os.path.dirname(__file__), "tau_R_sq_correlation.png"), dpi=300)


# %%

