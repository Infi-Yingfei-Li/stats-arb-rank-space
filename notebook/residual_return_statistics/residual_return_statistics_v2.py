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
PCA_TYPE = "rank_hybrid_Atlas"
#PCA_TYPE = "rank_hybrid_Atlas_high_freq"

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
    case "rank_hybrid_Atlas":
        plot_color = color_cycle[1]
    case "rank_hybrid_Atlas_high_freq":
        plot_color = color_cycle[2]

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
                "rank_max": 499,
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

#%% distribution of capitalization
plt.figure(figsize=(9, 3))
for j in range(len(five_year_interval_list)):
    start_time, end_time = five_year_interval_list[j][0], five_year_interval_list[j][1]
    t_idx = np.arange(np.searchsorted(equity_data_.time_axis, start_time), np.searchsorted(equity_data_.time_axis, end_time)+1, 1)
    capitalization = np.zeros((1000, len(t_idx))); capitalization[:] = np.nan
    for k in t_idx:
        idx = equity_data_.equity_idx_by_rank[0:1000, k].astype(int)
        capitalization[:, k-t_idx[0]] = equity_data_.capitalization[idx, k]
    avg = np.nanmean(capitalization, axis=1)
    plt.plot(np.arange(1, 1001, 1), avg/np.sum(avg), label=five_year_interval_legend_str[j])

plt.ylim(1e-4, 1e-1)
plt.xlabel("rank in capitalization"); plt.ylabel("fraction of market capitalization")
plt.xscale("log"); plt.yscale("log")
plt.legend()
plt.tick_params(direction='in', which='both')
plt.savefig(os.path.join(os.path.dirname(__file__), "capitalization_distribution.pdf"), dpi=300)


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
        #plt.xlabel("prob. dens. dist.: {}".format(r"$\frac{\epsilon_t^L}{\sigma\sqrt{t}}-\mathcal{N}(0,1)$"))
        plt.xlabel("{}".format(r"$\frac{\epsilon_t^L}{\sigma\sqrt{t}}$"))
        plt.yticks(ticks=np.arange(0, result["pdf"].shape[0]-1, 10), labels=np.arange(0, result["pdf"].shape[0]-1, 10))
        plt.xticks(ticks=[np.searchsorted(result["bin_center"], j) for j in np.arange(-3, 4, 1)], labels=np.arange(-3, 4, 1))
        #if j == len(three_year_interval_list)-1: plt.colorbar()
        plt.ylim(0, result["pdf"].shape[0]-1)
        #plt.title("{}-{}".format(three_year_interval_list[j][0].strftime("%Y/%m/%d"), three_year_interval_list[j][1].strftime("%Y/%m/%d")))
else:
    plt.figure(figsize=(18, 3))
    for j in range(len(five_year_interval_list)):
    #for j in [0]:
        plt.subplot(1, len(five_year_interval_list),j+1)
        result = cummulative_residual_return_BMref(five_year_interval_list[j][0], five_year_interval_list[j][1])
        plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
        #plt.imshow(result["pdf"]-result["pdf_normal"], aspect="auto", cmap="RdBu_r")
        plt.ylabel(r"$\alpha$ (days)")
        #plt.xlabel("{}".format(r"$\frac{\epsilon_t^L}{\sigma\sqrt{t}}$"))
        plt.xlabel("{}".format(r"$\frac{1}{\sigma\sqrt{\alpha}}\sum_{j=1}^{\alpha}\epsilon_{t+j-L}$"))
        plt.yticks(ticks=np.arange(0, result["pdf"].shape[0]-1, 10), labels=np.arange(0, result["pdf"].shape[0]-1, 10))
        plt.xticks(ticks=[np.searchsorted(result["bin_center"], j) for j in np.arange(-3, 4, 1)], labels=np.arange(-3, 4, 1))
        #if j == len(five_year_interval_list)-1: plt.colorbar()
        plt.ylim(0, result["pdf"].shape[0]-1)
        #plt.title("{}-{}".format(five_year_interval_list[j][0].strftime("%Y/%m/%d"), five_year_interval_list[j][1].strftime("%Y/%m/%d")))

#plt.suptitle("PCA {}: cummulative residual return versus Brownian motion".format(PCA_TYPE))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "cummulative_residual_return_std_norm_BM_compare_{}.png".format(PCA_TYPE)), dpi=300)
plt.savefig(os.path.join(os.path.dirname(__file__), "cummulative_residual_return_std_norm_BM_compare_{}.pdf".format(PCA_TYPE)), dpi=300)

#%%
plt.figure(figsize=(5, 1.5))
plt.imshow(np.linspace(0.03, -0.03, 1000).reshape(1, -1), aspect="auto", cmap="RdBu", vmin=-0.03, vmax=0.03)
plt.xticks([0, 500, 999], ["-0.03", "0", "0.03"])
plt.yticks([])
plt.gca().tick_params(axis='y', labelleft=False, labelright=True, left=False, right=True)
#plt.title(r"$\rho(\frac{\epsilon_t^L}{\sigma\sqrt{t}})-\mathcal{N}(0,1)$")
#plt.title(r"p.d.f.$(\frac{1}{\sigma\sqrt{\alpha}}\sum_{j=1}^{\alpha}\epsilon_{t-L+\alpha})-\frac{1}{\sqrt{2\pi}}\exp(-\frac{}{})$")
plt.title(r"p.d.f.$(\tilde{x}_{t-L+\alpha})-\frac{1}{\sqrt{2\pi}}\exp(-\frac{\tilde{x}_{t-L+\alpha}^2}{2})$", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "cumulative_residual_return_std_norm_BM_compare_colorbar.pdf"), dpi=300)

#%% mean reverting time (fit from OU process) - probability density function of tau across different time
def OU_process_rate(trading_signal, t_start, t_end, R2_threshold=0.7):
    time_axis = trading_signal.time_axis
    t_idx = np.arange(np.searchsorted(time_axis, t_start), np.searchsorted(time_axis, t_end)+1, 1)
    T_max = PCA_factor_config["residual_return_evaluation_window_length"]; T_min = 1
    time_hist = []; kappa_hist = []; kappa_flatten = []
    tau_hist = []; tau_flatten = []
    valid_count = 0; instance_count = 0
    for j in tqdm.tqdm(t_idx):
        time = time_axis[j]
        if time >= t_eval_start and time <= t_eval_end:
            result = trading_signal.trading_signal(time)
            idx_valid = [k for k in range(len(result["R_sq"])) if result["R_sq"][k] > R2_threshold]
            valid_count += len(idx_valid); instance_count += len(result["R_sq"])
            kappa_hist.append(list(np.array([result["kappa"][k] for k in idx_valid])/252))
            kappa_flatten.extend(list(np.array([result["kappa"][k] for k in idx_valid])/252))
            tau_hist.append(list(252/np.array([result["kappa"][k] for k in idx_valid])))
            tau_flatten.extend(list(252/np.array([result["kappa"][k] for k in idx_valid])))
            time_hist.append(time)

    print("valid_count: ", valid_count, "instance_count: ", instance_count, "valid_ratio: ", valid_count/instance_count)

    kappa_max = np.quantile(kappa_flatten, 0.95)
    #kappa_min = np.quantile(kappa_flatten, 0.05)
    kappa_min = 0
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

#result = OU_process_rate(trading_signal_OU_, t_eval_start, t_eval_end, R2_threshold=0.7)
result = OU_process_rate(trading_signal_OU_, t_eval_start, t_eval_end, R2_threshold=0.0)
time_hist = result["time"]; kappa_bin_center, kappa_pdf, kappa_hist, kappa_flatten = result["kappa"]
tau_bin_center, tau_pdf, tau_hist, tau_flatten = result["tau"]

#%% mean reverting time (fit from OU process) - histogram of tau
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
        plt.xlabel(r"$\tau$ (days)"); plt.ylabel("probability density")
        plt.ylim(0, 0.2)
        #plt.xlim(0, PCA_factor_config["residual_return_evaluation_window_length"])
        plt.xlim(0, 50)
        #plt.legend()
        plt.tick_params(direction='in', which='both')
        log.append([hist, bin_center])
else:
    for j in range(len(five_year_interval_list)):
        #result = OU_process_rate(trading_signal_OU_, five_year_interval_list[j][0], five_year_interval_list[j][1], R2_threshold=0.7)
        result = OU_process_rate(trading_signal_OU_, five_year_interval_list[j][0], five_year_interval_list[j][1], R2_threshold=0.0)
        time_hist = result["time"]
        kappa_bin_center, kappa_pdf, kappa_hist, kappa_flatten = result["kappa"]
        tau_bin_center, tau_pdf, tau_hist, tau_flatten = result["tau"]

        plt.subplot(1, len(five_year_interval_list), j+1)
        hist, bin_edges = np.histogram(tau_flatten, bins=PCA_factor_config["residual_return_evaluation_window_length"], range=(0, PCA_factor_config["residual_return_evaluation_window_length"]), density=True)
        bin_center = (bin_edges[0:-1] + bin_edges[1:])/2
        plt.plot(bin_center, np.maximum(hist, 1e-3), label=five_year_interval_legend_str[j], color=plot_color)
        #plt.vlines(bin_center[np.argmax(hist)], 0, np.max(hist), color=plot_color, linestyle="--", label="peak: {:.2f}".format(bin_center[np.argmax(hist)]))
        idx = np.where(bin_center>=30)[0]
        plt.fill_between(bin_center[idx], 1e-3, hist[idx], alpha=0.3, color=plot_color)
        idx = np.argmax(np.maximum(hist, 1e-3))
        print("peak: ", bin_center[idx], "std:", np.std(tau_flatten))
        plt.vlines(bin_center[idx], 1e-3, np.maximum(hist[idx], 1e-3), color=plot_color, linestyle="--", label="peak: {:.2f}".format(bin_center[idx]))
        plt.xlabel(r"$\tau$ (days)"); plt.ylabel("probability density")
        plt.ylim(1e-3, 0.5)
        #plt.xlim(0, PCA_factor_config["residual_return_evaluation_window_length"])
        plt.xlim(0, 50)
        plt.yscale("log")
        plt.tick_params(direction='in', which='both')
        #plt.legend()
        log.append([hist, bin_center])
#plt.suptitle(r"{}: prob. dens. dist. of $\tau$ by OU process".format(PCA_TYPE))
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "OU_process_tau_hist_{}.pdf".format(PCA_TYPE)), dpi=300)


#%% statistics of gap process
rank_min = 0; rank_max = 499
local_time_interval_distribution = []

if os.path.exists(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.npz")):
    result = np.load(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.npz"))
    local_time_interval_distribution = result["local_time_interval_distribution"]
    bin_center = result["bin_center"]
else:
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

#%%
plt.figure(figsize=(8, 4))
x = bin_center[1:]; y = np.nanmean(local_time_interval_distribution[1:, :], axis=1)
y = y/np.sum(y)
y = np.maximum(1e-7, y)
plt.plot(x, y)
plt.xlim(0, 900); plt.yscale("log");plt.ylim(1e-7, 1)
plt.xlabel(r"$\tau$ (min)"); plt.ylabel("probability density")

x = bin_center[1:]; y_cum_sum = np.cumsum(y)
idx = np.searchsorted(y_cum_sum, 0.99)
plt.fill_between(x[idx:], 1e-7, np.maximum(np.nanmean(local_time_interval_distribution[1:, :], axis=1)[idx:], 1e-7), alpha=0.3)

'''
segment = [10, 30]
idx = [j for j in range(len(bin_center)) if bin_center[j] >= segment[0] and bin_center[j] <= segment[1]]
X = bin_center[idx].reshape(-1, 1); Y = np.log(np.mean(local_time_interval_distribution[idx, :][:, t_idx], axis=1))
reg = LinearRegression().fit(X, Y)
X_continuous = np.linspace(segment[0], 45, 100).reshape(-1, 1)
Y_continuous = np.exp(reg.predict(X_continuous))
plt.plot(X_continuous, Y_continuous, linestyle="--", color="black", alpha=0.5)

segment = [100, 200]
idx = [j for j in range(len(bin_center)) if bin_center[j] >= segment[0] and bin_center[j] <= segment[1]]
X = bin_center[idx].reshape(-1, 1); Y = np.log(np.mean(local_time_interval_distribution[idx, :][:, t_idx], axis=1))
reg = LinearRegression().fit(X, Y)
X_continuous = np.linspace(45, segment[1], 100).reshape(-1, 1)
Y_continuous = np.exp(reg.predict(X_continuous))
plt.plot(X_continuous, Y_continuous, linestyle="--", color="black", alpha=0.5)
'''
plt.vlines(225, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], color='red', linestyle='--', alpha=0.5, label="225 min")
plt.tick_params(direction='in', which='both')
plt.savefig(os.path.join(os.path.dirname(__file__), "local_time_interval_distribution.pdf"), dpi=300)



#%% mean and variance: empirical versus parametric
R2_threshold = 0.7
empirical_mean = []; empirical_var = []
parametric_mean = []; parametric_var = []
total_instance = 0

t_idx = np.arange(np.searchsorted(PCA_factor_.time_axis, t_eval_start), np.searchsorted(PCA_factor_.time_axis, t_eval_end)+1, 1)
for j in tqdm.tqdm(t_idx):
    t = PCA_factor_.time_axis[j]
    result = trading_signal_OU_.trading_signal(t)
    idx_valid = [k for k in range(len(result["R_sq"])) if result["R_sq"][k] > R2_threshold]
    total_instance += len(idx_valid)
    mean = np.array(result["mu"])[idx_valid]; var = (np.array(result["sigma"])**2)[idx_valid]
    parametric_mean.extend(list(mean)); parametric_var.extend(list(var))

    epsilon = PCA_factor_.residual_return(t)["epsilon"]
    cumulative_epsilon = np.cumsum(epsilon, axis=1)
    mean = np.mean(cumulative_epsilon, axis=1, keepdims=False)
    var = np.var(cumulative_epsilon, ddof=1, axis=1, keepdims=False)
    empirical_mean.extend(list(mean[idx_valid])); empirical_var.extend(list(var[idx_valid]))

print("total_instance: ", total_instance)

#%%
plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.scatter(empirical_mean, parametric_mean, s=1)
plt.xlabel("empirical mean"); plt.ylabel("OU mean")
plt.xlim(-1, 1); plt.ylim(-1, 1)

plt.subplot(1,2,2)
plt.scatter(empirical_var, parametric_var, s=1)
plt.xlabel("empirical var"); plt.ylabel("OU var")
plt.xlim(0, 0.5); plt.ylim(0, 0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "{}_empirical_vs_parametric_mean_var.pdf".format(PCA_TYPE)), dpi=300)


#%%



