#%%
import os, sys, copy, h5py, datetime, gc, tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def select_equity_data_by_date(raw_file_name, t_start, t_end):
    data = pd.read_csv(raw_file_name, usecols = ['PERMNO','Ticker', "DlyCalDt", "DlyPrc", "ShrOut", "DlyCap", "DlyRet"])[['PERMNO','Ticker', "DlyCalDt", "DlyPrc", "ShrOut", "DlyCap", "DlyRet"]]
    idx = [j for j in range(data.shape[0]) if datetime.datetime.strptime(data.iloc[j,2], "%Y-%m-%d")>=t_start and datetime.datetime.strptime(data.iloc[j,2], "%Y-%m-%d")<=t_end]
    save_file_name = "".join(["equity_data", "_", datetime.datetime.strftime(t_start, "%Y%m%d"), "_", datetime.datetime.strftime(t_end, "%Y%m%d"), ".csv"])
    data = data.iloc[idx, :].copy()
    data.to_csv(save_file_name)

class equity_by_rank:
    def __init__(self, time, market_capitalization, equity_idx_by_rank):
        self.time = time; self.market_capitalization = market_capitalization; self.equity_idx_by_rank = equity_idx_by_rank

class equity_by_name:
    def __init__(self, equity_idx, price, share_outstanding, capitalization, return_, rank):
        self.equity_idx = equity_idx; self.price = price; self.share_outstanding = share_outstanding
        self.capitalization = capitalization; self.return_ = return_; self.rank = rank

class equity_data:
    def __init__(self, data_config):
        self.config = data_config
        is_complete_data = False
        equity_data_file_name = self.config["equity_file_name"].replace(".csv", ".npz") if (not is_complete_data) else ''.join([self.config["equity_file_name"][0:-4], "_complete.npz"])
        if not os.path.exists(equity_data_file_name):
            print("Initialize processing equity data.")
            factor_data = pd.read_csv(self.config["Fama_French_3factor_file_name"])
            factor_data.dropna(axis=0, how='any', inplace=True)
            time_axis = [datetime.datetime.strptime(j, "%m/%d/%y") for j in factor_data["date"].to_list()]
            time_axis_int = [int(datetime.datetime.strftime(j, "%Y%m%d")) for j in time_axis]
            risk_free_rate = factor_data["rf"].to_list()
            data = pd.read_csv(self.config["equity_file_name"], usecols = ['PERMNO','Ticker', "DlyCalDt", "DlyPrc", "ShrOut", "DlyCap", "DlyRet"])[['PERMNO','Ticker', "DlyCalDt", "DlyPrc", "ShrOut", "DlyCap", "DlyRet"]]
            data.dropna(axis=0, how='any', inplace=True); data.drop_duplicates(keep='first', inplace=True); data.reset_index(drop=True, inplace=True)
            PERMNO_list = np.sort(list(set(data["PERMNO"].to_list())))
            equity_idx_PERMNO_ticker = []
            for j in range(len(PERMNO_list)):
                sub_df = data[data["PERMNO"]==PERMNO_list[j]]["PERMNO", "Ticker"].copy()
                sub_df.drop_duplicates(keep='first', inplace=True); sub_df.reset_index(drop=True, inplace=True)
                for k in range(sub_df.shape[0]):
                    equity_idx_PERMNO_ticker.append([j, sub_df.iloc[k,0], sub_df.iloc[k,1]])
            equity_idx_PERMNO_ticker_df = {"equity_idx": [j[0] for j in equity_idx_PERMNO_ticker], "PERMNO": [j[1] for j in equity_idx_PERMNO_ticker], "Ticker": [j[2] for j in equity_idx_PERMNO_ticker]}
            equity_idx_PERMNO_ticker_df = pd.DataFrame(equity_idx_PERMNO_ticker)
            equity_idx_PERMNO_ticker_df.to_csv("equity_idx_PERMNO_ticker.csv", index=False)
            self.equity_idx_PERMNO_ticker_df = equity_idx_PERMNO_ticker_df
            equity_idx_list = list(np.arange(0, len(PERMNO_list), 1))

            # censoring data in daily return
            print("Censoring equity data in daily return.")
            return_abnormality_proportion_num = 0
            for j in np.arange(0, data.shape[0], 1):
                if np.abs(data.iloc[j,6])>self.config["filter_by_return_threshold"]:
                    return_abnormality_proportion_num += 1
                    if data.iloc[j,0] == data.iloc[j-1,0]:
                        data.iloc[j,6] = min(self.config["filter_by_return_threshold"], max(-self.config["filter_by_return_threshold"], data.iloc[j-1,6]))
                    else:
                        data.iloc[j,6] = 0.0

            self.f = h5py.File(self.config["equity_file_name"].replace(".csv", ".hdf5"), 'w')
            group = self.f.create_group("general_info")
            try:
                group.create_dataset("file_creation_time", data = [datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S").encode("utf-8")], dtype=h5py.special_dtype(vlen=str))
                group.create_dataset("ticker", data = [k.encode("utf-8") for k in self.equity_idx_PERMNO_ticker_df["Ticker"].to_list()], dtype=h5py.special_dtype(vlen=str))
            except:
                group.create_dataset("file_creation_time", data = [datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")], dtype=h5py.special_dtype(vlen=str))
                group.create_dataset("ticker", data = [k for k in self.equity_idx_PERMNO_ticker_df["Ticker"].to_list()], dtype=h5py.special_dtype(vlen=str))
            group.create_dataset("PERMNO", data = self.equity_idx_PERMNO_ticker_df["PERMNO"].to_list())
            group.create_dataset("equity_idx", data = self.equity_idx_PERMNO_ticker_df["equity_idx"].to_list())
            group.create_dataset("filter_by_return_threshold", data = [self.config["filter_by_return_threshold"]])
            group.create_dataset("filter_by_return_proportion", data = [return_abnormality_proportion_num/data.shape[0]])
            self.f.close()

            # load and transform data in rank space
            print("Transforming equity data in rank space.")
            if os.path.exists(os.path.join(os.path.dirname(__file__), "equity_by_rank_list.pkl")):
                equity_by_rank_list_pkl = open(os.path.join(os.path.dirname(__file__), "equity_by_rank_list.pkl"), 'rb')
                equity_by_rank_list = pickle.load(equity_by_rank_list_pkl); equity_by_rank_list_pkl.close()
                market_capitalization_pkl = open(os.path.join(os.path.dirname(__file__), "market_capitalization.pkl"), 'rb')
                market_capitalization = pickle.load(market_capitalization_pkl); market_capitalization_pkl.close()
            else:
                equity_by_rank_list = []; market_capitalization = []
            for j in tqdm.tqdm(np.arange(len(equity_by_rank_list), len(time_axis), 1)):
                if j % 1000 == 0:
                    equity_by_rank_list_pkl = open(os.path.join(os.path.dirname(__file__), "equity_by_rank_list.pkl"), 'wb')
                    pickle.dump(equity_by_rank_list, equity_by_rank_list_pkl); equity_by_rank_list_pkl.close()
                    market_capitalization_pkl = open(os.path.join(os.path.dirname(__file__), "market_capitalization.pkl"), 'wb')
                    pickle.dump(market_capitalization, market_capitalization_pkl); market_capitalization_pkl.close()
                time = time_axis[j]
                data_sub = data[data["DlyCalDt"]==datetime.datetime.strftime(time, "%Y-%m-%d")].copy()
                data_sub.sort_values(by=["DlyCap"], axis=0, ascending=False, inplace=True)
                capitalization_by_rank = data_sub["DlyCap"].to_list()
                PERMNO_by_rank = data_sub["PERMNO"].to_list()
                equity_idx_by_rank = [np.searchsorted(PERMNO_list, j) for j in PERMNO_by_rank]
                equity_by_rank_list.append(equity_by_rank(time, np.sum(capitalization_by_rank), equity_idx_by_rank))
                market_capitalization.append(np.sum(capitalization_by_rank))

            # load and transform data in name space
            print("Transforming equity data in name space.")
            if os.path.exists(os.path.join(os.path.dirname(__file__), "equity_by_name_list.pkl")):
                equity_by_name_list_pkl = open(os.path.join(os.path.dirname(__file__), "equity_by_name_list.pkl"), 'rb')
                equity_by_name_list = pickle.load(equity_by_name_list_pkl); equity_by_name_list_pkl.close()
            else:
                equity_by_name_list = []
            for j in tqdm.tqdm(np.arange(len(equity_by_name_list), len(equity_idx_list), 1)):
                if j % 1000 == 0:
                    equity_by_name_list_pkl = open(os.path.join(os.path.dirname(__file__), "equity_by_name_list.pkl"), 'wb')
                    pickle.dump(equity_by_name_list, equity_by_name_list_pkl); equity_by_name_list_pkl.close()
                data_sub = data[data["PERMNO"]==PERMNO_list[j]].copy()
                time_equity = [datetime.datetime.strptime(j, "%Y-%m-%d") for j in data_sub["DlyCalDt"].to_list()]
                price = []; share_outstanding = []; capitalization = []; return_ = []; rank = []
                for pt_time in range(len(time_axis)):
                    if time_axis[pt_time] in time_equity:
                        pt_equity = np.searchsorted(time_equity, time_axis[pt_time])
                        price.append(data_sub.iloc[pt_equity,3]); share_outstanding.append(data_sub.iloc[pt_equity,4])
                        capitalization.append(data_sub.iloc[pt_equity,5]); return_.append(data_sub.iloc[pt_equity,6])
                        rank.append(equity_by_rank_list[pt_time].equity_idx_by_rank.index(j))
                    else:
                        price.append(np.nan); share_outstanding.append(np.nan); capitalization.append(np.nan); return_.append(np.nan); rank.append(np.nan)
                equity_by_name_list.append(equity_by_name(j, price, share_outstanding, capitalization, return_, rank))

            self.time_axis = time_axis; self.time_axis_int = time_axis_int; self.risk_free_rate = np.array(risk_free_rate); self.equity_idx_list = equity_idx_list
            N = len(self.equity_idx_list); T = len(self.time_axis)
            self.equity_idx_by_rank = []
            self.price = []; self.share_outstanding = []; self.capitalization = []; self.return_ = []; self.rank = []
            for t in range(T):
                ar = equity_by_rank_list[t].equity_idx_by_rank; ar.extend([np.nan for _ in range(N-len(ar))])
                self.equity_idx_by_rank.append(ar)
            for j in range(len(self.equity_idx_list)):
                self.price.append(equity_by_name_list[j].price); self.share_outstanding.append(equity_by_name_list[j].share_outstanding)
                self.return_.append(equity_by_name_list[j].return_); self.rank.append(equity_by_name_list[j].rank); self.capitalization.append(equity_by_name_list[j].capitalization)
            self.equity_idx_by_rank = np.array(self.equity_idx_by_rank).T
            self.price = np.array(self.price); self.share_outstanding = np.array(self.share_outstanding); self.capitalization = np.array(self.capitalization)
            self.return_ = np.array(self.return_); self.rank = np.array(self.rank)

            np.savez(self.config["equity_file_name"].replace(".csv", ".npz"), time_axis_int=self.time_axis_int, risk_free_rate=self.risk_free_rate, equity_idx_list=self.equity_idx_list,\
                    equity_idx_by_rank = self.equity_idx_by_rank, price=self.price, share_outstanding=self.share_outstanding, capitalization=self.capitalization, return_=self.return_, rank=self.rank)
            os.remove(os.path.join(os.path.dirname(__file__), "equity_by_rank_list.pkl"))
            os.remove(os.path.join(os.path.dirname(__file__), "market_capitalization.pkl"))
            os.remove(os.path.join(os.path.dirname(__file__), "equity_by_name_list.pkl"))
            if not is_complete_data:
                del self.price, self.share_outstanding
        else:
            data = np.load(equity_data_file_name)
            if is_complete_data:
                self.time_axis = [datetime.datetime.strptime(str(j), "%Y%m%d") for j in data["time_axis_int"]]
                self.time_axis_int = np.array(data["time_axis_int"]); self.risk_free_rate = np.array(data["risk_free_rate"])
                self.equity_idx_list = np.array(data["equity_idx_list"])
                self.equity_idx_by_rank = np.array(data["equity_idx_by_rank"]); self.return_ = np.array(data["return_"])
                self.share_outstanding = np.array(data["share_outstanding"]); self.price = np.array(data["price"])
                self.capitalization = np.array(data["capitalization"]); self.rank = np.array(data["rank"])
            else:
                self.time_axis = [datetime.datetime.strptime(str(j), "%Y%m%d") for j in data["time_axis_int"]]
                self.time_axis_int = np.array(data["time_axis_int"]); self.risk_free_rate = np.array(data["risk_free_rate"])
                self.equity_idx_list = np.array(data["equity_idx_list"])
                self.equity_idx_by_rank = np.array(data["equity_idx_by_rank"]); self.return_ = np.array(data["return_"])
                self.rank = np.array(data["rank"]); self.capitalization = np.array(data["capitalization"])

        print("Initialize data loader complete.")

    def return_by_name(self, equity_idx, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
        return self.return_[equity_idx,:][:,t_idx]
    
    def return_by_name_rank_filtered(self, t_start, t_end, rank_min, rank_max):
        t_start_idx = np.searchsorted(self.time_axis, t_start); t_end_idx = np.searchsorted(self.time_axis, t_end)
        time_idx_list = np.arange(t_start_idx, t_end_idx+1, 1); equity_idx = []
        for j in range(self.return_.shape[0]):
            if (not any(np.isnan(self.return_[j,time_idx_list]))) and (self.rank[j,t_end_idx]>=rank_min and self.rank[j,t_end_idx]<=rank_max):
                equity_idx.append(j)
        return (equity_idx, self.return_[equity_idx, t_start_idx:(t_end_idx+1)])

    def return_by_rank_func(self, rank_idx, t_start, t_end, mode="theta-trans"):
        if mode == "permutation":
            t_idx = np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1)
            equity_idx_by_rank = self.equity_idx_by_rank[rank_idx, :][:, t_idx-1]
            idx_valid_rank = [j for j in range(len(rank_idx)) if not any(np.isnan(equity_idx_by_rank[j,:]))]
            rank_idx = [rank_idx[j] for j in idx_valid_rank]
            r = np.zeros((len(rank_idx), len(t_idx))); r[:] = np.nan
            for t in t_idx:
                equity_idx_prev = self.equity_idx_by_rank[rank_idx, t-1]
                r[:,t-t_idx[0]] = self.return_[equity_idx_prev.astype(np.int32), t]
            idx_valid_rank = [j for j in range(len(rank_idx)) if not any(np.isnan(r[j,:]))]
            rank_idx = [rank_idx[j] for j in idx_valid_rank]; r = r[idx_valid_rank,:]
            return {"time": [self.time_axis[j] for j in t_idx], "rank_idx": rank_idx, "return_by_rank": r}

        if mode == "theta-trans":
            result = self.occupation_rate_by_rank(rank_idx, t_start, t_end, mode="rank")
            equity_idx = result["equity_idx"]; rank_idx = result["rank_idx"]; theta = result["occupation_rate"]
            r = self.return_[equity_idx, :][:,t_idx]; theta_r = theta.dot(r)
            return {"time": [self.time_axis[j] for j in t_idx], "equity_idx": equity_idx, "rank_idx": rank_idx, "occupation_rate": theta, "return_by_rank": theta_r}

        if mode == "hybrid-Atlas":
            t_idx = np.arange(np.searchsorted(self.time_axis, t_start)-1, np.searchsorted(self.time_axis, t_end)+1, 1)
            equity_idx_by_rank = self.equity_idx_by_rank[rank_idx, :][:, t_idx]
            idx_valid_rank = [j for j in range(len(rank_idx)) if not any(np.isnan(equity_idx_by_rank[j,:]))]
            rank_idx = [rank_idx[j] for j in idx_valid_rank]
            t_idx = t_idx[1:]
            r = np.zeros((len(rank_idx), len(t_idx))); r[:] = np.nan
            for t in t_idx:
                equity_idx_now = self.equity_idx_by_rank[rank_idx, t]; equity_idx_prev = self.equity_idx_by_rank[rank_idx, t-1]
                capitalization_now = self.capitalization[equity_idx_now.astype(np.int32), t]
                capitalization_prev = self.capitalization[equity_idx_prev.astype(np.int32), t-1]
                r[:,t-t_idx[0]] = (capitalization_now-capitalization_prev)/capitalization_prev
            return {"time": [self.time_axis[j] for j in t_idx], "rank_idx": rank_idx, "return_by_rank": r}

    def occupation_rate_by_name(self, equity_idx, t_start, t_end):
        t_idx_list = list(np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1))
        rank_sub = copy.deepcopy((self.rank[equity_idx, :])[:, t_idx_list]).astype(np.int32); rank_min = np.min(rank_sub); rank_max = np.max(rank_sub)
        rank_idx = np.arange(rank_min, rank_max+1, 1, dtype=np.int32)
        occupation_time = np.zeros((len(rank_idx), len(equity_idx)))
        for j in range(len(equity_idx)):
            for t_idx in range(len(t_idx_list)):
                occupation_time[np.searchsorted(rank_idx,rank_sub[j,t_idx]),j] += 1
        occupation_rate = copy.deepcopy(occupation_time/np.sum(occupation_time, axis=0, keepdims=True))
        return {"time": self.time_axis[t_idx_list], "equity_idx": equity_idx, "rank_idx": list(rank_idx), "occupation_time": occupation_time, "occupation_rate": occupation_rate}

    def occupation_rate_by_rank(self, rank_idx, t_start, t_end, mode="rank", is_parallel=True):
        t_idx = list(np.arange(np.searchsorted(self.time_axis, t_start), np.searchsorted(self.time_axis, t_end)+1, 1))
        equity_idx_by_rank = copy.deepcopy(self.equity_idx_by_rank[rank_idx,:][:,t_idx])
        rank_idx = [rank_idx[j] for j in range(len(rank_idx)) if not any(np.isnan(equity_idx_by_rank[j,:]))]
        equity_idx = np.unique(self.equity_idx_by_rank[rank_idx, :][:, t_idx].flatten())
        equity_idx = np.sort(equity_idx).astype(np.int32)
        valid_idx = [j for j in range(len(equity_idx)) if not any(np.isnan(self.return_[equity_idx[j], t_idx]))]
        equity_idx = [equity_idx[j] for j in valid_idx]
        if mode == "name":
            return self.occupation_rate_by_name(equity_idx, t_start, t_end)
        if mode == "rank":
            is_parallel = True if len(rank_idx)>=200 else False
            if is_parallel:
                cpu_core = min(multiprocessing.cpu_count(), 24); num_per_batch = int(np.ceil(len(rank_idx)/cpu_core))
                equity_idx_by_rank = copy.deepcopy(self.equity_idx_by_rank[rank_idx,:][:,t_idx])
                def occupation_time_core(batch_idx):
                    occupation_time_sub = np.zeros((len(rank_idx), len(equity_idx)))
                    idx_rank_axis = np.arange(batch_idx*num_per_batch, min((batch_idx+1)*num_per_batch, len(rank_idx)), 1)
                    for j in idx_rank_axis:
                        for k in range(len(t_idx)):
                            if int(equity_idx_by_rank[j,k]) in equity_idx:
                                occupation_time_sub[j, np.searchsorted(equity_idx, int(equity_idx_by_rank[j,k]))] += 1
                    return occupation_time_sub
                result = Parallel(n_jobs=cpu_core)(delayed(occupation_time_core)(j) for j in range(cpu_core))
                occupation_time = np.sum(np.array(result), axis=0)
                del result; gc.collect()
            else:
                occupation_time = np.zeros((len(rank_idx), len(equity_idx)))
                for j in range(len(rank_idx)):
                    for k in range(len(t_idx)):
                        if int(self.equity_idx_by_rank[rank_idx[j],t_idx[k]]) in equity_idx:
                            occupation_time[j, np.searchsorted(equity_idx, int(self.equity_idx_by_rank[rank_idx[j],t_idx[k]]))] += 1

            idx_valid_rank = [j for j in range(len(rank_idx)) if np.sum(occupation_time[j,:])>0]
            rank_idx = [rank_idx[j] for j in idx_valid_rank]; occupation_time = occupation_time[idx_valid_rank,:]
            occupation_rate = np.array([list(occupation_time[j,:]/np.sum(occupation_time[j,:])) for j in range(occupation_time.shape[0])])

            return {"time": [self.time_axis[j] for j in t_idx], "equity_idx": equity_idx, "rank_idx": rank_idx, "occupation_time": occupation_time, "occupation_rate": occupation_rate}

class equity_data_high_freq:
    def __init__(self, config):
        self.config = config
        # load daily data derived from high frequency data
        file_name = os.path.join(os.path.dirname(__file__), "equity_data_high_frequency/equity_data_Polygon_daily_from_high_freq_year_2003_2022.npz")
        data = np.load(file_name, allow_pickle=True)
        time_axis_daily = [datetime.datetime.fromtimestamp(j) for j in data["time_axis"]]
        t_idx_start = np.searchsorted(time_axis_daily, datetime.datetime(2004, 1, 1))
        self.time_axis_daily = time_axis_daily[t_idx_start:]
        self.price_daily = data["price"][:, t_idx_start:]; self.capitalization_daily = data["capitalization"][:, t_idx_start:]
        self.rank_daily = data["rank"][:, t_idx_start:]; self.equity_idx_by_rank_daily = data["equity_idx_by_rank"][:, t_idx_start:]
        del data; gc.collect()
        print("Initialize data loader (high frequency) complete.")
    
    def daily_return_by_name(self, equity_idx, t_start, t_end):
        '''
        Caution: the calculated return is at face value from equity price, which does not take into account of dividend, stock split, etc.
        params:
            equity_idx: list of equity_idx such that it has valid price between (t_start - one day) and t_end
            t_start: datetime.datetime
            t_end: datetime.datetime
        '''
        t_idx = np.arange(np.searchsorted(self.time_axis_daily, t_start), np.searchsorted(self.time_axis_daily, t_end)+1, 1)
        return_ = np.zeros((len(equity_idx), len(t_idx))); return_[:] = np.nan
        for j in range(len(t_idx)):
            return_[:, j] = (self.price_daily[equity_idx,:][:,t_idx[j]]-self.price_daily[equity_idx,:][:,t_idx[j]-1])/self.price_daily[equity_idx,:][:,t_idx[j]-1]
        return return_

    def daily_return_by_rank_func(self, rank_idx, t_start, t_end, mode="hybrid-Atlas"):
        if mode == "hybrid-Atlas":
            t_idx = np.arange(np.searchsorted(self.time_axis_daily, t_start)-1, np.searchsorted(self.time_axis_daily, t_end)+1, 1)
            equity_idx_by_rank = self.equity_idx_by_rank_daily[rank_idx, :][:, t_idx]
            idx_valid_rank = [j for j in range(len(rank_idx)) if not any(np.isnan(equity_idx_by_rank[j,:]))]
            rank_idx = [rank_idx[j] for j in idx_valid_rank]
            t_idx = t_idx[1:]
            r = np.zeros((len(rank_idx), len(t_idx))); r[:] = np.nan
            for t in t_idx:
                equity_idx_now = self.equity_idx_by_rank_daily[rank_idx, t]; equity_idx_prev = self.equity_idx_by_rank_daily[rank_idx, t-1]
                capitalization_now = self.capitalization_daily[equity_idx_now.astype(np.int32), t]
                capitalization_prev = self.capitalization_daily[equity_idx_prev.astype(np.int32), t-1]
                r[:,t-t_idx[0]] = (capitalization_now-capitalization_prev)/capitalization_prev
            return {"time": [self.time_axis_daily[j] for j in t_idx], "rank_idx": rank_idx, "return_by_rank": r}

    def intraday_data(self, t_eval):
        '''
        5-min high frequency data.
        params:
            t_eval: datetime.datetime
        '''
        file_name = os.path.join(os.path.dirname(__file__), "equity_data_high_frequency/minute_aggs_tensorized_filtered/equity_data_Polygon_high_freq_5_min_{}.npz".format(datetime.datetime.strftime(t_eval, "%Y-%m-%d")))
        data = np.load(file_name, allow_pickle=True)
        time_axis_high_freq = [datetime.datetime.fromtimestamp(j) for j in data["time_axis_high_freq"]]
        capitalization_high_freq = data["capitalization_high_freq"]
        rank_high_freq = data["rank_high_freq"]
        equity_idx_by_rank_high_freq = data["equity_idx_by_rank_high_freq"]
        return {"time": time_axis_high_freq, "capitalization": capitalization_high_freq, "rank": rank_high_freq, "equity_idx_by_rank": equity_idx_by_rank_high_freq}



# %%
