import os, sys, copy, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.utils as utils

class SPX:
    def __init__(self, equity_data):
        result = equity_data.load_SPX()
        self.time_axis = result["time"]; self.initial_index = result["initial_index"]; self.index = result["index"]; self.return_ = result["return"]

    def portfolio(self, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis,t_start), np.searchsorted(self.time_axis,t_end)+1, 1)
        if any(np.isnan(np.array(self.return_)[t_idx])):
            raise Exception("Error: np.nan exists in SPX during the requested time window")
        else:
            return {"time": [self.time_axis[j] for j in t_idx], "index": [self.index[j] for j in t_idx], "return": [self.return_[j] for j in t_idx]}

class Russel2000:
    def __init__(self, equity_data):
        result = equity_data.load_Russel2000()
        self.time_axis = result["time"]; self.initial_price = result["initial_price"]; self.price = result["price"]; self.return_ = result["return"]

    def portfolio(self, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis,t_start), np.searchsorted(self.time_axis,t_end)+1, 1)
        if any(np.isnan(np.array(self.return_)[t_idx])):
            raise Exception("Error: np.nan exists in Russel2000 during the requested time window")
        else:
            return {"time": [self.time_axis[j] for j in t_idx], "price": [self.price[j] for j in t_idx], "return": [self.return_[j] for j in t_idx]}

class Russel3000:
    def __init__(self, equity_data):
        result = equity_data.load_Russel3000()
        self.time_axis = result["time"]; self.initial_price = result["initial_price"]; self.price = result["price"]; self.return_ = result["return"]

    def portfolio(self, t_start, t_end):
        t_idx = np.arange(np.searchsorted(self.time_axis,t_start), np.searchsorted(self.time_axis,t_end)+1, 1)
        if any(np.isnan(np.array(self.return_)[t_idx])):
            raise Exception("Error: np.nan exists in Russel3000 during the requested time window")
        else:
            return {"time": [self.time_axis[j] for j in t_idx], "price": [self.price[j] for j in t_idx], "return": [self.return_[j] for j in t_idx]}

