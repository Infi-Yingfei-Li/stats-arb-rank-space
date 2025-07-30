import os, sys, copy, h5py, datetime, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data.data as data
import market_decomposition.market_factor_classic as factor
import trading_signal.trading_signal as trading_signal
import portfolio_weights.portfolio_weights as portfolio_weights
import utils.utils as utils

IS_LARGE_MEMORY = True
IS_NEURAL_NETWORK_RETRAIN_HIGH_FREQ = True
GPU_NUMBER = 1
IS_ARGPARSE = False

if IS_ARGPARSE:
    parser = argparse.ArgumentParser(description='train GPU id')
    parser.add_argument('GPU_id', type=int, help='specify GPU id for training')
    args = parser.parse_args()
    GPU_id = args.GPU_id # distributed training on multiple GPUs with index 0, 1, 2, ..., GPU_NUMBER-1
else:
    GPU_id = 0
if GPU_id >= GPU_NUMBER:
    raise ValueError("GPU id invalid: GPU id should be less than GPU_NUMBER")

#PCA_TYPE = "name"
PCA_TYPE = "rank_hybrid_Atlas"
#PCA_TYPE = "rank_hybrid_Atlas_high_freq"
#PCA_TYPE = "rank_permutation"
#PCA_TYPE = "rank_theta_transform"

if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    t_eval_start = datetime.datetime(2005,7,1); t_eval_end = datetime.datetime(2022,12,15)
else:
    t_eval_start = datetime.datetime(1991,1,1); t_eval_end = datetime.datetime(2022,12,15)

equity_data_config = {"Fama_French_3factor_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/Fama_French_3factor_19700101_20221231.csv"),
            "Fama_French_5factor_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/Fama_French_5factor_19700101_20221231.csv"),
            "equity_file_name":  os.path.join(os.path.dirname(__file__), "data/equity_data/equity_data_19700101_20221231.csv"),
            "filter_by_return_threshold": 3,
            "SPX_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/SPX_19700101_20221231.csv"),
            "Russel2000_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/Russel2000_19879010_20231024.csv"),
            "Russel3000_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/Russel3000_19879010_20231110.csv"),
            "macroeconomics_124_factor_file_name": os.path.join(os.path.dirname(__file__), "data/equity_data/macroeconomics_124_factor_19700101_20221231.csv")}
equity_data_ = data.equity_data(equity_data_config)

equity_data_high_freq_config = {}
equity_data_high_freq_ = data.equity_data_high_freq(equity_data_high_freq_config)

if IS_NEURAL_NETWORK_RETRAIN_HIGH_FREQ:
    portfolio_performance_config = {"is_update_network": True,
                                    "train_lookback_window": 499,
                                    "validation_lookback_window": 1,
                                    "reevaluation_interval": 252//4,
                                    "GPU_number": GPU_NUMBER,
                                    "GPU_id": GPU_id,
                                    "leverage": 1,
                                    "transaction_cost_factor": 0.0002,
                                    "shorting_cost_factor": 0.0000,
                                    "high_freq_rebalance_interval": 60}

else:
    portfolio_performance_config = {"is_update_network": True,
                                    "train_lookback_window": 999,
                                    "validation_lookback_window": 89,
                                    "reevaluation_interval": 252,
                                    "GPU_number": GPU_NUMBER,
                                    "GPU_id": GPU_id,
                                    "leverage": 1,
                                    "transaction_cost_factor": 0.0002,
                                    "shorting_cost_factor": 0.0000,
                                    "high_freq_rebalance_interval": 60}

portfolio_performance_ = utils.portfolio_performance(equity_data_, equity_data_high_freq_, portfolio_performance_config)

PCA_factor_config = {"factor_evaluation_window_length": 252,
                "loading_evaluation_window_length": 60, 
                "residual_return_evaluation_window_length": 60,
                "theta_evaluation_window_length": 30,
                "rank_min": 0,
                "rank_max": 499,
                "factor_number": 5,
                "type": PCA_TYPE,
                "max_cache_len": 1100 if IS_LARGE_MEMORY else 400}
PCAf_ = factor.PCA_factor(equity_data_, equity_data_high_freq_, PCA_factor_config)

trading_signal_OU_process_config = {"max_cache_len": 100}
trading_signal_OU_process_ = trading_signal.trading_signal_OU_process(equity_data_, equity_data_high_freq_, PCAf_, trading_signal_OU_process_config)

portfolio_weights_OU_process_config = {"mean_reversion_time_filter_lookback_window": 24,
                                       "active_portfolio_size_threshold": 75,
                                        "R2_filter": 0.7,
                                        "threshold_open": 1.25,
                                        "threshold_close": 0.25,
                                        "max_holding_time": 90}
portfolio_weights_OU_process_ = portfolio_weights.portfolio_weights_OU_process(equity_data_, equity_data_high_freq_, PCAf_, trading_signal_OU_process_, portfolio_weights_OU_process_config)
if PCA_TYPE == "name":
    portfolio_performance_.register_portfolio("PCA_name-OU_process-OU_process", portfolio_weights_OU_process_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_permutation":
    portfolio_performance_.register_portfolio("PCA_rank_permutation-OU_process-OU_process", portfolio_weights_OU_process_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_hybrid_Atlas":
    portfolio_performance_.register_portfolio("PCA_rank_hybrid_Atlas-OU_process-OU_process", portfolio_weights_OU_process_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    portfolio_performance_.register_portfolio("PCA_rank_hybrid_Atlas_high_freq-OU_process-OU_process", portfolio_weights_OU_process_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_theta_transform":
    portfolio_performance_.register_portfolio("PCA_rank_theta-OU_process-OU_process", portfolio_weights_OU_process_, t_eval_start, t_eval_end)

portfolio_weights_CNN_transformer_config = {"PnL_evaluation_window_length": 24,
                                            "risk_aversion_factor": 2,
                                            "transaction_cost_aversion_factor": None,
                                            "rank_variation_aversion_factor": None,
                                            "dollar_neutrality_aversion_factor": None,
                                            "train_t_start": datetime.datetime(1991, 7, 1),
                                            "train_t_end": datetime.datetime(1991, 10, 31),
                                            "valid_t_start": datetime.datetime(1991, 11, 1),
                                            "valid_t_end": datetime.datetime(1991, 12, 15),
                                            "epoch_max": 100,
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

portfolio_weights_CNN_transformer_ = portfolio_weights.portfolio_weights_CNN_transformer(equity_data_, equity_data_high_freq_, PCAf_, trading_signal_OU_process_, portfolio_weights_CNN_transformer_config)

t_eval_start = datetime.datetime(2006, 1, 1); t_eval_end = datetime.datetime(2022, 12, 15) # full range

if PCA_TYPE == "name":
    portfolio_performance_.register_portfolio("PCA_name-CNN_transformer", portfolio_weights_CNN_transformer_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_hybrid_Atlas":
    portfolio_performance_.register_portfolio("PCA_rank_hybrid_Atlas-CNN_transformer", portfolio_weights_CNN_transformer_, t_eval_start, t_eval_end)
if PCA_TYPE == "rank_hybrid_Atlas_high_freq":
    portfolio_performance_.register_portfolio("PCA_rank_hybrid_Atlas_high_freq-CNN_transformer", portfolio_weights_CNN_transformer_, t_eval_start, t_eval_end)


