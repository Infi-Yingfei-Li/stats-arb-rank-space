#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import data.data as data

#%%
equity_data_config = {"Fama_French_3factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Fama_French_3factor_19700101_20221231.csv"),
            "Fama_French_5factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Fama_French_5factor_19700101_20221231.csv"),
            "equity_file_name":  os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_data_19700101_20221231.csv"),
            "filter_by_return_threshold": 3,
            "SPX_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/SPX_19700101_20221231.csv"),
            "Russel2000_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Russel2000_19879010_20231024.csv"),
            "Russel3000_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/Russel3000_19879010_20231110.csv"),
            "macroeconomics_124_factor_file_name": os.path.join(os.path.dirname(__file__), "../../data/equity_data/macroeconomics_124_factor_19700101_20221231.csv")}
equity_data_ = data.equity_data(equity_data_config)
equity_idx_PERMNO_ticker_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../../data/equity_data/equity_idx_PERMNO_ticker.csv"))

equity_data_high_freq_config = {}
equity_data_high_freq_ = data.equity_data_high_freq(equity_data_high_freq_config)

#%%
t_idx = np.arange(np.searchsorted(equity_data_.time_axis, datetime.datetime(2010,5,25)), np.searchsorted(equity_data_.time_axis, datetime.datetime(2010,5,26))+1, 1)
equity_idx_1 = equity_idx_PERMNO_ticker_df[equity_idx_PERMNO_ticker_df["ticker"]=="AAPL"].iloc[0, 0]
equity_idx_2 = equity_idx_PERMNO_ticker_df[equity_idx_PERMNO_ticker_df["ticker"]=="MSFT"].iloc[0, 0]
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.capitalization[equity_idx_1, t_idx])
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.capitalization[equity_idx_2, t_idx])

#%%
rank_idx = 500
result = equity_data_high_freq_.intraday_data(datetime.datetime(2010, 5, 26))
time = result["time"]; capitalization = result["capitalization"]; rank = result["rank"]; equity_idx_by_rank = result["equity_idx_by_rank"]
plt.plot(time, capitalization[equity_idx_by_rank[rank_idx,0].astype(int), :])
plt.plot(time, capitalization[equity_idx_by_rank[rank_idx+1,0].astype(int), :])

#%%
t_idx = np.arange(np.searchsorted(equity_data_.time_axis, datetime.datetime(2010,4,1)), np.searchsorted(equity_data_.time_axis, datetime.datetime(2010,8,31)))
equity_idx_1 = equity_idx_PERMNO_ticker_df[equity_idx_PERMNO_ticker_df["ticker"]=="AAPL"].iloc[0, 0]
equity_idx_2 = equity_idx_PERMNO_ticker_df[equity_idx_PERMNO_ticker_df["ticker"]=="MSFT"].iloc[0, 0]
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.capitalization[equity_idx_1, t_idx])
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.capitalization[equity_idx_2, t_idx])
plt.savefig(os.path.join(os.path.dirname(__file__), "stock_cap.pdf"), dpi=300)

#%%
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.rank[equity_idx_1, t_idx])
plt.plot([equity_data_.time_axis[j] for j in t_idx], equity_data_.rank[equity_idx_2, t_idx])
plt.savefig(os.path.join(os.path.dirname(__file__), "stock_rank.pdf"), dpi=300)

#%%
plt.bar([equity_data_.time_axis[j] for j in t_idx], equity_data_.return_[equity_idx_1, t_idx], width=2.0)
plt.bar([equity_data_.time_axis[j] for j in t_idx], equity_data_.return_[equity_idx_2, t_idx], width=2.0)
plt.savefig(os.path.join(os.path.dirname(__file__), "stock_return.pdf"), dpi=300)

#%%
matrix = np.zeros((2, 2))
matrix[0, 0] = 1; matrix[1, 1] = 1
plt.figure(figsize=(8, 8))
plt.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
plt.xticks([]); plt.yticks([])
cbar = plt.colorbar()
cbar.set_ticks([0, 1])
plt.savefig(os.path.join(os.path.dirname(__file__), "identity_matrix.pdf"), dpi=300)


#%%
matrix = np.zeros((2, 2))
matrix[0, 1] = 1; matrix[1, 0] = 1
plt.figure(figsize=(8, 8))
plt.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
plt.xticks([]); plt.yticks([])
cbar = plt.colorbar()
cbar.set_ticks([0, 1])
plt.savefig(os.path.join(os.path.dirname(__file__), "permutation_matrix.pdf"), dpi=300)

#%%
prop = np.sum(equity_data_.rank[equity_idx_1, t_idx].astype(int) == 1)/len(t_idx)
matrix = np.zeros((2, 2))
matrix[0, 0] = prop; matrix[1, 1] = prop
matrix[0, 1] = 1-prop; matrix[1, 0] = 1-prop
plt.figure(figsize=(8, 8))
plt.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
plt.xticks([]); plt.yticks([])
cbar = plt.colorbar()
cbar.set_ticks([0, 1])
plt.savefig(os.path.join(os.path.dirname(__file__), "theta.pdf"), dpi=300)

# %%
x = np.linspace(-10, 10, 1000)
plt.plot(x, 1/(1+np.exp(-x)))
plt.savefig(os.path.join(os.path.dirname(__file__), "sigmoid.pdf"), dpi=300)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Ornstein-Uhlenbeck process
theta = 0.15  # Speed of reversion
mu = 0.0  # Long-term mean
sigma = 0.3  # Volatility
dt = 0.01  # Time step
T = 10.0  # Total time
n = int(T/dt)  # Number of time steps

# Generate the Ornstein-Uhlenbeck process
np.random.seed(0)
x = np.zeros(n)
for t in range(n-1):
    x[t+1] = x[t] + theta*(mu - x[t])*dt + sigma*np.sqrt(dt)*np.random.normal()

# Plot the process
#%%
plt.figure(figsize=(9, 9))
plt.subplot(3,1,1)
plt.plot(range(len(x)), x)
plt.subplot(3,1,2)
kernel = [1,0,2]
x_smooth = np.convolve(x, kernel, 'same')
plt.plot(range(len(x)), x_smooth)
plt.subplot(3,1,3)
kernel = [2,0,-1]
x_smooth = np.convolve(x, kernel, 'same')
plt.plot(range(len(x)), x_smooth)
plt.savefig(os.path.join(os.path.dirname(__file__), "OU_process_convolve.pdf"), dpi=300)


#%%
import matplotlib.pyplot as plt
import numpy as np

# Create an array with positive and negative values
values = np.array([0, 1, 0, -1, 1, -1, 1, 0, -1, -1])
#values = np.array([1.5, -2.3, 3.4, -1.2, 0.8, -0.9, 1.1, -1.3, 2.2, -0.7])

# Create an array for the x positions of the bars
x_pos = np.arange(len(values))

# Create a color array
colors = ['red' if value >= 0 else 'blue' for value in values]

# Create the bar plot
plt.bar(x_pos, values, color=colors, alpha=0.7)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar plot for an array with positive and negative values')
plt.savefig(os.path.join(os.path.dirname(__file__), "bar_plot.pdf"), dpi=300)

#%% simulate a stock price
import numpy as np
import matplotlib.pyplot as plt
# Parameters for the stock price simulation
#seed = np.random.choice(10000)
seed = 29
np.random.seed(seed)
mu = 0.05  # Drift
sigma = 0.3  # Volatility
dt = 0.01  # Time step
T = 1.0  # Total time
n = int(T/dt)  # Number of time steps

# Generate the stock price trajectories
price_1 = np.zeros(n)
price_2 = np.zeros(n)
price_1[0] = 130.0
price_2[0] = 150.0
for t in range(n-1):
    price_1[t+1] = price_1[t] * (1 + mu*dt + sigma*np.sqrt(dt)*np.random.normal())
    price_2[t+1] = price_2[t] * (1 + mu*dt + sigma*np.sqrt(dt)*np.random.normal())

price_rank_1 = np.maximum(price_1, price_2)
price_rank_2 = np.minimum(price_1, price_2)
rebalance_point = [30, 60, 90]
w_rank_1 = 1; w_rank_2 = 0.6
w_name_1 = 1; w_name_2 = 0.6
w_rank_1_hist = [w_rank_1]; w_rank_2_hist = [w_rank_2]
w_name_1_hist = [w_name_1]; w_name_2_hist = [w_name_2]
transaction_cost = [0]; maintainence_cost = [0]

for t in range(1, n):
    if t in rebalance_point:
        w_rank_1 *= price_rank_1[t]/price_rank_1[t-1]
        w_rank_2 *= price_rank_2[t]/price_rank_2[t-1]
        w_name_1_artificial = w_name_1*price_1[t]/price_1[t-1]
        w_name_2_artificial = w_name_2*price_2[t]/price_2[t-1]
        if price_1[t] == price_rank_1[t]:
            w_name_1 = w_rank_1
            w_name_2 = w_rank_2
        else:
            w_name_1 = w_rank_2
            w_name_2 = w_rank_1

        transaction_cost.append(0.0002*np.abs(w_name_1-w_name_1_artificial)+0.0002*np.abs(w_name_2-w_name_2_artificial))
        maintainence_cost.append(w_name_1+w_name_2-w_name_1_artificial-w_name_2_artificial)
    else:
        w_rank_1 *= price_rank_1[t]/price_rank_1[t-1]
        w_rank_2 *= price_rank_2[t]/price_rank_2[t-1]
        w_name_1 *= price_1[t]/price_1[t-1]
        w_name_2 *= price_2[t]/price_2[t-1]
        transaction_cost.append(0); maintainence_cost.append(0)
    w_rank_1_hist.append(w_rank_1); w_rank_2_hist.append(w_rank_2)
    w_name_1_hist.append(w_name_1); w_name_2_hist.append(w_name_2)

PnL = np.array(w_name_1_hist)+np.array(w_name_2_hist)-np.cumsum(transaction_cost)-np.cumsum(maintainence_cost)
color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
nrow = 6; ncol = 1
plt.figure(figsize=(10, 10))
plt.subplot(nrow,ncol,1)
plt.plot(range(len(price_1)), price_1, label=r'$c_1$', color=color_cycle[0])
plt.plot(range(len(price_2)), price_2, label=r'$c_2$', color=color_cycle[1])
plt.ylim([120, 200]); plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

plt.subplot(nrow,ncol,ncol+1)
plt.plot(range(len(price_1)), np.maximum(price_1, price_2), label=r'$c_{(1)}$', color=color_cycle[2], alpha=0.5)
plt.plot(range(len(price_1)), np.minimum(price_1, price_2), label=r'$c_{(2)}$', color=color_cycle[3], alpha=0.5)
plt.ylim([120, 200]); plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

plt.subplot(nrow,ncol,2*ncol+1)
plt.plot(range(len(price_1)), w_name_1_hist, label=r'$w_1$', color=color_cycle[0])
plt.plot(range(len(price_1)), w_name_2_hist, label=r'$w_2$', color=color_cycle[1])
plt.plot(range(len(price_1)), w_rank_1_hist, label=r'$w_{(1)}$', color=color_cycle[2], alpha=0.5)
plt.plot(range(len(price_1)), w_rank_2_hist, label=r'$w_{(2)}$', color=color_cycle[3], alpha=0.5)
plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

plt.subplot(nrow,ncol,3*ncol+1)
plt.plot(range(len(price_1)), np.cumsum(maintainence_cost), label='M.C.', color=color_cycle[1])
plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

plt.subplot(nrow,ncol,4*ncol+1)
plt.plot(range(len(price_1)), np.cumsum(transaction_cost), label='T.C.', color=color_cycle[0])
plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

plt.subplot(nrow,ncol,5*ncol+1)
plt.plot(range(len(price_1)), PnL, label='PnL', color=color_cycle[0])
plt.xlim(-5, 105)
plt.vlines(rebalance_point,  plt.ylim()[0],  plt.ylim()[1], color='gray', linestyle='--', alpha=0.5)
plt.xticks([], []); plt.yticks([], [])
plt.legend()

#%% local time
# create two brownian motion
# Parameters
mu = 0.0  # Drift
sigma = 1.0  # Volatility
dt = 0.1  # Time step
T = 100.0  # Total time
n = int(T / dt)  # Number of time steps

# Initialize random seed
seed = 1
np.random.seed(seed)

# Generate random increments for two Brownian motions
increments_1 = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
increments_2 = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)

# Calculate the Brownian motion paths
path_1 = 2+np.cumsum(increments_1)  # Cumulative sum for path 1
path_2 = np.cumsum(increments_2)  # Cumulative sum for path 2

# calculate the local time
diff = np.abs(path_1 - path_2)
local_time = np.cumsum(diff < 0.1) * dt

# Plotting
color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure(figsize=(8, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()
# Plot the Brownian motion paths on the primary y-axis
ax1.plot(path_1, label=r'$c_1$', color=color_cycle[0])
ax1.plot(path_2, label=r'$c 2$', color=color_cycle[1])
#ax1.set_ylabel('Capitalization', color=color_cycle[0])
ax1.tick_params(axis='y', labelcolor=color_cycle[0], length=0, label1On=False, label2On=False)

# Plot the local time on the secondary y-axis
ax2.plot(local_time, label=r'$\Lambda(t)$', color='r', alpha=0.5)
#ax2.set_ylabel('Local Time', color='r')
ax2.tick_params(axis='y', labelcolor='r', length=0, label1On=False, label2On=False)

# Set x-axis label
ax1.set_xlabel('Time')
ax1.tick_params(axis='x', length=0, label1On=False, label2On=False)
# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "local_time_schematic_twin_axis.pdf"), dpi=300)

plt.show()

#%%


plt.subplot(2, 1, 1)
plt.plot(path_1)
plt.plot(path_2)
plt.xticks([], []); plt.yticks([], [])
plt.subplot(2, 1, 2)    
plt.plot(local_time)
plt.xticks([], []); plt.yticks([], [])
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "local_time_schematic.pdf"), dpi=300)


#%%


# %%
