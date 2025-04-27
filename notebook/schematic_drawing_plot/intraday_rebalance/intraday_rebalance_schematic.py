#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import data.data as data

#%% non-volatile case
time_axis = np.linspace(0, 1, 1000)
cap_1 = 1/(1+np.exp(-10*(time_axis-0.6)))+0.02
cap_2 = 0.75/(1+np.exp(10*(time_axis-0.6)))
cap_1_rank = np.maximum(cap_1, cap_2)
cap_2_rank = np.minimum(cap_1, cap_2)

rebalance_point = [np.searchsorted(time_axis, 0.35), np.searchsorted(time_axis, 0.7)]
pw_1 = [0.01]; pw_2 = [1]
pw_1_rank = [pw_2[0]]; pw_2_rank = [pw_1[0]]
transaction_cost = [0]
for j in np.arange(1, len(cap_1), 1):
    if j in rebalance_point:
        pw_1_rank.append(pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1]))
        pw_2_rank.append(pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1]))
        if cap_1[j] >= cap_2[j]:
            pw_1.append(pw_1_rank[-1])
            pw_2.append(pw_2_rank[-1])
        else:
            pw_1.append(pw_2_rank[-1])
            pw_2.append(pw_1_rank[-1])
        transaction_cost.append(0.0002*(np.abs(pw_1[-1]-pw_1[-2])+np.abs(pw_2[-1]-pw_2[-2])))
    else:
        pw_1_new = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_new = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        pw_1_rank_new = pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1])
        pw_2_rank_new = pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1])
        pw_1.append(pw_1_new); pw_2.append(pw_2_new)
        pw_1_rank.append(pw_1_rank_new); pw_2_rank.append(pw_2_rank_new)
        transaction_cost.append(0)

difference = np.array(pw_1_rank) + np.array(pw_2_rank) - np.array(pw_1) - np.array(pw_2)
plt.figure(figsize=(3, 10))
plt.subplot(5,1,1)
plt.plot(time_axis, cap_1, label='c1')
plt.plot(time_axis, cap_2, label='c2')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', linestyles="--", alpha=0.5)
plt.xlim([0, 0.8])
plt.xticks([]); plt.yticks([])
plt.subplot(5,1,2)
plt.plot(time_axis, pw_1, label='p1')
plt.plot(time_axis, pw_2, label='p2')
plt.plot(time_axis, pw_1_rank, label='p(1)', linewidth=5, color='red', alpha=0.5)
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', linestyles="--", alpha=0.5)
plt.xlim([0, 0.8])
plt.xticks([]); plt.yticks([])
plt.subplot(5,1,3)
plt.plot(time_axis, pw_1, label='p1')
plt.plot(time_axis, pw_2, label='p2')
plt.plot(time_axis, pw_2_rank, label='p(2)', linewidth=5, color='green', alpha=0.5)
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', linestyles="--", alpha=0.5)
plt.xlim([0, 0.8])
plt.xticks([]); plt.yticks([])
plt.subplot(5,1,4)
plt.plot(time_axis, difference, label='latency cost')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', linestyles="--", alpha=0.5)
plt.xlim([0, 0.8])
plt.xticks([]); plt.yticks([])
plt.subplot(5,1,5)
plt.plot(time_axis, np.cumsum(transaction_cost), label='transaction cost')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', linestyles="--", alpha=0.5)
plt.xlim([0, 0.8])
plt.xticks([]); plt.yticks([])
plt.savefig(os.path.join(os.path.dirname(__file__), 'intraday_scehamtic_trajectory.pdf'), dpi=300)


#%%
seed = 6
np.random.seed(seed)
time_axis = np.linspace(0, 1, 100)
cap_1 = 1/(1+np.exp(-10*(time_axis-0.5)))+0.02+np.random.normal(0, 0.2, len(time_axis)) + 1
cap_2 = 0.75/(1+np.exp(10*(time_axis-0.5)))+np.random.normal(0, 0.2, len(time_axis)) + 1
cap_1_rank = np.maximum(cap_1, cap_2)
cap_2_rank = np.minimum(cap_1, cap_2)

rebalance_point = [np.searchsorted(time_axis, 0.05), np.searchsorted(time_axis, 0.95)]
pw_1 = [0.1]; pw_2 = [1]
pw_1_rank = [pw_2[0]]; pw_2_rank = [pw_1[0]]
transaction_cost = [0]; latency_cost = [0]
for j in np.arange(1, len(cap_1), 1):
    if j in rebalance_point:
        pw_1_rank.append(pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1]))
        pw_2_rank.append(pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1]))
        pw_1_img = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_img = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        if cap_1[j] >= cap_2[j]:
            pw_1.append(pw_1_rank[-1])
            pw_2.append(pw_2_rank[-1])
        else:
            pw_1.append(pw_2_rank[-1])
            pw_2.append(pw_1_rank[-1])
        transaction_cost.append(0.0002*(np.abs(pw_1[-1]-pw_1[-2])+np.abs(pw_2[-1]-pw_2[-2])))
        latency_cost.append(pw_1[-1]-pw_1_img+pw_2[-1]-pw_2_img)
    else:
        pw_1_new = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_new = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        pw_1_rank_new = pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1])
        pw_2_rank_new = pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1])
        pw_1.append(pw_1_new); pw_2.append(pw_2_new)
        pw_1_rank.append(pw_1_rank_new); pw_2_rank.append(pw_2_rank_new)
        transaction_cost.append(0); latency_cost.append(0)
difference = np.array(pw_1_rank) + np.array(pw_2_rank) - np.array(pw_1) - np.array(pw_2)
plt.figure(figsize=(17, 16))
plt.subplot(5,3,1)
plt.plot(time_axis, cap_1, label='c1')
plt.plot(time_axis, cap_2, label='c2')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,4)
plt.plot(time_axis, difference, label='low')
plt.ylim([-0.1, 0.9])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,7)
plt.plot(time_axis, np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,10)
plt.plot(time_axis, np.cumsum(transaction_cost), label='low')
plt.ylim([-0.0005, 0.0035])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,13)
plt.plot(time_axis, np.cumsum(transaction_cost)+np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)

rebalance_time = np.arange(0.2, 1, 0.2)
rebalance_point = [np.searchsorted(time_axis, j) for j in rebalance_time]
pw_1 = [0.1]; pw_2 = [1]
pw_1_rank = [pw_2[0]]; pw_2_rank = [pw_1[0]]
transaction_cost = [0]; latency_cost = [0]
for j in np.arange(1, len(cap_1), 1):
    if j in rebalance_point:
        pw_1_rank.append(pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1]))
        pw_2_rank.append(pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1]))
        pw_1_img = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_img = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        if cap_1[j] >= cap_2[j]:
            pw_1.append(pw_1_rank[-1])
            pw_2.append(pw_2_rank[-1])
        else:
            pw_1.append(pw_2_rank[-1])
            pw_2.append(pw_1_rank[-1])
        transaction_cost.append(0.0002*(np.abs(pw_1[-1]-pw_1[-2])+np.abs(pw_2[-1]-pw_2[-2])))
        latency_cost.append(pw_1[-1]-pw_1_img+pw_2[-1]-pw_2_img)
    else:
        pw_1_new = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_new = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        pw_1_rank_new = pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1])
        pw_2_rank_new = pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1])
        pw_1.append(pw_1_new); pw_2.append(pw_2_new)
        pw_1_rank.append(pw_1_rank_new); pw_2_rank.append(pw_2_rank_new)
        transaction_cost.append(0); latency_cost.append(0)
difference = np.array(pw_1_rank) + np.array(pw_2_rank) - np.array(pw_1) - np.array(pw_2)

plt.subplot(5,3,2)
plt.plot(time_axis, cap_1, label='c1')
plt.plot(time_axis, cap_2, label='c2')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,5)
plt.plot(time_axis, difference, label='low')
plt.ylim([-0.1, 0.9])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,8)
plt.plot(time_axis, np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,11)
plt.plot(time_axis, np.cumsum(transaction_cost), label='low')
plt.ylim([-0.0005, 0.0035])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,14)
plt.plot(time_axis, np.cumsum(transaction_cost)+np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)

rebalance_time = np.arange(.02, 1, 0.02)
rebalance_point = [np.searchsorted(time_axis, j) for j in rebalance_time]
pw_1 = [0.1]; pw_2 = [1]
pw_1_rank = [pw_2[0]]; pw_2_rank = [pw_1[0]]
transaction_cost = [0]; latency_cost = [0]
for j in np.arange(1, len(cap_1), 1):
    if j in rebalance_point:
        pw_1_rank.append(pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1]))
        pw_2_rank.append(pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1]))
        pw_1_img = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_img = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        if cap_1[j] >= cap_2[j]:
            pw_1.append(pw_1_rank[-1])
            pw_2.append(pw_2_rank[-1])
        else:
            pw_1.append(pw_2_rank[-1])
            pw_2.append(pw_1_rank[-1])
        transaction_cost.append(0.0002*(np.abs(pw_1[-1]-pw_1[-2])+np.abs(pw_2[-1]-pw_2[-2])))
        latency_cost.append(pw_1[-1]-pw_1_img+pw_2[-1]-pw_2_img)
    else:
        pw_1_new = pw_1[-1]*(cap_1[j]/cap_1[j-1])
        pw_2_new = pw_2[-1]*(cap_2[j]/cap_2[j-1])
        pw_1_rank_new = pw_1_rank[-1]*(cap_1_rank[j]/cap_1_rank[j-1])
        pw_2_rank_new = pw_2_rank[-1]*(cap_2_rank[j]/cap_2_rank[j-1])
        pw_1.append(pw_1_new); pw_2.append(pw_2_new)
        pw_1_rank.append(pw_1_rank_new); pw_2_rank.append(pw_2_rank_new)
        transaction_cost.append(0); latency_cost.append(0)
difference = np.array(pw_1_rank) + np.array(pw_2_rank) - np.array(pw_1) - np.array(pw_2)

plt.subplot(5,3,3)
plt.plot(time_axis, cap_1, label='c1')
plt.plot(time_axis, cap_2, label='c2')
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,6)
plt.plot(time_axis, difference, label='low')
plt.ylim([-0.1, 0.9])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,9)
plt.plot(time_axis, np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,12)
plt.plot(time_axis, np.cumsum(transaction_cost), label='low')
plt.ylim([-0.0005, 0.0035])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.subplot(5,3,15)
plt.plot(time_axis, np.cumsum(transaction_cost)+np.cumsum(latency_cost), label='low')
plt.ylim([-0.1, 1.1])
plt.vlines([time_axis[j] for j in rebalance_point], plt.ylim()[0], plt.ylim()[1], color='red', alpha=0.2)
plt.tick_params(direction='in', axis='both', which='both', top=False,  right=False)
plt.savefig(os.path.join(os.path.dirname(__file__), 'schematic_intraday_frequency.pdf'), dpi=300)


# %%
