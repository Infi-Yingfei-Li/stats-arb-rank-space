#%%
import os, sys, copy, h5py, datetime, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def OU_process():
    theta = 10  # Rate of reversion
    mu = 0.0     # Mean
    sigma = 0.1  # Volatility
    dt = 0.005    # Time step
    T = 1        # Total time
    n = int(T / dt)  # Number of time steps
    times = np.linspace(0, T, n)

    # Initial condition
    X0 = 0

    # Simulate the process
    X = np.zeros(n)
    X[0] = X0
    for t in range(1, n):
        dW = np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW
    return (times, X)

plt.figure(figsize=(10, 6))
for count in range(5):
    times, X = OU_process()
    plt.plot(times, X)
plt.xticks([]); plt.yticks([])
plt.savefig(os.path.join(os.path.dirname(__file__), 'OU_process.pdf'), dpi=300)




# %%
