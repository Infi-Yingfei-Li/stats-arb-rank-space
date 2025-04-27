#%%
import os, sys, copy, h5py, datetime, tqdm
import numpy as np
import matplotlib.pyplot as plt

#%%
g1 = 0.2; g2 = -0.2
gamma_1 = -0.5; gamma_2 = 0.5
sigma_1 = 0.2; sigma_2 = 0.2

t = np.linspace(0, 1, 101)
dt = t[1] - t[0]
Y1 = 0; Y2 = 0
Z1 = 0; Z2 = 0
WY1 = 0; WY2 = 0
WZ1 = 0; WZ2 = 0

Y1_ar = np.zeros(len(t)); Y2_ar = np.zeros(len(t))
Z1_ar = np.zeros(len(t)); Z2_ar = np.zeros(len(t))
WY1_ar = np.zeros(len(t)); WY2_ar = np.zeros(len(t))
WZ1_ar = np.zeros(len(t)); WZ2_ar = np.zeros(len(t))

#%%
for i in range(1, len(t), 1):
    Y1 += gamma_1*dt + sigma_1*np.random.normal(0, dt)
    Y2 += gamma_2*dt + sigma_2*np.random.normal(0, dt)
    Z1 = max(Y1, Y2)
    Z2 = min(Y1, Y2)


