#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

#%%
two_stock_model_config = {"mu_1": 0.01, "mu_2": 0.015, "sigma_1": 0.075, "sigma_2": 0.1, "gamma_1": -0.035, "r": 0.001, 
                          "g": lambda x: 0, "Delta": 0.01, "alpha": 0.005, 
                          "S_1": 105, "S_2": 100, "W_Z_1": 1, "W_Z_2": 0.8, "Q": 0,
                          "delta_t": 1/(7*60)}
'''
two_stock_model_config = {"mu_1": 0.01, "mu_2": 0.015, "sigma_1": 0.075, "sigma_2": 0.1, "gamma_1": -0.035, "r": 0.001, 
                          "g": lambda x: 0, "Delta": 0.01, "alpha": 0.005, 
                          "S_1": 105, "S_2": 100, "W_Z_1": 1, "W_Z_2": 0.8, "Q": 0,
                          "delta_t": 1/20}
'''

class two_stock_model:
    def __init__(self, two_stock_model_config):
        self.config = two_stock_model_config
        self.t = 0; self.delta_t = self.config["delta_t"]
        self.mu_1 = self.config["mu_1"]; self.mu_2 = self.config["mu_2"]
        self.sigma_1 = self.config["sigma_1"]; self.sigma_2 = self.config["sigma_2"]
        self.gamma_1 = self.config["gamma_1"]; self.gamma_2 = -self.mu_1-self.mu_2-self.gamma_1
        self.r = self.config["r"]
        self.g = self.config["g"]
        def f(x):
            if x == 0:
                return 0
            return (self.config["Delta"]/2 if x > 0 else -self.config["Delta"]/2) + self.config["alpha"]*x
        self.f = f
        self.Delta = self.config["Delta"]; self.alpha = self.config["alpha"]
        self.S_1 = self.config["S_1"]; self.S_2 = self.config["S_2"]
        self.Z_1 = max(self.S_1, self.S_2); self.Z_2 = min(self.S_1, self.S_2)
        self.W_Z_1 = self.config["W_Z_1"]; self.W_Z_2 = self.config["W_Z_2"]
        self.W_Y_1 = copy.deepcopy(self.W_Z_1); self.W_Y_2 = copy.deepcopy(self.W_Z_2)
        self.W_Y_1_tar = copy.deepcopy(self.W_Z_1 if self.S_1>=self.S_2 else self.W_Z_2)
        self.W_Y_2_tar = copy.deepcopy(self.W_Z_2 if self.S_1>=self.S_2 else self.W_Z_1)
        self.Q = self.config["Q"]

        self.t_hist = [copy.deepcopy(self.t)]
        self.S_1_hist = [copy.deepcopy(self.S_1)]; self.S_2_hist = [copy.deepcopy(self.S_2)]
        self.Z_1_hist = [copy.deepcopy(self.Z_1)]; self.Z_2_hist = [copy.deepcopy(self.Z_2)]
        self.W_Z_1_hist = [copy.deepcopy(self.W_Z_1)]; self.W_Z_2_hist = [copy.deepcopy(self.W_Z_2)]
        self.W_Y_1_hist = [copy.deepcopy(self.W_Y_1)]; self.W_Y_2_hist = [copy.deepcopy(self.W_Y_2)]
        self.W_Y_1_tar_hist = [copy.deepcopy(self.W_Y_1_tar)]; self.W_Y_2_tar_hist = [copy.deepcopy(self.W_Y_2_tar)]
        self.Q_hist = [copy.deepcopy(self.Q)]
        self.q_1_hist = []; self.q_2_hist = []

    def step(self, q_1, q_2):
        if self.t > 1:
            raise ValueError("Trading period is over")
        
        self.t += self.delta_t
        self.S_1 += (self.mu_1*self.S_1 + (self.gamma_1 if self.S_1 >= self.S_2 else self.gamma_2)*self.S_1 + self.g(q_1))*self.delta_t + self.sigma_1*self.S_1*np.sqrt(self.delta_t)*np.random.normal()
        self.S_2 += (self.mu_2*self.S_2 + (self.gamma_2 if self.S_1 >= self.S_2 else self.gamma_1)*self.S_2 + self.g(q_2))*self.delta_t + self.sigma_2*self.S_2*np.sqrt(self.delta_t)*np.random.normal()
        self.Z_1 = max(self.S_1, self.S_2); self.Z_2 = min(self.S_1, self.S_2)
        self.W_Z_1 += self.W_Z_1*(self.Z_1-self.Z_1_hist[-1])/self.Z_1_hist[-1]
        self.W_Z_2 += self.W_Z_2*(self.Z_2-self.Z_2_hist[-1])/self.Z_2_hist[-1]
        self.W_Y_1 += self.W_Y_1*(self.S_1-self.S_1_hist[-1])/self.S_1_hist[-1] + q_1*self.S_1_hist[-1]*self.delta_t
        self.W_Y_2 += self.W_Y_2*(self.S_2-self.S_2_hist[-1])/self.S_2_hist[-1] + q_2*self.S_2_hist[-1]*self.delta_t
        self.W_Y_1_tar = self.W_Z_1 if self.S_1>=self.S_2 else self.W_Z_2
        self.W_Y_2_tar = self.W_Z_2 if self.S_1>=self.S_2 else self.W_Z_1
        self.Q += self.r*self.Q*self.delta_t - q_1*(self.S_1_hist[-1] + self.f(q_1))*self.delta_t - q_2*(self.S_2_hist[-1] + self.f(q_2))*self.delta_t
        
        self.t_hist.append(copy.deepcopy(self.t))
        self.S_1_hist.append(copy.deepcopy(self.S_1)); self.S_2_hist.append(copy.deepcopy(self.S_2))
        self.Z_1_hist.append(copy.deepcopy(self.Z_1)); self.Z_2_hist.append(copy.deepcopy(self.Z_2))
        self.W_Z_1_hist.append(copy.deepcopy(self.W_Z_1)); self.W_Z_2_hist.append(copy.deepcopy(self.W_Z_2))
        self.W_Y_1_hist.append(copy.deepcopy(self.W_Y_1)); self.W_Y_2_hist.append(copy.deepcopy(self.W_Y_2))
        self.W_Y_1_tar_hist.append(copy.deepcopy(self.W_Y_1_tar)); self.W_Y_2_tar_hist.append(copy.deepcopy(self.W_Y_2_tar))
        self.Q_hist.append(copy.deepcopy(self.Q))
        self.q_1_hist.append(copy.deepcopy(q_1)); self.q_2_hist.append(copy.deepcopy(q_2))

        if self.t >= 1:
            self.q_1_hist.append(0); self.q_2_hist.append(0)
            term_1 = ((self.W_Z_1 if self.S_1>=self.S_2 else self.W_Z_2) - self.W_Y_1)/self.S_1
            term_2 = ((self.W_Z_2 if self.S_1>=self.S_1 else self.W_Z_1) - self.W_Y_2)/self.S_2
            self.U = self.Q - term_1*(self.S_1 + self.f(term_1)) - term_2*(self.S_2 + self.f(term_2))

    def plot(self):
        plt.figure(figsize=(10, 10))
        nrow = 6
        plt.subplot(nrow, 1, 1)
        plt.plot(self.t_hist, self.S_1_hist, label=r'$S_{1,t}$', color="orange")
        plt.plot(self.t_hist, self.S_2_hist, label=r'$S_{2,t}$', color='green')
        plt.legend(); plt.ylabel("stock price")

        plt.subplot(nrow, 1, 2)
        plt.plot(self.t_hist, self.W_Z_1_hist, label=r'$W_{(1)}$', color="red")
        plt.plot(self.t_hist, self.W_Z_2_hist, label=r'$W_{(2)}$', color="blue")
        plt.legend()
        plt.ylabel("wealth (rank space)")

        plt.subplot(nrow, 1, 3)
        plt.plot(self.t_hist, self.q_1_hist, label=r'$q_{1,t}$', color="orange")
        plt.plot(self.t_hist, self.q_2_hist, label=r'$q_{2,t}$', color='green')
        plt.legend()
        plt.ylabel("trading rate")

        plt.subplot(nrow, 1, 4)
        plt.plot(self.t_hist, self.W_Y_1_hist, label=r'$W_{1}^{Y}$', color="orange")
        plt.plot(self.t_hist, self.W_Y_1_tar_hist, label=r'$W_{1}^{Y, tar}$', color='black', linestyle='--')
        plt.legend()
        plt.ylabel("wealth (name space)")
        
        plt.subplot(nrow, 1, 5)
        plt.plot(self.t_hist, self.W_Y_2_hist, label=r'$W_{2}^{Y}$', color="green")
        plt.plot(self.t_hist, self.W_Y_2_tar_hist, label=r'$W_{2}^{Y, tar}$', color='black', linestyle='--')
        plt.legend()
        plt.ylabel("wealth (name space)")
        
        plt.subplot(nrow, 1, 6)
        plt.plot(self.t_hist, self.Q_hist, label=r'$Q_t$', color="black")
        plt.legend()
        plt.tight_layout()

#%% strategy 1: no trading
model = two_stock_model(two_stock_model_config)
while model.t <= 1:
    model.step(0, 0)

model.plot()
model.U


# %%
