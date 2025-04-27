#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

#%% theory derivation
X_t, W_t, Y_t, Z_t = sp.symbols('X_t W_t Y_t Z_t')
mu, sigma = sp.symbols('mu sigma')
pi_t, r = sp.symbols('pi_t r')
kappa_Y, sigma_Y = sp.symbols('kappa_Y sigma_Y')
kappa_Z, sigma_Z = sp.symbols('kappa_Z sigma_Z')
gamma = sp.symbols('gamma')
t, w, x, y, z = sp.symbols('t w x y z')
T = sp.symbols('T')

h = sp.Function('h')(t, x, y, z)
U = -sp.exp(-gamma*W_t)
H = -sp.exp(-gamma*(w*sp.exp(r*(T-t))+h))

partial_t_H = sp.diff(H, t); partial_w_H = sp.diff(H, w)
partial_x_H = sp.diff(H, x); partial_y_H = sp.diff(H, y); partial_z_H = sp.diff(H, z)
partial_ww_H = sp.diff(H, w, w); partial_xx_H = sp.diff(H, x, x); partial_yy_H = sp.diff(H, y, y); partial_zz_H = sp.diff(H, z, z)
D_H = (mu-r)*partial_w_H + sigma*sigma*sp.diff(H, w, x)

expression = partial_t_H + partial_x_H*mu - partial_y_H*kappa_Y*y - partial_z_H*kappa_Z*z\
+ 0.5*sigma*sigma*partial_xx_H + 0.5*sigma_Y*sigma_Y*partial_yy_H + 0.5*sigma_Z*sigma_Z*partial_zz_H\
+ partial_w_H*w*r - 0.5*D_H*D_H/(partial_ww_H*sigma*sigma)
expression_2 = sp.simplify(expression/(-gamma*H))
#expression_2

#%%
gamma = 0.1; r = 0.001
kappa_Y = 1/0.01; sigma_Y = 1
kappa_Z = 1/0.10; sigma_Z = 1
t_min = 0; t_max = 1; t_len = 1001

model_config = {"kappa_Y": kappa_Y, "kappa_Z": kappa_Z, "sigma_Y": sigma_Y, "sigma_Z": sigma_Z, "gamma": gamma, "r": r,
              "t_min": t_min, "t_max": t_max, "t_len": t_len}

class model:
    def __init__(self, model_config):
        self.config = model_config
        self.dt = (self.config["t_max"]-self.config["t_min"])/(self.config["t_len"]-1)

        self.t = 0; self.X_t = 0; self.W_t = 0; self.Y_t = 0; self.Z_t = 0
        self.t_hist = [0]; self.X_t_hist = [0]; self.W_t_hist = [0]; self.Y_t_hist = [0]; self.Z_t_hist = [0]
        self.pi_hist = []
    
    def simulate_trajectory(self, pi_t, is_plot=False):
        # simulate first 100*t_len time steps for stochastic equilibration starting point
        X_t = 0; W_t = 0; Y_t = 0; Z_t = 0
        '''
        for _ in range(10*t_len):
            dY_t = -self.config["kappa_Y"]*Y_t*self.dt + self.config["sigma_Y"]*np.sqrt(self.dt)*np.random.normal(0, 1)
            dZ_t = -self.config["kappa_Z"]*Z_t*self.dt + self.config["sigma_Z"]*np.sqrt(self.dt)*np.random.normal(0, 1)
            dX_t = self.mu(Y_t, Z_t)*self.dt + self.sigma(Y_t, Z_t)*np.sqrt(self.dt)*np.random.normal(0, 1)
            Y_t = Y_t + dY_t ; Z_t = Z_t + dZ_t; X_t = X_t + dX_t
        '''
        t_hist = [0]; W_t_hist = [W_t]; X_t_hist = [X_t]; Y_t_hist = [Y_t]; Z_t_hist = [Z_t]
        for t in np.arange(1, self.config["t_len"], 1):
            dY_t = -self.config["kappa_Y"]*Y_t*self.dt + self.config["sigma_Y"]*np.sqrt(self.dt)*np.random.normal(0, 1)
            dZ_t = -self.config["kappa_Z"]*Z_t*self.dt + self.config["sigma_Z"]*np.sqrt(self.dt)*np.random.normal(0, 1)
            dX_t = self.mu(Y_t, Z_t)*self.dt + self.sigma(Y_t, Z_t)*np.sqrt(self.dt)*np.random.normal(0, 1)
            dW_t = pi_t[t-1]*dX_t + (W_t - pi_t[t-1])*self.config["r"]*self.dt
            Y_t = Y_t + dY_t ; Z_t = Z_t + dZ_t; X_t = X_t + dX_t; W_t = W_t + dW_t
            t_hist.append(t*self.dt); W_t_hist.append(W_t); X_t_hist.append(X_t); Y_t_hist.append(Y_t); Z_t_hist.append(Z_t)

        if is_plot:
            plt.plot(t_hist, Y_t_hist, label="Y_t")
            plt.plot(t_hist, Z_t_hist, label="Z_t")
            plt.plot(t_hist, X_t_hist, label="X_t")
            #plt.plot(t_hist, W_t_hist, label="W_t")
            plt.legend()

    def simulate_step(self, pi_t):
        self.pi_hist.append(pi_t)
        dY_t = -self.config["kappa_Y"]*self.Y_t*self.dt + self.config["sigma_Y"]*np.sqrt(self.dt)*np.random.normal(0, 1)
        dZ_t = -self.config["kappa_Z"]*self.Z_t*self.dt + self.config["sigma_Z"]*np.sqrt(self.dt)*np.random.normal(0, 1)
        dX_t = self.mu(self.Y_t, self.Z_t)*self.dt + self.sigma(self.Y_t, self.Z_t)*np.sqrt(self.dt)*np.random.normal(0, 1)
        dW_t = pi_t*self.dt + (self.W_t - pi_t)*self.config["r"]*self.dt
        self.t += self.dt
        self.X_t += dX_t; self.W_t += dW_t; self.Y_t += dY_t; self.Z_t += dZ_t
        self.t_hist.append(copy.deepcopy(self.t)); self.X_t_hist.append(copy.deepcopy(self.X_t)); self.W_t_hist.append(copy.deepcopy(self.W_t))
        self.Y_t_hist.append(copy.deepcopy(self.Y_t)); self.Z_t_hist.append(copy.deepcopy(self.Z_t))

    def plot(self):
        plt.plot(self.t_hist, self.Y_t_hist, label="Y_t")
        plt.plot(self.t_hist, self.Z_t_hist, label="Z_t")
        plt.plot(self.t_hist, self.X_t_hist, label="X_t")
        plt.plot(self.t_hist, self.W_t_hist, label="W_t")
        plt.legend()

    def mu(self, y, z):
        return y*z
    
    def sigma(self, y, z):
        return np.sqrt(y**2+z**2)+0.01
    
pi = np.zeros(t_len)
model_ = model(model_config)
model_.simulate_trajectory(pi, is_plot=True)

#%% pde solver
t_min = 0; t_max = 1; t_len = 101
x_min = -1; x_max = 1; x_len = 501
y_min = -1; y_max = 1; y_len = 501
z_min = -1; z_max = 1; z_len = 501

pde_config = {"kappa_Y": kappa_Y, "kappa_Z": kappa_Z, "sigma_Y": sigma_Y, "sigma_Z": sigma_Z, "gamma": gamma, "r": r,
              "t_min": t_min, "t_max": t_max, "t_len": t_len,
              "x_min": x_min, "x_max": x_max, "x_len": x_len,
              "y_min": y_min, "y_max": y_max, "y_len": y_len,
              "z_min": z_min, "z_max": z_max, "z_len": z_len}

class pde_solver:
    def __init__(self, pde_config):
        self.config = pde_config
        self.h = np.zeros((self.config["x_len"], self.config["y_len"], self.config["z_len"]))
        self.h_all = np.zeros((1, self.config["x_len"], self.config["y_len"], self.config["z_len"]))

        self.dt = (self.config["t_max"]-self.config["t_min"])/(self.config["t_len"]-1)
        self.dx = (self.config["x_max"]-self.config["x_min"])/(self.config["x_len"]-1)
        self.dy = (self.config["y_max"]-self.config["y_min"])/(self.config["y_len"]-1)
        self.dz = (self.config["z_max"]-self.config["z_min"])/(self.config["z_len"]-1)

        self.t_axis = np.linspace(self.config["t_min"], self.config["t_max"], self.config["t_len"])
        self.x_axis = np.linspace(self.config["x_min"], self.config["x_max"], self.config["x_len"])
        self.y_axis = np.linspace(self.config["y_min"], self.config["y_max"], self.config["y_len"])
        self.z_axis = np.linspace(self.config["z_min"], self.config["z_max"], self.config["z_len"])

    def solve(self):
        if os.path.exists("pde_result.npz"):
            result = np.load("pde_result.npz")
            self.h_all = result["h"]
            self.t_axis = result["t"]
            self.x_axis = result["x"]
            self.y_axis = result["y"]
        else:
            for t in tqdm.tqdm(range(self.config["t_len"])):
                par_t_h = self.par_t_h()
                h_new = self.h - self.dt*par_t_h
                self.h = h_new
                self.h_all = np.concatenate((self.h_all, copy.deepcopy(np.expand_dims(self.h, axis=0))), axis=0)

            np.savez("pde_result.npz", t=self.t_axis, x=self.x_axis, y=self.y_axis, z=self.z_axis, h=self.h_all[::-1, :, :, :])

    def pi_opt(self, t, w, x, y, z):
        t_idx = max(0, min(self.config["t_len"]-1, int(t-self.config["t_min"]/self.dt)))
        x_idx = max(0, min(self.config["x_len"]-1, int((x-self.config["x_min"])/self.dx)))
        y_idx = max(0, min(self.config["y_len"]-1, int((y-self.config["y_min"])/self.dy)))
        z_idx = max(0, min(self.config["z_len"]-1, int((z-self.config["z_min"])/self.dz)))
        
        h = self.h_all[t_idx, x_idx, y_idx, z_idx]
        if x_idx == 0:
            par_x_h = (self.h_all[t_idx, x_idx+1, y_idx, z_idx]-self.h_all[t_idx, x_idx, y_idx, z_idx])/self.dx
        elif x_idx == self.config["x_len"]-1:
            par_x_h = (self.h_all[t_idx, x_idx, y_idx, z_idx]-self.h_all[t_idx, x_idx-1, y_idx, z_idx])/self.dx
        else:
            par_x_h = (self.h_all[t_idx, x_idx+1, y_idx, z_idx]-self.h_all[t_idx, x_idx-1, y_idx, z_idx])/(2*self.dx)

        H = -np.exp(-self.config["gamma"]*(w*np.exp(self.config["r"]*(self.config["t_max"]-t))+h))
        par_ww_H = np.power(self.config["gamma"],2)*H*np.exp(2*self.config["r"]*(1-t))
        par_w_H = -self.config["gamma"]*H*np.exp(self.config["r"]*(1-t))
        par_wx_H = np.power(self.config["gamma"],2)*H*np.exp(self.config["r"]*(1-t))*par_x_h

        mu = self.mu(y, z); sigma_sq = self.sigma(y, z)**2
        pi = -(1/(par_ww_H*sigma_sq))*((mu-r)*par_w_H + sigma_sq*par_wx_H)
        return pi

    def par_t_h(self):
        ar = np.zeros((self.config["x_len"], self.config["y_len"], self.config["z_len"])); ar[:] = np.nan

        kappa_Y = self.config["kappa_Y"]; kappa_Z = self.config["kappa_Z"]
        gamma = self.config["gamma"]; r = self.config["r"]
        sigma_Y_sq = self.config["sigma_Y"]**2; sigma_Z_sq = self.config["sigma_Z"]**2

        for i in range(self.config["x_len"]):
            for j in range(self.config["y_len"]):
                for k in range(self.config["z_len"]):
                    x = self.config["x_min"]+i*self.dx; y = self.config["y_min"]+j*self.dy; z = self.config["z_min"]+k*self.dz
                    mu = self.mu(y, z); sigma_sq = self.sigma(y, z)**2
                    par_x_h = self.par_x_h(i, j, k); par_y_h = self.par_y_h(i, j, k); par_z_h = self.par_z_h(i, j, k)
                    par_xx_h = self.par_xx_h(i, j, k); par_yy_h = self.par_yy_h(i, j, k); par_zz_h = self.par_zz_h(i, j, k)

                    ar[i, j, k] = -0.5*sigma_sq*par_xx_h - 0.5*sigma_Y_sq*par_yy_h - 0.5*sigma_Z_sq*par_zz_h\
                    - r*par_x_h + kappa_Y*y*par_y_h + kappa_Z*z*par_z_h\
                    +0.5*gamma*sigma_Y_sq*np.power(par_y_h, 2) + 0.5*gamma*sigma_Z_sq*np.power(par_z_h, 2)\
                    - np.power(mu-r, 2)/(2*gamma*sigma_sq)

        return ar

    def mu(self, y, z):
        return y*z

    def sigma(self, y, z):
        return np.sqrt(y**2+z**2)+0.01

    def par_x_h(self, i, j, k):
        if i == 0:
            return (self.h[i+1, j, k]-self.h[i, j, k])/self.dx
        elif i == self.config["x_len"]-1:
            return (self.h[i, j, k]-self.h[i-1, j, k])/self.dx
        else:
            return (self.h[i+1, j, k]-self.h[i-1, j, k])/(2*self.dx)
        
    def par_xx_h(self, i, j, k):
        if i == 0:
            return (self.par_x_h(i+1, j, k)-self.par_x_h(i, j, k))/self.dx
        elif i == self.config["x_len"]-1:
            return (self.par_x_h(i, j, k)-self.par_x_h(i-1, j, k))/self.dx
        else:
            return (self.par_x_h(i+1, j, k)-self.par_x_h(i-1, j, k))/(2*self.dx)
        
    def par_y_h(self, i, j, k):
        if j == 0:
            return (self.h[i, j+1, k]-self.h[i, j, k])/self.dy
        elif j == self.config["y_len"]-1:
            return (self.h[i, j, k]-self.h[i, j-1, k])/self.dy
        else:
            return (self.h[i, j+1, k]-self.h[i, j-1, k])/(2*self.dy)
        
    def par_yy_h(self, i, j, k):
        if j == 0:
            return (self.par_y_h(i, j+1, k)-self.par_y_h(i, j, k))/self.dy
        elif j == self.config["y_len"]-1:
            return (self.par_y_h(i, j, k)-self.par_y_h(i, j-1, k))/self.dy
        else:
            return (self.par_y_h(i, j+1, k)-self.par_y_h(i, j-1, k))/(2*self.dy)
        
    def par_z_h(self, i, j, k):
        if k == 0:
            return (self.h[i, j, k+1]-self.h[i, j, k])/self.dz
        elif k == self.config["z_len"]-1:
            return (self.h[i, j, k]-self.h[i, j, k-1])/self.dz
        else:
            return (self.h[i, j, k+1]-self.h[i, j, k-1])/(2*self.dz)
        
    def par_zz_h(self, i, j, k):
        if k == 0:
            return (self.par_z_h(i, j, k+1)-self.par_z_h(i, j, k))/self.dz
        elif k == self.config["z_len"]-1:
            return (self.par_z_h(i, j, k)-self.par_z_h(i, j, k-1))/self.dz
        else:
            return (self.par_z_h(i, j, k+1)-self.par_z_h(i, j, k-1))/(2*self.dz)

#%%
gamma = 0.1; r = 0.001
kappa_Y = 1/0.01; sigma_Y = 1
kappa_Z = 1/0.10; sigma_Z = 1
t_min = 0; t_max = 1; t_len = 1001

model_config = {"kappa_Y": kappa_Y, "kappa_Z": kappa_Z, "sigma_Y": sigma_Y, "sigma_Z": sigma_Z, "gamma": gamma, "r": r,
              "t_min": t_min, "t_max": t_max, "t_len": t_len}

model_ = model(model_config)
pi = np.zeros(t_len)
model_.simulate_trajectory(pi, is_plot=True)

#%%
t_min = 0; t_max = 1; t_len = 501
x_min = -1; x_max = 1; x_len = 201
y_min = -1; y_max = 1; y_len = 201
z_min = -1; z_max = 1; z_len = 201

pde_config = {"kappa_Y": kappa_Y, "kappa_Z": kappa_Z, "sigma_Y": sigma_Y, "sigma_Z": sigma_Z, "gamma": gamma, "r": r,
              "t_min": t_min, "t_max": t_max, "t_len": t_len,
              "x_min": x_min, "x_max": x_max, "x_len": x_len,
              "y_min": y_min, "y_max": y_max, "y_len": y_len,
              "z_min": z_min, "z_max": z_max, "z_len": z_len}

pde_solver_ = pde_solver(pde_config)
pde_solver_.solve()

#%%




