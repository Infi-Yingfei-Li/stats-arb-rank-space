#%%
import os, sys, copy, h5py, datetime, tqdm, gc
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

#%%
x = sp.symbols('x')
h = sp.Function('h^epsilon')(x)
par_x_h = sp.symbols('\partial_{x}h^\epsilon'); par_xx_h = sp.symbols('\partial_{xx}h^\epsilon')

t, dt = sp.symbols('t dt')
S_1, S_2 = sp.symbols('S_1 S_2')
mu_1, mu_2, sigma_1, sigma_2, rho = sp.symbols('(\mu_1+\gamma_{R(1)}) (\mu_2+\gamma_{R(2)}) sigma_1 sigma_2 rho')
q_1, q_2 = sp.symbols('q_1 q_2')
g_q1 = sp.symbols('g(q_{1})'); g_q2 = sp.symbols('g(q_{2})')
dB_1, dB_2 = sp.symbols('dB_1 dB_2')
dS_1 = mu_1*dt + g_q1*dt + sigma_1*dB_1
dS_2 = mu_2*dt + g_q2*dt + sigma_2*dB_2

Z_1, Z_2 = sp.symbols('S_{(1)} S_{(2)}')
dZ_1 = 0.5*(dS_1+dS_2) + 0.5*par_x_h*(dS_1-dS_2)+0.25*par_xx_h*(sigma_1**2+sigma_2**2-2*rho*sigma_1*sigma_2)*dt
dZ_2 = 0.5*(dS_1+dS_2) - 0.5*par_x_h*(dS_1-dS_2)-0.25*par_xx_h*(sigma_1**2+sigma_2**2-2*rho*sigma_1*sigma_2)*dt

WY_1, WY_2 = sp.symbols('W_1 W_2')
dWY_1 = (WY_1/S_1)*dS_1 + q_1*S_1*dt
dWY_2 = (WY_2/S_2)*dS_2 + q_2*S_2*dt

WZ_1, WZ_2 = sp.symbols('W_{(1)} W_{(2)}')
dWZ_1 = (WZ_1/Z_1)*dZ_1
dWZ_2 = (WZ_2/Z_2)*dZ_2

Q, r = sp.symbols('Q_t r')
alpha = sp.symbols('alpha')
f_q1, f_q2 = sp.symbols('f(q_{1}) f(q_{2})')
dQ = r*Q*dt - q_1*(S_1+f_q1)*dt - q_2*(S_2+f_q2)*dt

#%%
Hu = sp.Function('H^u')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)
H = sp.Function('H')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)

par_t_H = sp.symbols('\partial_{t}H')
par_S1_H = sp.symbols('\partial_{S_1}H'); par_S2_H = sp.symbols('\partial_{S_2}H')
par_Q_H = sp.symbols('\partial_{Q}H')
par_WY_1_H = sp.symbols('\partial_{W_{1}}H'); par_WY_2_H = sp.symbols('\partial_{W_{2}}H')
par_WZ_1_H = sp.symbols('\partial_{W_{(1)}}H'); par_WZ_2_H = sp.symbols('\partial_{W_{(2)}}H')

par_S1_S1_H = sp.symbols('\partial_{S_1S_1}H'); par_S2_S2_H = sp.symbols('\partial_{S_2S_2}H')
par_Q_Q_H = sp.symbols('\partial_{QQ}H')
par_WY_1_WY_1_H = sp.symbols('\partial_{W_{1}W_{1}}H'); par_WY_2_WY_2_H = sp.symbols('\partial_{W_{2}W_{2}}H')
par_WZ_1_WZ_1_H = sp.symbols('\partial_{W_{(1)}W_{(1)}}H'); par_WZ_2_WZ_2_H = sp.symbols('\partial_{W_{(2)}W_{(2)}}H')

par_S1_S2_H = sp.symbols('\partial_{S_1S_2}H'); par_S1_Q_H = sp.symbols('\partial_{S_1Q}H')
par_S1_WY_1_H = sp.symbols('\partial_{S_1W_{1}}H'); par_S1_WY_2_H = sp.symbols('\partial_{S_1W_{2}}H')
par_S1_WZ_1_H = sp.symbols('\partial_{S_1W_{(1)}}H'); par_S1_WZ_2_H = sp.symbols('\partial_{S_1W_{(2)}}H')

par_S2_Q_H = sp.symbols('\partial_{S_2Q}H')
par_S2_WY_1_H = sp.symbols('\partial_{S_2W_{1}}H'); par_S2_WY_2_H = sp.symbols('\partial_{S_2W_{2}}H')
par_S2_WZ_1_H = sp.symbols('\partial_{S_2W_{(1)}}H'); par_S2_WZ_2_H = sp.symbols('\partial_{S_2W_{(2)}}H')

par_Q_WY_1_H = sp.symbols('\partial_{QW_{1}}H'); par_Q_WY_2_H = sp.symbols('\partial_{QW_{2}}H')
par_Q_WZ_1_H = sp.symbols('\partial_{QW_{(1)}}H'); par_Q_WZ_2_H = sp.symbols('\partial_{QW_{(2)}}H')

par_WY_1_WY_2_H = sp.symbols('\partial_{W_{1}W_{2}}H')
par_WY_1_WZ_1_H = sp.symbols('\partial_{W_{1}W_{(1)}}H'); par_WY_1_WZ_2_H = sp.symbols('\partial_{W_{1}W_{(2)}}H')

par_WY_2_WZ_1_H = sp.symbols('\partial_{W_{2}W_{(1)}}H'); par_WY_2_WZ_2_H = sp.symbols('\partial_{W_{2}W_{(2)}}H')

par_WZ_1_WZ_2_H = sp.symbols('\partial_{W_{(1)}W_{(2)}}H')

#%% calculate L_t^u
Hu = sp.Function('H^u')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)
H = sp.Function('H')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)

dH = par_t_H*dt + par_S1_H*dS_1 + par_S2_H*dS_2 + par_Q_H*dQ + par_WY_1_H*dWY_1 + par_WY_2_H*dWY_2 + par_WZ_1_H*dWZ_1 + par_WZ_2_H*dWZ_2\
+ 0.5*par_S1_S1_H*(dS_1**2) + 0.5*par_S2_S2_H*(dS_2**2) + 0.5*par_Q_Q_H*(dQ**2) + 0.5*par_WY_1_WY_1_H*(dWY_1**2) + 0.5*par_WY_2_WY_2_H*(dWY_2**2) + 0.5*par_WZ_1_WZ_1_H*(dWZ_1**2) + 0.5*par_WZ_2_WZ_2_H*(dWZ_2**2)\
+ par_S1_S2_H*(dS_1*dS_2) + par_S1_Q_H*(dS_1*dQ) + par_S1_WY_1_H*(dS_1*dWY_1) + par_S1_WY_2_H*(dS_1*dWY_2) + par_S1_WZ_1_H*(dS_1*dWZ_1) + par_S1_WZ_2_H*(dS_1*dWZ_2)\
+ par_S2_Q_H*(dS_2*dQ) + par_S2_WY_1_H*(dS_2*dWY_1) + par_S2_WY_2_H*(dS_2*dWY_2) + par_S2_WZ_1_H*(dS_2*dWZ_1) + par_S2_WZ_2_H*(dS_2*dWZ_2)\
+ par_Q_WY_1_H*(dQ*dWY_1) + par_Q_WY_2_H*(dQ*dWY_2) + par_Q_WZ_1_H*(dQ*dWZ_1) + par_Q_WZ_2_H*(dQ*dWZ_2)\
+ par_WY_1_WY_2_H*(dWY_1*dWY_2) + par_WY_1_WZ_1_H*(dWY_1*dWZ_1) + par_WY_1_WZ_2_H*(dWY_1*dWZ_2)\
+ par_WY_2_WZ_1_H*(dWY_2*dWZ_1) + par_WY_2_WZ_2_H*(dWY_2*dWZ_2)\
+ par_WZ_1_WZ_2_H*(dWZ_1*dWZ_2)

exp1 = sp.collect(sp.expand(dH), dt).coeff(dt, 1)
exp1 = exp1 - sp.collect(sp.expand(exp1), dB_1).coeff(dB_1, 1)*dB_1 - sp.collect(sp.expand(exp1), dB_2).coeff(dB_2, 1)*dB_2
exp1 = sp.simplify(exp1)

exp2 = sp.collect(sp.expand(dH), dB_1).coeff(dB_1, 2)
exp2 = sp.simplify(exp2)

exp3 = sp.collect(sp.expand(dH), dB_2).coeff(dB_2, 2)
exp3 = sp.simplify(exp3)

exp4 = sp.collect(sp.expand(dH), dB_1).coeff(dB_1, 1)
exp4 = sp.collect(sp.expand(exp4), dB_2).coeff(dB_2, 1)
exp4 = sp.simplify(rho*exp4)

term_dt = sp.simplify(exp1 + exp2 + exp3 + exp4).subs(rho, 0)

variable_list = [q_1, q_2, g_q1, g_q2, f_q1, f_q2, par_t_H, par_S1_H, par_S2_H, par_Q_H, par_WY_1_H, par_WY_2_H, par_WZ_1_H, par_WZ_2_H,
                 par_S1_S1_H, par_S2_S2_H, par_Q_Q_H, par_WY_1_WY_1_H, par_WY_2_WY_2_H, par_WZ_1_WZ_1_H, par_WZ_2_WZ_2_H,
                 par_S1_S2_H, par_S1_Q_H, par_S1_WY_1_H, par_S1_WY_2_H, par_S1_WZ_1_H, par_S1_WZ_2_H,
                 par_S2_Q_H, par_S2_WY_1_H, par_S2_WY_2_H, par_S2_WZ_1_H, par_S2_WZ_2_H,
                 par_Q_WY_1_H, par_Q_WY_2_H, par_Q_WZ_1_H, par_Q_WZ_2_H,
                 par_WY_1_WY_2_H, par_WY_1_WZ_1_H, par_WY_1_WZ_2_H,
                 par_WY_2_WZ_1_H, par_WY_2_WZ_2_H,
                 par_WZ_1_WZ_2_H]

coefficients = []
for var in variable_list:
    print("----------")
    print(var)
    coefficients.append(sp.collect(sp.expand(term_dt, var), var).coeff(var, 1))
    print(coefficients[-1])
    term_dt = sp.simplify(term_dt - var*coefficients[-1])

print(term_dt)
coefficients_latex = [sp.latex(sp.simplify(coefficients[i])*variable_list[i]) for i in range(len(coefficients))]
np.savetxt('coefficients.txt', coefficients_latex, fmt='%s')


#%% calculate sup_u L_t^u
Hu = sp.Function('H^u')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)
H = sp.Function('H')(t, S_1, S_2, Q, WY_1, WY_2, WZ_1, WZ_2)

dH = par_t_H*dt + par_S1_H*dS_1 + par_S2_H*dS_2 + par_Q_H*dQ + par_WY_1_H*dWY_1 + par_WY_2_H*dWY_2 + par_WZ_1_H*dWZ_1 + par_WZ_2_H*dWZ_2\
+ 0.5*par_S1_S1_H*(dS_1**2) + 0.5*par_S2_S2_H*(dS_2**2) + 0.5*par_Q_Q_H*(dQ**2) + 0.5*par_WY_1_WY_1_H*(dWY_1**2) + 0.5*par_WY_2_WY_2_H*(dWY_2**2) + 0.5*par_WZ_1_WZ_1_H*(dWZ_1**2) + 0.5*par_WZ_2_WZ_2_H*(dWZ_2**2)\
+ par_S1_S2_H*(dS_1*dS_2) + par_S1_Q_H*(dS_1*dQ) + par_S1_WY_1_H*(dS_1*dWY_1) + par_S1_WY_2_H*(dS_1*dWY_2) + par_S1_WZ_1_H*(dS_1*dWZ_1) + par_S1_WZ_2_H*(dS_1*dWZ_2)\
+ par_S2_Q_H*(dS_2*dQ) + par_S2_WY_1_H*(dS_2*dWY_1) + par_S2_WY_2_H*(dS_2*dWY_2) + par_S2_WZ_1_H*(dS_2*dWZ_1) + par_S2_WZ_2_H*(dS_2*dWZ_2)\
+ par_Q_WY_1_H*(dQ*dWY_1) + par_Q_WY_2_H*(dQ*dWY_2) + par_Q_WZ_1_H*(dQ*dWZ_1) + par_Q_WZ_2_H*(dQ*dWZ_2)\
+ par_WY_1_WY_2_H*(dWY_1*dWY_2) + par_WY_1_WZ_1_H*(dWY_1*dWZ_1) + par_WY_1_WZ_2_H*(dWY_1*dWZ_2)\
+ par_WY_2_WZ_1_H*(dWY_2*dWZ_1) + par_WY_2_WZ_2_H*(dWY_2*dWZ_2)\
+ par_WZ_1_WZ_2_H*(dWZ_1*dWZ_2)

exp1 = sp.collect(sp.expand(dH), dt).coeff(dt, 1)
exp1 = exp1 - sp.collect(sp.expand(exp1), dB_1).coeff(dB_1, 1)*dB_1 - sp.collect(sp.expand(exp1), dB_2).coeff(dB_2, 1)*dB_2
exp1 = sp.simplify(exp1)

exp2 = sp.collect(sp.expand(dH), dB_1).coeff(dB_1, 2)
exp2 = sp.simplify(exp2)

exp3 = sp.collect(sp.expand(dH), dB_2).coeff(dB_2, 2)
exp3 = sp.simplify(exp3)

exp4 = sp.collect(sp.expand(dH), dB_1).coeff(dB_1, 1)
exp4 = sp.collect(sp.expand(exp4), dB_2).coeff(dB_2, 1)
exp4 = sp.simplify(rho*exp4)

term_dt = sp.simplify(exp1 + exp2 + exp3 + exp4).subs(rho, 0)
term_dt = term_dt.subs(g_q1, 0).subs(g_q2, 0)
term_dt = term_dt.subs(f_q1, alpha*q_1).subs(f_q2, alpha*q_2)
term_dt = term_dt.subs(q_1, S_1*(par_WY_1_H/par_Q_H-1)/(2*alpha)).subs(q_2, S_2*(par_WY_2_H/par_Q_H-1)/(2*alpha))

variable_list = [par_t_H, par_S1_H, par_S2_H, par_Q_H, par_WY_1_H, par_WY_2_H, par_WZ_1_H, par_WZ_2_H,
                 par_S1_S1_H, par_S2_S2_H, par_Q_Q_H, par_WY_1_WY_1_H, par_WY_2_WY_2_H, par_WZ_1_WZ_1_H, par_WZ_2_WZ_2_H,
                 par_S1_S2_H, par_S1_Q_H, par_S1_WY_1_H, par_S1_WY_2_H, par_S1_WZ_1_H, par_S1_WZ_2_H,
                 par_S2_Q_H, par_S2_WY_1_H, par_S2_WY_2_H, par_S2_WZ_1_H, par_S2_WZ_2_H,
                 par_Q_WY_1_H, par_Q_WY_2_H, par_Q_WZ_1_H, par_Q_WZ_2_H,
                 par_WY_1_WY_2_H, par_WY_1_WZ_1_H, par_WY_1_WZ_2_H,
                 par_WY_2_WZ_1_H, par_WY_2_WZ_2_H,
                 par_WZ_1_WZ_2_H]

coefficients = []
for var in variable_list:
    print("----------")
    print(var)
    coefficients.append(sp.collect(sp.expand(term_dt, var), var).coeff(var, 1))
    print(coefficients[-1])
    term_dt = sp.simplify(term_dt - var*coefficients[-1])

print(term_dt)
coefficients_latex = [sp.latex(sp.simplify(coefficients[i])*variable_list[i]) for i in range(len(coefficients))]
np.savetxt('coefficients_2.txt', coefficients_latex, fmt='%s')


#%%




