#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:20:54 2021

@author: jilberto
"""

import sympy as sp
import numpy as np
import matplotlib.pylab as plt

F = sp.MatrixSymbol('F', 3, 3)
p = sp.symbols('p')
E = 0.5*(F.T*F - sp.Identity(3))

# Fung type
C, bf, bt, bfs = sp.symbols('C, bf, bt, bfs ')


Q = bf*E[0,0]**2 + bt*(E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2) + bfs*(E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2)
W = 0.5*C*(sp.exp(Q) - 1)
P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
                [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
                [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])

lam = sp.symbols('lam')
F_biax = sp.Matrix([[lam, 0., 0.],
                    [0., lam, 0.],
                    [0., 0., 1/lam**2]])

P_biax = P.subs(F, F_biax)
sigma_biax = P_biax*F_biax.T

func_fung_11 = sp.lambdify([lam, C, bf, bt, bfs], sigma_biax[0,0])
func_fung_22 = sp.lambdify([lam, C, bf, bt, bfs], sigma_biax[1,1])
    
def Fung_func(l, C, bf, bt, bfs):
    s_11 = func_fung_11(l, C, bf, bt, bfs)
    s_22 = func_fung_22(l, C, bf, bt, bfs)
    return np.vstack([s_11, s_22])


# HO material 
a, b, af, bf = sp.symbols('a, b, af, bf')
f0 = sp.MatrixSymbol('f0', 3, 1)
f0 = sp.Matrix([1.,0.,0.])

C = F.T*F
I_C = C[0,0] + C[1,1] + C[2,2]
I_Cf = f0.T*C*f0
I_Cf = I_Cf[0,0]

W = a/(2*b)*(sp.exp(b*(I_C-3))-1) + af/(2*bf)*(sp.exp(bf*(I_Cf - 1)**2)-1)
P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
               [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
               [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])
lam = sp.symbols('lam')
F_biax = sp.Matrix([[lam, 0., 0.],
                    [0., lam, 0.],
                    [0., 0., 1/lam**2]])

P_biax = P.subs(F, F_biax)
sigma_biax = P_biax*F_biax.T

# I want this to be zero
s3_HO = sigma_biax[2,2] + p
p_biax = sp.solve(s3_HO, p)[0]

s1_HO = sigma_biax[0,0] + p_biax
s2_HO = sigma_biax[1,1] + p_biax

func_s1_HO = sp.lambdify([lam, a, b, af, bf ], s1_HO)
func_s2_HO = sp.lambdify([lam, a, b, af, bf ], s2_HO)
def HO_func(l, a, b, af, bf):
    s1 = func_s1_HO(l, a, b, af, bf)
    s2 = func_s2_HO(l, a, b, af, bf)
    
    return s1, s2

xdata = np.linspace(1., 1.15)
ydata = np.zeros([len(xdata),2])
f = np.zeros([len(xdata),2])


def func(c, C_, bf_, bt_, bfs_):
    a,b,af,bf = c
    x = np.linspace(1., 1.15)
    ho = HO_func(x, a, b, af, bf)
    fu = Fung_func(x, C_, bf_, bt_, bfs_)
    e1 = ho[0] - fu[0]
    e2 = ho[1] - fu[1]
    return np.concatenate((e1,e2))


from scipy.optimize import least_squares
plt.close('all')
plt.figure()

# # Remote
# b, bf = 5,5
# C_, bf_, bt_, bfs_  =  4.89, 8.08, 2.78, 7.77 # Case 2 infarct
C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
sol = least_squares(func, np.ones(4), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_remote = sol.x
a,b,af,bf = param_remote

print(sol.x)

ydata = Fung_func(xdata, C_, bf_, bt_, bfs_)
print(HO_func(xdata, a, b, af, bf))
plt.plot(0.5*(xdata**2-1), ydata[0], 'bo')
plt.plot(0.5*(xdata**2-1), ydata[1], 'ro')
plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[0], 'b')
plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[1], 'r')
plt.grid(True)
plt.savefig('check_raw.png', bbox_inches='tight')

#%% Fit a, af
plt.figure()
# Data
xfiber, yfiber = np.loadtxt('raw_data/base_fiber.csv', delimiter = ',', unpack=True)
xcfiber, ycfiber = np.loadtxt('raw_data/base_crossfiber.csv', delimiter = ',', unpack=True)
xfiber = xfiber - np.min(xfiber)
xcfiber = xcfiber - np.min(xcfiber)

xf = np.sqrt(2*xfiber + 1)
xcf = np.sqrt(2*xcfiber + 1)
def func(c, C_, bf_, bt_, bfs_):
    a,af = c
    b,bf =5,5
    ho_f, _ = HO_func(xf, a, b, af, bf)
    _, ho_fs = HO_func(xcf, a, b, af, bf)
    e1 = ho_f - yfiber
    e2 = ho_fs - ycfiber
    e = np.hstack([e1,e2])
    
    return e.flatten()

# Infarct
C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_base = sol.x
a,af = param_base
b, bf = 5,5
print(param_base)

plt.figure(0)
plt.plot(xfiber, yfiber, 'bo', label='Baseline Fiber')
plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[0], 'b-')
plt.plot(xcfiber, ycfiber, 'ro', label='Baseline Crossfiber')
plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[1], 'r-')
plt.xlim([0,0.1])
plt.ylim([0,10])
plt.grid(True)
plt.savefig('check_raw.png', bbox_inches='tight')


# # Data
# xfiber, yfiber = np.loadtxt('raw_data/infarct_fiber.csv', delimiter = ',', unpack=True)
# xcfiber, ycfiber = np.loadtxt('raw_data/infarct_crossfiber.csv', delimiter = ',', unpack=True)
# xfiber = xfiber - np.min(xfiber)
# xcfiber = xcfiber - np.min(xcfiber)

# xf = np.sqrt(2*xfiber + 1)
# xcf = np.sqrt(2*xcfiber + 1)
# def func(c, C_, bf_, bt_, bfs_):
#     a,af = c
#     b,bf = 5,5
#     ho_f, _ = HO_func(xf, a, b, af, bf)
#     _, ho_fs = HO_func(xcf, a, b, af, bf)
#     e1 = ho_f - yfiber
#     e2 = ho_fs - ycfiber
#     e = np.hstack([e1,e2])
    
#     return e.flatten()

# C_, bf_, bt_, bfs_  =  4.01, 19.01, 8.07, 17.19 # 12 weeks
# sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
# param_infarct = sol.x
# a,af = param_infarct



# plt.figure(0)
# plt.plot(xfiber, yfiber, 'bs--', label='Infarct Fiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[0], 'b--')
# plt.grid(True)
# plt.xlim([0.,0.1])
# plt.ylim([0.,30])
# plt.legend()
# plt.savefig('fit_a_f.pdf', bbox_inches='tight')


# plt.figure(1)
# plt.plot(xcfiber, ycfiber, 'rs--', label='Infarct Crossfiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[1], 'r--')
# plt.grid(True)
# plt.xlim([0.,0.1])
# plt.ylim([0.,30])
# plt.legend()
# plt.savefig('fit_a_cf.pdf', bbox_inches='tight')

# print(param_infarct[0]/param_base[0], param_infarct[1]/param_base[1])

# #%% Fit a, b, af, bf
# plt.close('all')
# # Data
# xfiber, yfiber = np.loadtxt('raw_data/base_fiber.csv', delimiter = ',', unpack=True)
# xcfiber, ycfiber = np.loadtxt('raw_data/base_crossfiber.csv', delimiter = ',', unpack=True)
# xfiber = xfiber - np.min(xfiber)
# xcfiber = xcfiber - np.min(xcfiber)

# xf = np.sqrt(2*xfiber + 1)
# xcf = np.sqrt(2*xcfiber + 1)
# def func(c, C_, bf_, bt_, bfs_):
#     a,b,af,bf = c
#     ho_f, _ = HO_func(xf, a, b, af, bf)
#     _, ho_fs = HO_func(xcf, a, b, af, bf)
#     e1 = ho_f - yfiber
#     e2 = ho_fs - ycfiber
#     e = np.hstack([e1,e2])
    
#     return e.flatten()

# # Infarct
# C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
# sol = least_squares(func, np.ones(4), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
# param_base = sol.x
# a,b,af,bf = param_base

# plt.figure(0)
# plt.plot(xfiber, yfiber, 'bo-', label='Baseline Fiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[0], 'b-')

# plt.figure(1)
# plt.plot(xcfiber, ycfiber, 'ro-', label='Baseline Crossfiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[1], 'r-')



# # Data
# xfiber, yfiber = np.loadtxt('raw_data/infarct_fiber.csv', delimiter = ',', unpack=True)
# xcfiber, ycfiber = np.loadtxt('raw_data/infarct_crossfiber.csv', delimiter = ',', unpack=True)
# xfiber = xfiber - np.min(xfiber)
# xcfiber = xcfiber - np.min(xcfiber)

# xf = np.sqrt(2*xfiber + 1)
# xcf = np.sqrt(2*xcfiber + 1)
# def func(c, C_, bf_, bt_, bfs_):
#     a,b,af,bf = c
#     ho_f, _ = HO_func(xf, a, b, af, bf)
#     _, ho_fs = HO_func(xcf, a, b, af, bf)
#     e1 = ho_f - yfiber
#     e2 = ho_fs - ycfiber
#     e = np.hstack([e1,e2])
    
#     return e.flatten()

# C_, bf_, bt_, bfs_  =  4.01, 19.01, 8.07, 17.19 # 12 weeks
# sol = least_squares(func, np.array([1.,1.,1.,1.]), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
# param_infarct = sol.x
# a,b,af,bf = param_infarct

# plt.figure(0)
# plt.plot(xfiber, yfiber, 'bs--', label='Infarct Fiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[0], 'b--')
# plt.grid(True)
# plt.xlim([0.,0.1])
# plt.ylim([0.,30])
# plt.legend()
# plt.savefig('fit_ab_f.pdf', bbox_inches='tight')


# plt.figure(1)
# plt.plot(xcfiber, ycfiber, 'rs--', label='Infarct Crossfiber')
# plt.plot(0.5*(xdata**2-1), HO_func(xdata, a, b, af, bf)[1], 'r--')
# plt.grid(True)
# plt.xlim([0.,0.1])
# plt.ylim([0.,30])
# plt.legend()
# plt.savefig('fit_ab_cf.pdf', bbox_inches='tight')

# print(param_infarct[0]/param_base[0], param_infarct[1]/param_base[1], 
#       param_infarct[2]/param_base[2], param_infarct[3]/param_base[3])