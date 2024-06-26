#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:20:54 2021

@author: jilberto
"""

import sympy as sp
import numpy as np
import matplotlib.pylab as plt

def sym_mat(matrix):
    matout=(matrix + matrix.T)/2
    return matout

def frob_inner(mat1,mat2):
    matout=sp.Trace(mat1.T * mat2)
    return matout

F = sp.MatrixSymbol('F', 3, 3)
p = sp.symbols('p')
E = 0.5*(F.T*F - sp.Identity(3))

# Fung type
C_f, bf, bt, bfs = sp.symbols('C_f, bf, bt, bfs ')


Q = bf*E[0,0]**2 + bt*(E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2) + bfs*(E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2)
W = 0.5*C_f*(sp.exp(Q) - 1)
P = sp.Matrix([[W.diff(F[0,0]), W.diff(F[0,1]), W.diff(F[0,2])],
                [W.diff(F[1,0]), W.diff(F[1,1]), W.diff(F[1,2])],
                [W.diff(F[2,0]), W.diff(F[2,1]), W.diff(F[2,2])]])

lam = sp.symbols('lam')
F_biax = sp.Matrix([[lam, 0., 0.],
                    [0., lam, 0.],
                    [0., 0., 1/lam**2]])

P_biax = P.subs(F, F_biax)
sigma_biax = P_biax*F_biax.T

func_fung_11 = sp.lambdify([lam, C_f, bf, bt, bfs], sigma_biax[0,0])
func_fung_22 = sp.lambdify([lam, C_f, bf, bt, bfs], sigma_biax[1,1])


def Fung_func(l, C_f, bf, bt, bfs):

    s_11 = func_fung_11(l, C_f, bf, bt, bfs)
    s_22 = func_fung_22(l, C_f, bf, bt, bfs)
    return np.vstack([s_11, s_22])

# Nordsletten material
#b1, b2, bff,bss,bnn,bfs,bfn,bsn = sp.symbols('b1,b2,bff,bss,bnn,bfs,bfn,bsn')
bff_scale, kappa = sp.symbols('bff,kappa')

# see Fig4 caption in the paper "A viscoelastic model for human..."
b1=10.02
b2=1.158

bff=1*bff_scale
bss=1*kappa
bnn=1*kappa

bfs=6.175
bfn=3.520
bsn=2.895

f0 = sp.MatrixSymbol('f0', 3, 1)
f0 = sp.Matrix([1.,0.,0.])

s0 = sp.MatrixSymbol('s0', 3, 1)
s0 = sp.Matrix([0.,1.,0.])

n0 = sp.MatrixSymbol('n0', 3, 1)
n0 = sp.Matrix([0.,0.,1.])


C = F.T*F
I_C = C[0,0] + C[1,1] + C[2,2]
I_Cf = f0.T*C*f0
I_Cs = s0.T*C*s0
I_Cn = n0.T*C*n0
I_Cf = I_Cf[0,0]

I_ff=I_Cf
I_ss=1.0*F[0, 1]**2 + 1.0*F[1, 1]**2 + 1.0*F[2, 1]**2   # <<>> ???
I_nn=1.0*F[0, 2]**2 + 1.0*F[1, 2]**2 + 1.0*F[2, 2]**2

I_fs=frob_inner(C,sym_mat(f0*s0.T))
I_fn=frob_inner(C,sym_mat(f0*n0.T))
I_sn=frob_inner(C,sym_mat(s0*n0.T))

W1=2.718281828459045**(b1*(I_C - 3))
W2=2.718281828459045**(b2*(I_fs**2 + I_fn**2 + I_sn**2))

S_sum1_term1 = bff*(W1*I_ff - 1)*(f0*f0.T)
S_sum1_term2 = bss*(W1*I_ss - 1)*(s0*s0.T)
S_sum1_term3 = bnn*(W1*I_nn - 1)*(n0*n0.T)

S_sum2_term1 = bfs*I_fs*sym_mat(f0*s0.T)
S_sum2_term2 = bfn*I_fn*sym_mat(f0*n0.T)
S_sum2_term3 = bsn*I_sn*sym_mat(s0*n0.T)


S_sum1 = S_sum1_term1 + S_sum1_term2 + S_sum1_term3
S_sum2 = W2*(S_sum2_term1 + S_sum2_term2 + S_sum2_term3)

S = S_sum1 + S_sum2
P = F.inv() * S     # <<>>
lam = sp.symbols('lam')
F_biax = sp.Matrix([[lam, 0., 0.],
                    [0., lam, 0.],
                    [0., 0., 1/lam**2]])

P_biax = P.subs(F, F_biax)
sigma_biax = P_biax*F_biax.T

# I want this to be zero
s3_doubleE = sigma_biax[2,2] + p
p_biax = sp.solve(s3_doubleE, p)[0]

s1_doubleE = sigma_biax[0,0] + p_biax
s2_doubleE = sigma_biax[1,1] + p_biax


func_s1_doubleE = sp.lambdify([lam, bff_scale, kappa], s1_doubleE)
func_s2_doubleE = sp.lambdify([lam, bff_scale, kappa], s2_doubleE)

#func_s1_doubleE = sp.lambdify([lam, b1, b2, bff, bss, bnn, bfs, bfn, bsn], s1_doubleE)
#func_s2_doubleE = sp.lambdify([lam, b1, b2, bff, bss, bnn, bfs, bfn, bsn], s2_doubleE)
def doubleE_func(l, bff_scale, kappa):



    s1 = func_s1_doubleE(l, bff_scale, kappa)
    s2 = func_s2_doubleE(l, bff_scale, kappa)

    return s1, s2

xdata = np.linspace(1., 1.15)
ydata = np.zeros([len(xdata),2])
f = np.zeros([len(xdata),2])


def func(c, C_, bf_, bt_, bfs_):
    bff_scale, kappa = c
    x = np.linspace(1., 1.15)
    ho = doubleE_func(x, bff_scale, kappa)

    fu = Fung_func(x, C_, bf_, bt_, bfs_)
    e1 = ho[0] - fu[0]
    e2 = ho[1] - fu[1]
    return np.concatenate((e1,e2))





from scipy.optimize import least_squares
plt.close('all')
plt.figure()

# # Remote
#b, bf = 5,5
C_, bf_, bt_, bfs_  =  4.89, 8.08, 2.78, 7.77 # Case 2 infarct
C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_remote = sol.x
bff_scale, kappa = param_remote

print('Remote params')
print(param_remote)
print(bff)
print(bss)
print(bnn)
print(' ------------- ')


ydata = Fung_func(xdata, C_, bf_, bt_, bfs_)
plt.plot(0.5*(xdata**2-1), ydata[0], 'bo')
plt.plot(0.5*(xdata**2-1), ydata[1], 'ro')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa)[0], 'b')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa)[1], 'r')
#%% Fit a, af
plt.figure()
# Data
xfiber, yfiber = np.loadtxt('./raw_data/base_fiber.csv', delimiter = ',', unpack=True)
xcfiber, ycfiber = np.loadtxt('./raw_data/base_crossfiber.csv', delimiter = ',', unpack=True)
xfiber = xfiber - np.min(xfiber)
xcfiber = xcfiber - np.min(xcfiber)

xf = np.sqrt(2*xfiber + 1)
xcf = np.sqrt(2*xcfiber + 1)
def func(c, C_, bf_, bt_, bfs_):
    bff_scale,kappa = c
    #b1,b2 =5,5
    ho_f, _ = doubleE_func(xf, bff_scale, kappa)
    _, ho_fs = doubleE_func(xcf, bff_scale, kappa)
    e1 = ho_f - yfiber
    e2 = ho_fs - ycfiber
    e = np.hstack([e1,e2])

    return e.flatten()

# Infarct
C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_base = sol.x
bff_scale,kappa = param_base
#b1,b2 =5,5

print('Base params')
print(param_base)
print(bff)
print(bss)
print(bnn)
print(' ------------- ')



plt.figure(0)
plt.plot(xfiber, yfiber, 'bo-', label='Baseline Fiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa )[0], 'b-')

plt.figure(1)
plt.plot(xcfiber, ycfiber, 'ro-', label='Baseline Crossfiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa)[1], 'r-')



# Data
xfiber, yfiber = np.loadtxt('raw_data/infarct_fiber.csv', delimiter = ',', unpack=True)
xcfiber, ycfiber = np.loadtxt('raw_data/infarct_crossfiber.csv', delimiter = ',', unpack=True)
xfiber = xfiber - np.min(xfiber)
xcfiber = xcfiber - np.min(xcfiber)

xf = np.sqrt(2*xfiber + 1)
xcf = np.sqrt(2*xcfiber + 1)
def func(c, C_, bf_, bt_, bfs_):
    bff,kappa = c
    #b1,b2 =5,5
    ho_f, _ = doubleE_func(xf, bff, kappa )
    _, ho_fs = doubleE_func(xcf, bff, kappa )
    e1 = ho_f - yfiber
    e2 = ho_fs - ycfiber
    e = np.hstack([e1,e2])

    return e.flatten()

C_, bf_, bt_, bfs_  =  4.01, 19.01, 8.07, 17.19 # 12 weeks
sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_infarct = sol.x
bff_scale,kappa = param_infarct

print('Infarct params')
print(param_infarct)
print(bff)
print(bss)
print(bnn)
print(' ------------- ')



plt.figure(0)
plt.plot(xfiber, yfiber, 'bs--', label='Infarct Fiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa)[0], 'b--')
plt.grid(True)
plt.xlim([0.,0.1])
plt.ylim([0.,30])
plt.legend()
plt.savefig('fit_a_f.pdf', bbox_inches='tight')


plt.figure(1)
plt.plot(xcfiber, ycfiber, 'rs--', label='Infarct Crossfiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale, kappa)[1], 'r--')
plt.grid(True)
plt.xlim([0.,0.1])
plt.ylim([0.,30])
plt.legend()
plt.savefig('fit_a_cf.pdf', bbox_inches='tight')

#print(param_infarct[0]/param_base[0], param_infarct[1]/param_base[1], param_infarct[2]/param_base[2])

#%% Fit a, b, af, bf
plt.close('all')
# Data
xfiber, yfiber = np.loadtxt('raw_data/base_fiber.csv', delimiter = ',', unpack=True)
xcfiber, ycfiber = np.loadtxt('raw_data/base_crossfiber.csv', delimiter = ',', unpack=True)
xfiber = xfiber - np.min(xfiber)
xcfiber = xcfiber - np.min(xcfiber)

xf = np.sqrt(2*xfiber + 1)
xcf = np.sqrt(2*xcfiber + 1)
def func(c, C_, bf_, bt_, bfs_):
    bff_scale,kappa = c
    ho_f, _ = doubleE_func(xf, bff_scale,kappa)
    _, ho_fs = doubleE_func(xcf, bff_scale,kappa)
    e1 = ho_f - yfiber
    e2 = ho_fs - ycfiber
    e = np.hstack([e1,e2])

    return e.flatten()

# Infarct
C_, bf_, bt_, bfs_  =  0.39, 72.03, 5.08, 36.69 # Baseline
sol = least_squares(func, np.ones(2), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_base = sol.x
bff_scale,kappa = param_base

plt.figure(0)
plt.plot(xfiber, yfiber, 'bo-', label='Baseline Fiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale,kappa)[0], 'b-')

plt.figure(1)
plt.plot(xcfiber, ycfiber, 'ro-', label='Baseline Crossfiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale,kappa)[1], 'r-')



# Data
xfiber, yfiber = np.loadtxt('raw_data/infarct_fiber.csv', delimiter = ',', unpack=True)
xcfiber, ycfiber = np.loadtxt('raw_data/infarct_crossfiber.csv', delimiter = ',', unpack=True)
xfiber = xfiber - np.min(xfiber)
xcfiber = xcfiber - np.min(xcfiber)

xf = np.sqrt(2*xfiber + 1)
xcf = np.sqrt(2*xcfiber + 1)
def func(c, C_, bf_, bt_, bfs_):
    bff_scale,kappa = c
    ho_f, _ = doubleE_func(xf, bff_scale,kappa)
    _, ho_fs = doubleE_func(xcf, bff_scale,kappa)
    e1 = ho_f - yfiber
    e2 = ho_fs - ycfiber
    e = np.hstack([e1,e2])

    return e.flatten()

C_, bf_, bt_, bfs_  =  4.01, 19.01, 8.07, 17.19 # 12 weeks
sol = least_squares(func, np.array([1.,1.]), args=(C_, bf_, bt_, bfs_), bounds=(0.,np.inf))
param_infarct = sol.x
bff_scale,kappa = param_infarct

plt.figure(0)
plt.plot(xfiber, yfiber, 'bs--', label='Infarct Fiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale,kappa)[0], 'b--')
plt.grid(True)
plt.xlim([0.,0.1])
plt.ylim([0.,30])
plt.legend()
plt.savefig('fit_ab_f.pdf', bbox_inches='tight')


plt.figure(1)
plt.plot(xcfiber, ycfiber, 'rs--', label='Infarct Crossfiber')
plt.plot(0.5*(xdata**2-1), doubleE_func(xdata, bff_scale,kappa)[1], 'r--')
plt.grid(True)
plt.xlim([0.,0.1])
plt.ylim([0.,30])
plt.legend()
plt.savefig('fit_ab_cf.pdf', bbox_inches='tight')


for i in range(len(param_infarct)):
    print(param_infarct[i])
    print(param_remote[i])
    print('-------')
