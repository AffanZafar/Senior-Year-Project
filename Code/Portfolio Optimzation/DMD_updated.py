#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime
import math
from scipy import stats
import time 
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

interest_rate = pd.read_csv('Libor3m.csv')
stock = pd.read_csv('SBUX.csv')
interest_rate = interest_rate.drop(columns=['DATE'])
stock = stock['Close']
stock = stock.values
interest_rate["USD3MTD156N"]= pd.to_numeric(interest_rate.USD3MTD156N, errors='coerce')
interest_rate = interest_rate.values
sigma  = (stock.std()*100)/120

class EUCallOption: 
    def d1(self, S, K, r, sigma, dt):
        d_1 =  (math.log((S/K)) + r+math.pow(sigma, 2.)/2*dt)/ (sigma* np.sqrt(dt))
        return d_1
    def d2( self, d1, sigma, dt):
        return d1 - (sigma*np.sqrt(dt))
    
    def optionPrice(self, S, d1, d2, K, r, dt):
        nd1 = stats.norm.cdf(d1)
        nd2 = stats.norm.cdf(d2)
        return S*nd1 - K*(math.exp(-r*dt))*nd2
       
    def delta(self, d1):
        return stats.norm.cdf(d1)
    
    def __init__(self, asset_price, strike_price, volatility, expiration_date, risk_free_rate, drift, dt):
        self.asset_price = asset_price
        self.strike_price =  strike_price
        self.expiration_date = expiration_date
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.drift = drift
        d1 = self.d1(asset_price, strike_price, risk_free_rate, volatility, dt)
        d2 = self.d2(d1, volatility, dt)
        self.price = self.optionPrice(asset_price, d1, d2, strike_price, risk_free_rate, dt )
        self.delta =self.delta(d1)

def FDM(K, sigma, r, T, L, Nx, Nt):
    S = np.linspace(0,L, Nx)
    t= np.linspace(0, T, Nt)
    dS = S[1] - S[0]
    dt = t[1] - t[0]
    beta = dt/(dS**2)
    
    #Solution U(x, t)
    U = np.zeros((Nx, Nt))
    #print(U.shape)
    
    #terminal conditions
    for n in range(0,Nx):
        U[n][0]=max(S[n]-K,0)
    #print("U\n", U)
    
    
    #boundary for S for all t
    U[0, :] =0.
    for m in range(1, Nt):
        U[Nx-1, :] =S[Nx-1]-K*np.exp(t[m-1]*(-r))
    
    a=[]
    b=[]
    c=[]
    for i in range(1,  Nx-1):
        a.append(0.5*(r*i-sigma*sigma*i*i)*Nt)
        b.append(1+(r+sigma*sigma*i*i)*Nt)
        c.append(-0.5*(sigma*sigma*i*i+r*i)*Nt)
        
    F = np.zeros((Nx-2, Nx-2))
    
    F[0][0]=b[0]
    F[Nx-3][Nx-3]=b[Nx-3]    
    for i in range(1, Nx-2):
        F[i][i]= b[i]
        F[i][i-1] = a[i]
        F[i-1][i] = c[i-1]
    F_inv=np.linalg.inv(F)
    for j in range(Nt-2, -1,-1):
        rSize = Nx-2
        P = np.zeros(rSize)
        P[0] = a[0]*U[0][j+1] 
        P[Nx-3] = c[rSize-1]*U[Nx-1][j]  
        
        U_next =np.zeros(rSize)
        for i in range(0,Nx-2):
            U_next[i]= U[i+1][j]
        
        UU = np.matmul(F_inv, U_next-P)
        for i in range(0, rSize):
            U[i+1][j+1] = UU[i]
            
    # snapshot
    x = U.reshape(Nx*Nt,1)
    return t,S,U,x

train_length = len(stock)

K = np.arange(100-4*sigma*1.2,100+5*sigma*1.2, sigma*0.4)

t = np.full([len(K)], None)
S = np.full([len(K)], None)
U = np.full([len(K)], None)
V = np.full([len(K)], None)
X = []
'''
t_dt_var = np.full([len(K)], None)
S_dt_var = np.full([len(K)], None)
U_dt_var = np.full([len(K)], None)
V_dt_var = np.full([len(K)], None)
X_dt_var = []

t_St_var = np.full([len(K)], None)
S_St_var = np.full([len(K)], None)
U_St_var = np.full([len(K)], None)
V_St_var = np.full([len(K)], None)
X_St_var = []

t_dt_St_var = np.full([len(K)], None)
S_dt_St_var = np.full([len(K)], None)
U_dt_St_var = np.full([len(K)], None)
V_dt_St_var = np.full([len(K)], None)
X_dt_St_var = []
'''
for i in range (1):
    
    t[i], S[i], U[i], V[i] = FDM(150, 0.05, 0.13, 10, 120, 120, 10)
    X.append(FDM(150, 0.05, 0.13, 10, 120, 120, 10)[3])
    
    #Plot the surface
    xgrid, ygrid = np.meshgrid(t[i], S[i])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xgrid, ygrid, U[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$Time-t$')
    ax.set_ylabel('$Stock Price-S$')
    ax.set_zlabel('$Option Value-U$')
    ax.set_title('K=150,sigma=0.05')
    plt.show()
    
    print('\n\n', i+1, 'Training point for t\n', t[i])
    print('\n', i+1, 'Training point for S\n', S[i])
    print('\n', i+1, 'Training point for U\n', U[i])
    '''
    t_dt_var[i], S_dt_var[i], U_dt_var[i], V_dt_var[i] = FDM(K[i], sigma, 0.13, 10, 120, 120, 40)
    X_dt_var.append(FDM(K[i], sigma, 0.13, 10, 120, 120, 40)[3])
    
    t_St_var[i], S_St_var[i], U_St_var[i], V_St_var[i] = FDM(K[i], sigma, 0.13, 10, 600, 120, 10)
    X_St_var.append(FDM(K[i], sigma, 0.13, 10, 600, 120, 10)[3])
    
    t_dt_St_var[i], S_dt_St_var[i], U_dt_St_var[i], V_dt_St_var[i] = FDM(K[i], sigma, 0.13, 10, 600, 120, 40)
    X_dt_St_var.append(FDM(K[i], sigma, 0.13, 10, 600, 120, 40)[3])

X = np.array(X).T
X.reshape(1200,1)
print(X.shape, X)


X_dt_var = np.array(X_dt_var).T
X_dt_var.reshape(4800,27)
print(X_dt_var.shape, X_dt_var)

X_St_var = np.array(X_St_var).T
X_St_var.reshape(1200,27)
print(X_St_var.shape, X_St_var)

X_dt_St_var = np.array(X_dt_St_var).T
X_dt_St_var.reshape(4800,27)
print(X_dt_St_var.shape, X_dt_St_var)
'''
#X, X_dt_var, X_St_var, X_dt_St_var to be plugged in for SVD
#U, U_dt_var, U_St_var, U_dt_St_var to be plugged in for SVD (only selective values)
U_120_10 = U[0]
U_120 = U_120_10[:, 1]
S_120 = S[0]
plt.plot(S_120,U_120)

print(S_120.shape,U_120.shape)


# In[85]:


import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

X_aug1 = np.array([U_120[:-2], U_120[1:-1]])
X_aug2 = np.array([U_120[1:-1], U_120[2:]])
dt = S_120[2] - S_120[1]
plt.plot(S_120,U_120)
plt.title('Orignal')
plt.show()

#SVD
U,S,Vh = LA.svd(X_aug1,False)
V = Vh.conj().T

#Atilde with A reduced by left singular vector
#U takes the complex conjugate and makes it a transposed matrix.
Atilde = np.dot(np.dot(np.dot(U.conj().T, X_aug2), V), LA.inv(np.diag(S)))

#Find the eigenvalues and eigenvectors of Atilde
Lam, W = LA.eig(Atilde)

print("eigenvalue:", Lam)

#Find the eigenvector of A from the eigenvector of Atilde
Phi = np.dot(np.dot(np.dot(X_aug2, V), LA.inv(np.diag(S))), W)

#Discrete to continuous exp(**)of**Seeking
Omega = np.diag(np.log(Lam)/dt)

print("Omega:", Omega)#Just in case, it will be 1j

#Restore the original function from Omega and Phi.
#What you are doing is the same as in 1D, but the writing style is different.
b = np.dot(LA.pinv(Phi), X_aug1[:,0])


x_dmd = np.zeros([2,len(S_120)])
for i, _t in enumerate(S_120):
    x_dmd[:,i] = np.dot(np.dot(Phi, np.exp(Omega * _t)), b)

plt.plot(S_120, x_dmd[0,:])
plt.title('Reconstructed')
plt.show()


# In[16]:


'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

X_aug1 = np.array([U_120[:-2], U_120[1:-1]])
X_aug2 = np.array([U_120[1:-1], U_120[2:]])
dt = S_120[2] - S_120[1]

# SVD of input matrix
U2,Sig2,Vh2 = svd(X_aug1, False)

r = 2
U = U2[:,:r]
Sig = diag(Sig2)[:r,:r]
V = Vh2.conj().T[:,:r]

# build A tilde
Atil = dot(dot(dot(U.conj().T, X_aug2), V), inv(Sig))
mu,W = eig(Atil)

# build DMD modes
Phi = dot(dot(dot(X_aug2, V), inv(Sig)), W)

b = dot(pinv(Phi), X_aug1[:,0])
Psi = np.zeros([r, len(S_120)], dtype='complex')
for i,_t in enumerate(S_120):
    Psi[:,i] = multiply(power(mu, _t/dt), b)
    
Psi.shape    

D2 = dot(Phi, Psi)
np.allclose(U_120, D2) # True
''' 

