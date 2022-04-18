#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from numpy import linalg as la


# In[3]:



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
    x = U.reshape(1200,1)
    return t,S,U,x

train_length = len(stock)

K = np.arange(100-4*sigma*1.2,100+5*sigma*1.2, sigma*0.4)

t = np.full([len(K)], None)
S = np.full([len(K)], None)
U = np.full([len(K)], None)
V = np.full([len(K)], None)
X = []

for i in range (len(K)):
    
    t[i], S[i], U[i], V[i] = FDM(K[i], sigma, 0.13, 10, 120, 120, 10)
    X.append(FDM(K[i], sigma, 0.13, 10, 120, 120, 10)[3])
    
    #Plot the surface
    xgrid, ygrid = np.meshgrid(t[i], S[i])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xgrid, ygrid, U[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$Time-t$')
    ax.set_ylabel('$Stock Price-S$')
    ax.set_zlabel('$Option Value-U$')
    ax.set_title(K[i])
    plt.show()
    
    print('\n\n', i+1, 'Training point for t\n', t[i])
    print('\n', i+1, 'Training point for S\n', S[i])
    print('\n', i+1, 'Training point for U\n', U[i])
    
X = np.array(X).T
X.reshape(1200,27)
print(X.shape, X)


# In[4]:


print(X.shape, X)


# In[5]:


X_new = X.reshape(1200,27)


# In[6]:


print(X_new.shape)


# In[7]:


X_T = X_new.T
X1 = np.delete(X_T, -1, 1) #X_k
X2 = np.delete(X_T, 0, 1) #X_k+1 (one day)


# In[8]:


print(X1.shape)


# In[11]:


U, s, vh = np.linalg.svd(X1, full_matrices=True)
S = np.diag(s)
print(s) #First 16 k's capture most of the variation in the data


# In[12]:


UT = U.T 
V = vh.T  

S_inv = la.inv(S)
arr = np.zeros((27,1172))
ST = np.concatenate((la.inv(S),arr), axis =1)
Atilde = UT @ X2 @ V @ ST.T


print(Atilde.shape)


# In[13]:


eig_vals, eig_vecs = la.eig(Atilde)
phi = X2 @ V @ ST.T @ eig_vecs 
print(phi.shape)

