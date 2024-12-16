# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:32:45 2024

@author: zlihc
"""

import numpy as np
import matplotlib.pyplot as plt



n_betas = 12000  # Number of beta coefficients
beta = np.random.normal(loc=0, scale=1, size=n_betas)

# beta=[np.random.randn(1).item() if np.random.rand(1).item()>0.5 else 0 for i in range(0,n_betas)]
delta_x=12/n_betas
# Step 2: Construct observed data by adding noise
noise = np.random.normal(loc=0, scale=1, size=n_betas)  # Noise added to beta
observed_beta = beta + noise
bins = np.arange(-6, 6, 0.1)



def getGaussian(x, mu, sigma_sq):
    return (1 / (np.sqrt(2 * np.pi * sigma_sq))) * np.exp(-((x - mu)**2) / (2 * sigma_sq))

def getNormalL(observed_data,list_s):
    sigma_min,sigma_max=np.min(list_s)/10,2*np.max(observed_data**2-list_s**2)
    list_sigma=[delta_x,sigma_min]
    sigma=sigma_min
    for m in range(0,100):
        sigma=1.414*sigma
        if sigma<=sigma_max:
            list_sigma.append(sigma)
        else:
            break
    K,n_data=len(list_sigma),len(observed_data)
    L=np.zeros((K,n_data))
    for j in range(0,n_data):
        for k in range(0,K):
            L[k,j]=getGaussian(observed_data[j],0,list_s[j]**2+list_sigma[k]**2)
    return L,list_sigma

def getUniformL(list_arg):
    pass

def solveNewDeal(observed_data,list_arg,prior_type='normal',n_iter=1000,lambda_0=10):
    # prepare sigma_k for K different components
    if prior_type=='normal':
        L,list_para=getNormalL(observed_data,list_arg)
    elif prior_type=='uniform':
        L=getUniformL(observed_data,list_arg)
    K,n_data=len(list_para),len(observed_data)
    
    list_pi=np.ones(K)/n_data
    list_pi[0]=1-(K-1)/n_data
    list_lam=np.ones((K,1))
    list_lam[0]=lambda_0
    omega=np.zeros_like(L)
    list_n=np.zeros((K,1))
    list_log_likelihood=[]
    
    # EM algorithm optimizing pi_k
    for i in range(n_iter):
        temp=np.diag(list_pi.flatten())@L
        omega=temp/(np.ones(K).reshape(1,K)@temp).reshape(n_data,)
        list_n=omega@np.ones(n_data).reshape(n_data,1)+list_lam-1
        list_pi=list_n/np.sum(list_n)
        
        log_likelihood=np.sum(np.log(np.ones((1,K)).dot(temp)))+(list_lam-1).T@np.log(list_pi+1e-6)
        list_log_likelihood.append(log_likelihood.item()/n_data)
    return list_pi,list_log_likelihood,omega,list_para,L

list_s=np.ones(n_betas)
list_pi,list_log_likelihood,omega,list_sigma,L=solveNewDeal(observed_beta, list_s,n_iter=1000,lambda_0=5000)
plt.plot(range(0,len(list_log_likelihood)),list_log_likelihood)

K=len(list_pi)
list_beta=np.linspace(-6,6,n_betas)

marginal_p=np.zeros(n_betas)
for j in range(0,len(list_beta)):
    for k in range(0,K):
        marginal_p[j]+=list_pi[k]*getGaussian(list_beta[j], 0, list_s[j]**2+list_sigma[k]**2)
plt.figure()
plt.plot(list_beta,marginal_p)
plt.hist( observed_beta,bins=len(bins),density=True, alpha=0.5, label='Observed Counts', color='blue')

null_component=np.zeros_like(beta)
lfdr=np.zeros_like(beta)
for j in range(0,n_betas):
    null_component[j]=getGaussian(list_beta[j], 0, list_s[j]**2)*list_pi[0]
    # prior_pj=0
    # for k in range(0,K):
    #     prior_pj+=list_pi[k]*getGaussian(0, 0, list_sigma[k]**2)
lfdr=null_component/marginal_p
alternative_component=marginal_p-null_component
plt.figure()
plt.plot(list_beta,null_component,label='null')
plt.plot(list_beta,alternative_component,label='alternative')
plt.legend()

plt.figure()
plt.plot(list_beta,lfdr)
plt.ylabel('fdr(z)')
plt.xlabel('z score')


