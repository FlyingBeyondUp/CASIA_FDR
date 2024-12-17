# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:21:59 2024

@author: LZ166
"""
import numpy as np
import matplotlib.pyplot as plt
from distributions import Spiky, NearNormal,FlatTop,Skew,BigNormal,Bimodal,StandardNormal
from NewDeal import generateData, getGaussian,NewDeal
from scipy.stats import norm

n_betas = 12000 # Number of beta coefficients
observed_beta=generateData(prior=StandardNormal(name='custom_dist'),signal_strength=1,signal_ratio=0.2,n_data=n_betas)
list_s=np.ones(n_betas)
bins = np.arange(-6, 6, 0.1)


newDeal=NewDeal(observed_beta, list_s,prior_type='uniform')
list_pi,list_log_likelihood,omega=newDeal.solve(n_iter=400,lambda_0=10)
plt.xlabel('iteration')
plt.ylabel('log-likelihood/n_betas')
plt.plot(range(0,len(list_log_likelihood)),list_log_likelihood)
# plt.savefig('EM convergence.pdf', bbox_inches='tight')

K=len(list_pi)
list_beta=np.linspace(-6,6,n_betas)
grid=newDeal.grid

marginal_p=newDeal.getMargin(list_beta, list_s)

null_component=np.zeros_like(observed_beta)
lfdr=np.zeros_like(observed_beta)
for j in range(0,n_betas):
    null_component[j]=getGaussian(list_beta[j], 0, list_s[j]**2)*list_pi[0]
lfdr=null_component/marginal_p
alternative_component=marginal_p-null_component

plt.figure(figsize=(10, 6))
plt.ylabel('density')
plt.xlabel('z')
plt.title('The New Deal')
plt.plot(list_beta,marginal_p,label='marginal',color='blue')
plt.hist( observed_beta,bins=len(bins),density=True, alpha=0.5, label='Observed Counts', color='blue')
plt.plot(list_beta,null_component,label='null',color='orange')
plt.plot(list_beta,alternative_component,label='alternative',color='red')
plt.legend()
# plt.savefig('0.8 null and 0.2 normal, UA mode at zero.pdf', bbox_inches='tight')

plt.figure()
plt.plot(list_beta,lfdr)
plt.ylabel('fdr(z)')
plt.xlabel('z')


# n_iter=20
# true_pi0=0.2
# list_signal_strength=[0.5,0.8,1,1.2,1.5,1.8,2,3,4]
# list_pi0_error=[]
# for signal_strength in list_signal_strength:
#     list_pi0=[]
#     for i in range(n_iter):
#         observed_beta=generateData(prior=StandardNormal(name='custom_dist'),signal_strength=signal_strength,signal_ratio=1-true_pi0)
#         newDeal=NewDeal(observed_beta, list_s)
#         list_pi,list_log_likelihood,omega=newDeal.solve(n_iter=400,lambda_0=10)
#         list_pi0.append(list_pi[0])
#     list_pi0_error.append(np.abs(np.mean(list_pi0)-true_pi0))
# plt.figure()
# plt.xlabel('signal noise ratio')
# plt.ylabel(r'absolute error of the estimated $\pi_0$')
# plt.plot(list_signal_strength,list_pi0_error)
# plt.plot(list_signal_strength,list_pi0_error,marker='^')
# # plt.savefig('true_pi0=0.2 prior_type=normal SNR.pdf', bbox_inches='tight')
# print(list_pi0)
# print(np.mean(list_pi0))