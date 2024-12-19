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
from multiprocessing import Pool

n_betas = 12000 # Number of beta coefficients
observed_beta=generateData(prior=StandardNormal(name='custom_dist'),signal_strength=1,signal_ratio=0.2,n_data=n_betas)
list_s=np.ones(n_betas)
bins = np.arange(-6, 6, 0.1)


newDeal=NewDeal(observed_beta, list_s,prior_type='uniform')
list_pi,list_log_likelihood,omega=newDeal.fit(n_iter=400,lambda_0=10)
plt.xlabel('iteration')
plt.ylabel('log-likelihood/n_betas')
plt.plot(range(0,len(list_log_likelihood)),list_log_likelihood)
# plt.savefig('EM convergence.pdf', bbox_inches='tight')

K=len(list_pi)
list_beta=np.linspace(-6,6,n_betas)
grid=newDeal.grid

marginal_p,null_component,alternative_component,lfdr,lfsr=newDeal.solve(list_beta, list_s)

# null_component=np.zeros_like(observed_beta)
# lfdr=np.zeros_like(observed_beta)
# for j in range(0,n_betas):
#     null_component[j]=getGaussian(list_beta[j], 0, list_s[j]**2)*list_pi[0]
# lfdr=null_component/marginal_p
# alternative_component=marginal_p-null_component

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
plt.plot(list_beta,lfdr,label='local false discovery rate')
plt.plot(list_beta,lfsr,label='local false sign rate')
plt.ylabel('false rate')
plt.xlabel('z')
plt.legend()


n_dataset=10
n_data=1200
list_beta=np.linspace(-6,6,n_data)
list_s=np.ones(n_data)
titles=['spiky','near normal', 'flat top', 'skew', 'big normal', 'bimodal']
list_prior=[Spiky(name='custom_dist'),NearNormal(name='custom_dist'),FlatTop(name='custom_dist'),Skew(name='custom_dist'),BigNormal(name='custom_dist'),Bimodal(name='custom_dist')]
list_true_pi0=np.linspace(0, 1,11)
dict_pi0_normal,dict_pi0_uniform={title:[] for title in titles },{title:[] for title in titles }
dict_lfdr_normal,dict_lfsr_normal={title:[] for title in titles },{title:[] for title in titles }

dict_true_margin,dict_true_lfdr,dict_true_lfsr={},{},{}
for i in range(len(titles)):
    for j in range(0,len(list_true_pi0)):
        true_margin,true_lfsr=np.zeros_like(list_beta),np.zeros_like(list_beta)
        for k in range(0,len(list_beta)):
            true_margin[k]=list_true_pi0[j]*getGaussian(list_beta[k], 0, list_s[k]**2)+(1-list_true_pi0[j])*np.sum(getGaussian(list_beta[k], list_beta,list_s[k]**2)*list_prior[i].pdf(list_beta))*(12/len(list_beta))
            true_lfsr[k]=(1-list_true_pi0[j])*np.min([np.sum(list_prior[i].pdf(list_beta[list_beta>0])*getGaussian(list_beta[k],list_beta[list_beta>0],list_s[k])),np.sum(list_prior[i].pdf(list_beta[list_beta<0])*getGaussian(list_beta[k],list_beta[list_beta<0],list_s[k]))])*12/len(list_beta)
        true_lfdr=getGaussian(list_beta, 0, list_s**2)*list_true_pi0[j]/true_margin
        true_lfsr=true_lfsr/true_margin+true_lfdr
        dict_true_lfdr[(i,j)],dict_true_margin[(i,j)]=true_lfdr,true_margin
        dict_true_lfsr[(i,j)]=true_lfsr
        
for i in range(0,len(list_prior)):
    print(titles[i])
    for true_pi0 in list_true_pi0:
        print(true_pi0)
        list_pi0_normal,list_pi0_uniform=[],[]
        list_lfdr_normal,list_lfdr_uniform=[],[]
        mean_lfdr,mean_lfsr=np.zeros_like(list_beta),np.zeros_like(list_beta)
        for n in range(n_dataset):
            observed_beta=generateData(list_prior[i],signal_strength=1,signal_ratio=1-true_pi0,n_data=n_data)
            newDeal_normal,newDeal_uniform=NewDeal(observed_beta, list_s),NewDeal(observed_beta, list_s,prior_type='uniform')
            
            list_pi_normal,list_log_likelihood,omega=newDeal_normal.fit(n_iter=500,lambda_0=10)
            list_pi_uniform,list_log_likelihood,omega=newDeal_uniform.fit(n_iter=500,lambda_0=10)
            list_pi0_normal.append(list_pi_normal[0])
            list_pi0_uniform.append(list_pi_uniform[0])
            
            marginal_p,null_component,alternative_component,lfdr,lfsr=newDeal_normal.solve(list_beta,list_s)
            mean_lfdr+=lfdr
            mean_lfsr+=lfsr
        mean_lfdr=mean_lfdr/n_dataset
        mean_lfsr=mean_lfsr/n_dataset
        
        dict_lfsr_normal[titles[i]].append(mean_lfsr)
        dict_lfdr_normal[titles[i]].append(mean_lfdr)
        dict_pi0_normal[titles[i]].append((np.max(list_pi0_normal),np.mean(list_pi0_normal),np.min(list_pi0_normal)))
        dict_pi0_uniform[titles[i]].append((np.max(list_pi0_uniform),np.mean(list_pi0_uniform),np.min(list_pi0_uniform)))

num_subfigures = 6
fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
y_min, y_max = 0, 1
# Plot each subfigure
for i in range(num_subfigures):
    for j in range(0,3):
        axes[i].scatter(list_true_pi0,np.array(dict_pi0_normal[titles[i]]).T[j],color='green',facecolors='none')
        axes[i].scatter(list_true_pi0,np.array(dict_pi0_uniform[titles[i]]).T[j],color='purple',facecolors='none')
    axes[i].plot(list_true_pi0,list_true_pi0,color='black',linewidth=3)
    axes[i].set_title(titles[i])
    axes[i].set_aspect(aspect='auto')
    axes[i].set_ylim(y_min, y_max)
for ax in axes[1:]:
    ax.set_yticklabels([])
# Adjust layout to prevent overlap
plt.tight_layout()
fig.text(0.5, 0.001, r'true $\pi_0$', ha='center')
fig.text(0.004, 0.5, r'estimated $\pi_0$', va='center', rotation='vertical')
plt.savefig('estimated_pi0 vs true pi0.pdf',bbox_inches='tight')


num_subfigures = 6
fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
y_min, y_max = 0, 1
# Plot each subfigure
for i in range(num_subfigures):
    for j in range(2,len(list_true_pi0)):
        plot_until=np.argmax(dict_true_lfdr[(i,j)]>=0.2)
        axes[i].scatter(dict_true_lfdr[(i,j)][0:plot_until],dict_lfdr_normal[titles[i]][j][0:plot_until],color='green',s=1)                
        axes[i].plot(dict_true_lfdr[(i,j)][0:plot_until],dict_true_lfdr[(i,j)][0:plot_until],color='black')
        axes[i].plot(dict_true_lfdr[(i,j)][0:plot_until],2*dict_true_lfdr[(i,j)][0:plot_until],color='red')
    axes[i].set_title(titles[i])
    axes[i].set_aspect(aspect='auto')
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xlim(0,0.2)
for ax in axes[1:]:
    ax.set_yticklabels([])
# Adjust layout to prevent overlap
plt.tight_layout()
fig.text(0.5, 0.001, r'true lfdr', ha='center')
fig.text(0.004, 0.5, r'estimated lfdr', va='center', rotation='vertical')
plt.savefig('estimated lfdr vs true lfdr.pdf',bbox_inches='tight')


num_subfigures = 6
fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
y_min, y_max = 0, 1
# Plot each subfigure
for i in range(num_subfigures):
    for j in range(2,len(list_true_pi0)):
        plot_until=np.argmax(dict_true_lfsr[(i,j)]>=0.2)
        axes[i].scatter(dict_true_lfsr[(i,j)][0:plot_until],dict_lfsr_normal[titles[i]][j][0:plot_until],color='green',s=1)                
        axes[i].plot(dict_true_lfsr[(i,j)][0:plot_until],dict_true_lfsr[(i,j)][0:plot_until],color='black')
        axes[i].plot(dict_true_lfsr[(i,j)][0:plot_until],2*dict_true_lfsr[(i,j)][0:plot_until],color='red')
    axes[i].set_title(titles[i])
    axes[i].set_aspect(aspect='auto')
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xlim(0,0.2)
for ax in axes[1:]:
    ax.set_yticklabels([])
# Adjust layout to prevent overlap
plt.tight_layout()
fig.text(0.5, 0.001, r'true lfsr', ha='center')
fig.text(0.004, 0.5, r'estimated lfsr', va='center', rotation='vertical')
plt.savefig('estimated lfsr vs true lfsr.pdf',bbox_inches='tight')


true_pi0=0.3
dict_true_cdf,dict_estimated_cdf={},{}
delta_x=12/len(list_beta)
for i in range(len(titles)):
    print(titles[i])
    dict_true_cdf[titles[i]]=(1-true_pi0)*list_prior[i].cdf(list_beta)+true_pi0*(list_beta>0)
    average_estimated_g=np.zeros_like(list_beta)
    for n in range(n_dataset):
        observed_beta=generateData(list_prior[i],signal_strength=1,signal_ratio=1-true_pi0,n_data=n_data)
        newDeal_normal=NewDeal(observed_beta, list_s)
        list_pi_normal,list_log_likelihood,omega=newDeal_normal.fit(n_iter=500,lambda_0=10)
        average_estimated_g+=newDeal_normal.priorCDF(list_beta)
    dict_estimated_cdf[titles[i]]=average_estimated_g/n_dataset
num_subfigures = 6
fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
y_min, y_max = 0, 1
# Plot each subfigure
for i in range(num_subfigures):
    axes[i].plot(list_beta,dict_estimated_cdf[titles[i]],color='blue',label='ash normal')
    axes[i].plot(list_beta,dict_true_cdf[titles[i]],color='red',label='true')
    axes[i].set_title(titles[i])
    axes[i].set_aspect(aspect='auto')
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xlim(0,0.2)
for ax in axes[1:]:
    ax.set_yticklabels([])
# Adjust layout to prevent overlap
plt.tight_layout()
fig.text(0.5, 0.001, r'z', ha='center')
fig.text(0.004, 0.5, r'cdf', va='center', rotation='vertical')



###############################################################################
# test the influence of point mass at origin
additional_lfsr_normal={title:[] for title in titles}
for i in range(0,len(list_prior)):
    print(titles[i])
    for true_pi0 in list_true_pi0:
        print(true_pi0)
        mean_lfsr=np.zeros_like(list_beta)
        for n in range(n_dataset):
            observed_beta=generateData(list_prior[i],signal_strength=1,signal_ratio=1-true_pi0,n_data=n_data)
            newDeal_normal=NewDeal(observed_beta, list_s,origin_point_mass=False)
            
            list_pi_normal,list_log_likelihood,omega=newDeal_normal.fit(n_iter=500,lambda_0=10)
            
            marginal_p,null_component,alternative_component,lfdr,lfsr=newDeal_normal.solve(list_beta,list_s)
            mean_lfsr+=lfsr
        mean_lfsr=mean_lfsr/n_dataset
        
        additional_lfsr_normal[titles[i]].append(mean_lfsr)

num_subfigures = 6
fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
y_min, y_max = 0, 1
# Plot each subfigure
for i in range(num_subfigures):
    for j in range:
        plot_until=np.argmax(dict_true_lfsr[(i,j)]>=0.2)
        axes[i].scatter(dict_true_lfsr[(i,j)][0:plot_until],additional_lfsr_normal[titles[i]][j][0:plot_until],color='green',s=1)                
        axes[i].plot(dict_true_lfsr[(i,j)][0:plot_until],dict_true_lfsr[(i,j)][0:plot_until],color='black')
        axes[i].plot(dict_true_lfsr[(i,j)][0:plot_until],2*dict_true_lfsr[(i,j)][0:plot_until],color='red')
    axes[i].set_title(titles[i])
    axes[i].set_aspect(aspect='auto')
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xlim(0,0.2)
for ax in axes[1:]:
    ax.set_yticklabels([])
# Adjust layout to prevent overlap
plt.tight_layout()
fig.text(0.5, 0.001, r'true lfsr', ha='center')
fig.text(0.004, 0.5, r'estimated lfsr', va='center', rotation='vertical')
plt.savefig('without point mass for data and estimation, estimated lfsr vs true lfsr.pdf',bbox_inches='tight')

###############################################################################





good_observation=generateData(prior=StandardNormal(), signal_ratio=0.5,noise_sigma=1)
poor_observation=generateData(prior=StandardNormal(), signal_ratio=0.5,noise_sigma=10)
combined=np.array(list(good_observation)+list(poor_observation))

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