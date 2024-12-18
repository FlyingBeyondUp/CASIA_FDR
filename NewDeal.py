# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:32:45 2024

@author: zlihc
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n_betas = 12000 # Number of beta coefficients
delta_x=12/n_betas

def generateData(prior,signal_ratio,signal_strength=1,noise_sigma=1,n_data=12000):
    data=[signal_strength*prior.rvs(size=1).item() if np.random.rand(1).item()<signal_ratio else 0 for i in range(0,n_data)]

    # Construct observed data by adding noise
    noise = np.random.normal(loc=0, scale=noise_sigma, size=n_data)  # Noise added to beta
    observed_data = data + noise
    return observed_data

def getGaussian(x, mu, sigma_sq):
    return (1 / (np.sqrt(2 * np.pi * sigma_sq))) * np.exp(-((x - mu)**2) / (2 * sigma_sq))



class NewDeal:
    def __init__(self,observed_data,list_s,prior_type='normal',origin_point_mass=True,sup_num_grid=100):
        self.data=observed_data
        self.list_s=list_s
        self.prior_type=prior_type
        self.delta_x=12/observed_data.shape[0]
        self.origin_point_mass=origin_point_mass

        if prior_type=='normal':
            self.f_margin=lambda x,s,grid: (1 / (np.sqrt(2 * np.pi * (s**2+grid**2)))) * np.exp(-x**2 / (2 * (s**2+grid**2)))
            self.f_prior=lambda x,grid: (1 / (np.sqrt(2 * np.pi * grid**2))) * np.exp(-x**2 / (2 * grid**2))
        elif prior_type=='uniform':
            self.f_margin=lambda x,s,grid: (norm.cdf((x+grid)/s)-norm.cdf((x-grid)/s))/(2*grid)
            self.f_prior=lambda x,grid: (np.abs(x)<grid)/(2*grid)
        
        grid_min,grid_max=np.min(self.list_s)/10,2*np.max(self.data**2-self.list_s**2)
        if origin_point_mass==True:
            self.grid=[0.01*self.delta_x,grid_min]
        else:
            self.grid=[grid_min]
        para=grid_min
        for m in range(0,sup_num_grid):
            para=1.414*para
            if para<=grid_max:
                self.grid.append(para)
            else:
                break
        return
    
    def getL(self):
        # para_grid=getParaGrid(observed_data, list_s)
        K,n_data=len(self.grid),len(self.data)
        L=np.zeros((K,n_data))
        for k in range(0,K):
                L[k,:]=self.f_margin(self.data[:],self.list_s[:],self.grid[k])
        return L
    
    def EM(self,L,n_iter=400,lambda_0=10):
        (K,n_data)=L.shape
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
        return list_pi,list_log_likelihood,omega
    
    def fit(self,n_iter=400,lambda_0=10):
        L=self.getL()
        self.list_pi,self.list_log_likelihood,self.omega=self.EM(L,n_iter,lambda_0)
        return self.list_pi,self.list_log_likelihood,self.omega
    
    def solve(self,list_beta,list_s):
        marginal_p=self.list_pi[0]*self.f_margin(list_beta,list_s,self.grid[0])
        # lfdr=self.list_pi[0]*getGaussian(list_beta,0,list_s**2)*(self.f_prior(np.zeros_like(list_beta),self.grid[0]))**(1-self.origin_point_mass)
        
        for k in range(1,len(self.list_pi)):
            marginal_p+=self.list_pi[k]*self.f_margin(list_beta,list_s,self.grid[k])
        #     lfdr+=self.list_pi[k]*self.f_prior(np.zeros_like(list_beta),self.grid[k])*getGaussian(list_beta,0,list_s**2)
        # lfdr=lfdr/marginal_p
        null_component=getGaussian(list_beta, 0, list_s**2)*self.list_pi[0]
        lfdr=null_component/marginal_p
        alternative_component=marginal_p-null_component
        
        lfsr=[]
        list_positive_beta,list_negative_beta=list_beta[list_beta>=self.delta_x],list_beta[list_beta<=-self.delta_x]
        for j in range(0,len(list_beta)):
            Prob_positive=np.sum([self.list_pi[k]*np.sum(self.f_prior(list_positive_beta,self.grid[k])*getGaussian(list_beta[j],list_positive_beta,list_s[j]))*self.delta_x for k in range(0,len(self.list_pi))])
            Prob_negative=np.sum([self.list_pi[k]*np.sum(self.f_prior(list_negative_beta,self.grid[k])*getGaussian(list_beta[j],list_negative_beta,list_s[j]))*self.delta_x for k in range(0,len(self.list_pi))])
            lfsr.append(np.min([Prob_positive,Prob_negative])/marginal_p[j]+lfdr[j])
        return marginal_p,null_component,alternative_component,lfdr,lfsr
    def priorCDF(self,list_beta):
        g=np.zeros_like(list_beta)
        for k in range(self.list_pi):
            g+=self.list_pi[k]*norm.cdf(list_beta,loc=0,scale=self.grid[k])
        return g



