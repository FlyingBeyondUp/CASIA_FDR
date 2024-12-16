# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:30:33 2024

@author: LZ166
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import CubicSpline

# Step 1: Generate beta coefficients from a normal distribution
n_betas = 2000  # Number of beta coefficients
# beta = np.random.normal(loc=0, scale=1, size=n_betas)

beta=[np.random.randn(1).item() if np.random.rand(1).item()>0.8 else 0 for i in range(0,n_betas)]
# Step 2: Construct observed data by adding noise
noise = np.random.normal(loc=0, scale=1, size=n_betas)  # Noise added to beta
observed_beta = beta + noise
bins = np.arange(-6, 6, 0.1)
binned_counts, _ = np.histogram(observed_beta, bins=bins)
bins=bins[:-1]

def f0p(x_values,mu,sigma_sq,pi0):
    return np.exp(-(x_values-mu)**2/(2*sigma_sq))*pi0/np.sqrt(2*np.pi*sigma_sq)

def solveTwoGroupModel(data,bins,binned_counts,poly_degree=7):
    # f(z)=pi_0f_0(z)+pi_1f_1(z)
    # f(z) is approximated by Poisson regression
    poly = PolynomialFeatures(poly_degree)
    X_poly = poly.fit_transform(bins.reshape(-1, 1))
    X_poly = sm.add_constant(X_poly)
    y = binned_counts
    poisson_model = sm.GLM(y, X_poly, family=sm.families.Poisson()).fit()
    
    predicted_counts = poisson_model.predict(X_poly)
    delta_x=bins[1]-bins[0] 
    predicted_density=predicted_counts/(np.sum(predicted_counts)*delta_x)
    spl_y=CubicSpline(bins,np.log(predicted_density))
    
    # f_0(z) is approximated by a Normal distribution centered at origin
    sigma_sq=-1/spl_y(0,nu=2)
    mu=sigma_sq*spl_y(0,nu=1)
    pi0=np.exp(spl_y(0)+1/2*(mu/sigma_sq+np.log(2*np.pi*sigma_sq)))
    pi0=pi0 if pi0<=1 else 1
    
    lfdr=pi0*f0p(bins,mu,sigma_sq,pi0)/(np.exp(spl_y(bins))+1e-5)
    return spl_y,lfdr,predicted_density,[mu,sigma_sq,pi0]
spl_y,lfdr,predicted_density,[mu,sigma_sq,pi0]=solveTwoGroupModel(observed_beta, bins,binned_counts)


# Step 7: Plotting the results
plt.figure(figsize=(10, 6))
plt.hist( observed_beta,bins=len(bins),density=True, alpha=0.5, label='Observed Counts', color='blue')
plt.plot(bins, predicted_density, label='Fitted Polynomial Poisson Model', color='red', linewidth=2)
plt.plot(bins,pi0*f0p(bins,mu,sigma_sq,pi0),label='null distribution')
plt.plot(bins,np.exp(spl_y(bins))-pi0*f0p(bins,mu,sigma_sq,pi0),label='alternative distribution')
plt.title('Polynomial Poisson Regression Fit to Binned Observed Data')
plt.xlabel('Value')
plt.ylabel('Counts ratio')
plt.legend()
plt.savefig('data_2group.pdf', bbox_inches='tight')

plt.figure()
plt.plot(bins,lfdr)
plt.xlabel('beta')
plt.ylabel('local false discovery rate')
plt.legend()
plt.savefig('lfdr_2group.pdf', bbox_inches='tight')




       
      











