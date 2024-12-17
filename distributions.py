# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:11:25 2024

@author: LZ166
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

class Spiky(rv_continuous):
    def _pdf(self, x):
        # Define the probability density function (PDF) here
        return 0.4*getGaussian(x, 0, 0.25**2)+0.2*getGaussian(x, 0, 0.5**2)+0.2*getGaussian(x, 0, 1)+0.2*getGaussian(x, 0, 2)
class NearNormal(rv_continuous):
    def _pdf(self, x):
        return 2/3*getGaussian(x,0,1)+1/3*getGaussian(x,0,2**2)

class Skew(rv_continuous):
    def _pdf(self, x):
        return 1/4*getGaussian(x, -2,2**2)+1/4*getGaussian(x,-1, 1.5**2)+1/3*getGaussian(x,0, 1**2)+1/6*getGaussian(x,1, 1)

class BigNormal(rv_continuous):
    def _pdf(self, x):
        return getGaussian(x,0,4**2)

class Bimodal(rv_continuous):
    def _pdf(self, x):
        return 1/2*getGaussian(x,-2,1**2)+1/2*getGaussian(x,2,1**2)

class FlatTop(rv_continuous):
     def _pdf(self, x):
         return 1/7*(getGaussian(x, -1.5,0.5**2)+getGaussian(x, -1,0.5**2)+getGaussian(x, -0.5,0.5**2)+getGaussian(x, 0,0.5**2)+getGaussian(x, 0.5,0.5**2)+getGaussian(x, 1,0.5**2)+getGaussian(x, 1.5,0.5**2))

class StandardNormal(rv_continuous):
    def _pdf(self, x):
        return getGaussian(x, 0, 1)


def getGaussian(x, mu, sigma_sq):
    return (1 / (np.sqrt(2 * np.pi * sigma_sq))) * np.exp(-((x - mu)**2) / (2 * sigma_sq))

if __name__=='main':
    bins = np.arange(-6, 6, 0.1)


    n_betas=1000
    list_beta=np.linspace(-6,6,n_betas)
    spiky=Spiky(name='custom_dist')
    nearNormal=NearNormal(name='custom_dist')
    skew=Skew(name='custom_dist')
    bigNormal=BigNormal(name='custom_dist')
    bimodal=Bimodal(name='custom_dist')
    flatTop=FlatTop(name='custom_dist')

    density_spiky = spiky.pdf(list_beta)
    density_nearNormal = nearNormal.pdf(list_beta)
    density_skew = skew.pdf(list_beta)
    density_bigNormal = bigNormal.pdf(list_beta)
    density_bimodal = bimodal.pdf(list_beta)
    density_flatTop = flatTop.pdf(list_beta)
    densities=[density_spiky,density_nearNormal,density_flatTop,density_skew,density_bigNormal,density_bimodal]

    titles=['spiky','near normal', 'flat top', 'skew', 'big normal', 'bimodal']
    plt.plot(list_beta,density_spiky)

    num_subfigures = 6
    fig, axes = plt.subplots(1, num_subfigures, figsize=(num_subfigures*3, 6))
    y_min, y_max = 0, 0.8
    # Plot each subfigure
    for i in range(num_subfigures):
        axes[i].plot(list_beta,densities[i],color='black',linewidth=3)
        axes[i].set_title(titles[i])
        axes[i].set_aspect(aspect='auto')
        axes[i].set_ylim(y_min, y_max)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.text(0.5, 0.001, 'z', ha='center')
    fig.text(0.004, 0.5, 'density', va='center', rotation='vertical')

    plt.savefig('distributions.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()