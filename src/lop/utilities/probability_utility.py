# probability_utility.py
# Written Ian Rankin -  May 2024
#
# utility funcion for functions to handle probabilistic models.
# In this approximate CDF of gaussian models.
#
# This uses the approxcdf library
# https://approxcdf.readthedocs.io/en/latest/

import numpy as np

from scipy.stats import multivariate_normal
import scipy.special as spec

# https://approxcdf.readthedocs.io/en/latest/
# I cloned it myself to work with python 3.8 rather than 3.9
# minor change of reducing the min numpy version allowed.

try:
    import approxcdf
except:
    print('Cannot import approxcdf, any function which uses this, will not work')

# @param mu - the mean of the normal random distribution
# @param cov - the covariance of the normal random distribtuion
# @param method - [opt - default 'auto'] the method to calculate the cdf [auto, full, independent] 
# 
def calc_cdf(mu, cov, method='mvn'):
    if method == 'auto':
        method = 'mvn'
    
    if method == 'full':
        rv = multivariate_normal(mean=mu, cov=cov)
        
        
        zero_vector = np.zeros(len(mu))
        return rv.cdf(zero_vector)

    elif method == 'independent':
        
        return np.prod(spec.ndtr((0 - mu) / np.diagonal(cov)))


    elif method == 'switch':
        p = 1.0
        cov = np.copy(cov)
        mu = np.copy(mu)
        mod_cov = np.abs(np.copy(cov))
        np.fill_diagonal(mod_cov, 0)

        while cov.shape[0] > 1:
            idx = np.unravel_index(np.argmax(mod_cov), cov.shape)

            #pdb.set_trace()

            small_cov = np.array([  [cov[idx[0], idx[0]], cov[idx[0], idx[1]]], \
                                    [cov[idx[1], idx[0]], cov[idx[1], idx[1]]]])

            small_mu = np.array([mu[idx[0]], mu[idx[1]]])


            rv = multivariate_normal(mean=small_mu, cov=small_cov)
            p *= rv.cdf(np.array([0,0]))

            cov = np.delete(cov, idx, 0)
            cov = np.delete(cov, idx, 1)

            mod_cov = np.delete(mod_cov, idx, 0)
            mod_cov = np.delete(mod_cov, idx, 1)
            mu = np.delete(mu, idx, 0)

        if cov.shape[0] == 1:
            p *= spec.ndtr((0 - mu[0]) / cov[0,0])

        return p
    
    elif method == 'mvn':
        return approxcdf.mvn_cdf(-mu, cov)

