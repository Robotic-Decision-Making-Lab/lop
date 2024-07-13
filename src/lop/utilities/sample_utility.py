# sample_utility.py
# Writen Ian Rankin - July 2024
#
# This is a set of function that are helper functions for sampling.
# Basically a set of code that might be helpful to be reusable, but
# don't fit anywhere else.

import numpy as np

#try:
#    from numba import jit


## 100% taken from stack overflow (Paul Panzer)
# https://stackoverflow.com/questions/58566613/numpy-list-of-random-subsets-of-specified-size
# except for the numba tag
# @param A - the set to select from
# @param k - the number of sets to return
# @param n - the size of each set to return
#
# @return numpy array (k x n) of the unique sets
def without_repl(A,k,n):
    N = A.size
    out = np.empty((n,k),A.dtype)
    A = A.copy()
    for i in range(n):
        for j in range(k):
            l = np.random.randint(j,N)
            out[i,j] = A[l]
            A[l] = A[j]
            A[j] = out[i,j]
    return out

def pp_overdraw(A,k,n):
    N = len(A)
    p = np.linspace(1-(k-1)/N,1-1/N,k-1).prod()
    nn = int(n/p + 4*np.sqrt(n-n*p)) + 1
    out = np.random.randint(0,N,(nn,k))
    os_np = np.sort(out,1)
    valid = (os_np[:,:-1] != os_np[:,1:]).all(1)
    validx, = valid.nonzero()

    return pp_overdraw_for(os_np, n, k, nn, N, out, A, valid, validx)
    


def pp_overdraw_for(os_np, n, k, nn, N, out, A, valid, validx):
    
    while len(validx)<n: # very unlikely
        replace = np.random.randint(0,N,(nn-len(validx),k))
        rs = np.sort(replace,1)
        rvalid = (rs[:,:-1] != rs[:,1:]).all(1)
        out[~valid,:] = replace
        valid[~valid] = rvalid
        validx, = valid.nonzero()
    return A[out[validx[:n]]]

#except:
#    pass


## sample unique sets
# Wrapper to make it easier to use.
# @param s - the set to select from (if int treated as np.arange(s))
# @param num_sets - the number of unique sets of size set_size to return
# @param set_size - the size of the sets to return. Must be smaller than len(s)
#
# @return a numpy array of 
def sample_unique_sets(s, num_sets, size_set):
    if isinstance(s, int):
        s_len = s
        s = np.arange(s_len, dtype=int)
    else:
        s_len = len(s)

    
    #try:
    return pp_overdraw(s, size_set, num_sets)
    #except:
    #    raise NotImplementedError("Did not bother to implement sample unique sets without numba. \nHaha, future Ian you either got to do it now, or properly install numba.")

