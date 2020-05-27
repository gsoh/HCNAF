"""
NAF experiments modified to work with HCNAF
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 17:13:18 2018

@author: chin-weihuang
"""



from sklearn.datasets import make_swiss_roll
from scipy.stats import multivariate_normal
import numpy as np
import torch


class Distr(object):
    
    hasenergyf = False
    hassplr = False
    
    def energy(self, x):
        raise NotImplementedError
    
    def sampler(self, x):
        raise NotImplementedError

class Mixture(Distr):
    """Makes a probabilistic mixture out of any distr_list where distr implements rvs and pdf."""

    hasenergyf = True
    hassplr = True
    
    def __init__(self, probs, distr_list):
        self.probs = np.asarray(probs)
        self.distr_list = distr_list

    def energy(self, x):
        pdf = np.nasarray([distr.pdf(x) for distr in self.distr_list])
        assert pdf.shape == (len(self.distr_list), len(x))
        return np.dot(self.probs, pdf)

    def sampler(self, n, **kwargs):
        counts = np.random.multinomial(n, self.probs)
        assert np.sum(counts) == n
        assert len(counts == self.probs)
        samples = []
        for k, distr in zip(counts, self.distr_list):
            samples.append(distr.rvs(k))

        samples = np.vstack(samples)
        np.random.shuffle(samples)
        return torch.from_numpy(samples.astype('float32'))


class SwissRoll(Distr):
    
    hasenergyf = False
    hassplr = True
    
    def __init__(self, noise=0.5):
        self.noise = noise
        
    def sampler(self, n):
        return torch.from_numpy(
            make_swiss_roll(n, self.noise)[0][:,[0,2]].astype('float32') / 3.)


class TenByTen(Mixture):
    
    def __init__(self):
        
        nmodesperdim = 10
        grid = np.linspace(-5,5,nmodesperdim)
        grid = np.meshgrid(grid,grid)
        grid = np.concatenate([grid[0].reshape(nmodesperdim**2,1),
                               grid[1].reshape(nmodesperdim**2,1)],1)
        
        super(TenByTen, self).__init__(
            np.ones(nmodesperdim**2) / float(nmodesperdim**2),
            [multivariate_normal(mean, 1/float(nmodesperdim*np.log(nmodesperdim))) for mean in grid])


class NByN(Mixture):
    '''
    Modification of TenByTen to create n*n gaussians
    '''
    
    def __init__(self, nmodesperdim=10):
    
        grid = np.linspace(-5,5,nmodesperdim)
        grid = np.meshgrid(grid,grid)
        grid = np.concatenate([grid[0].reshape(nmodesperdim**2,1),
                               grid[1].reshape(nmodesperdim**2,1)],1)
        
        super(NByN, self).__init__(
            np.ones(nmodesperdim**2) / float(nmodesperdim**2),
            [multivariate_normal(mean, 1/float(nmodesperdim*np.log(nmodesperdim))) for mean in grid])

    

class FourDiamond(Mixture):
    
    def __init__(self):
        
        super(FourDiamond, self).__init__(
            [0.1, 0.3, 0.4, 0.2], 
            [multivariate_normal([-5., 0]),
             multivariate_normal([5., 0]),
             multivariate_normal([0, 5.]),
             multivariate_normal([0, -5.])])
    
    
    
                            