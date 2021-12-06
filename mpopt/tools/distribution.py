#!/usr/bin/env python
import logging
import warnings
import numpy as np
from numpy.lib.ufunclike import isposinf
from scipy.stats import chi

EPS = 1e-8


class MultiVariateNormalDistribution(object):

    def __init__(self, shift, scale, cov, dim=None):
        # main components
        self.shift = shift
        self.scale = scale
        self.cov = cov
        
        # params
        self.dim = dim if dim is not None else shift.shape[0]

        # states
        self.eigvecs = None
        self.eigvals = None
        self.inv_cov = None
        self.invsqrt_cov = None
        self.rev = None

        # decompose cov
        self.decomposed = False

    def decompose(self, force_positive=False, shrinkage=0, rescale=None, bound_size=float('inf')):

        # force symmetric 
        self.cov = (self.cov + self.cov.T) / 2.0

        # solve
        self.eigvals, self.eigvecs = np.linalg.eigh(self.cov)

        # force positive definite
        if force_positive:
            self.eigvals = np.clip(self.eigvals, EPS, None)

        # shrinkage
        if shrinkage > 0:
            trace_cov = np.sum(self.eigvals)
            self.eigvals = (1 - shrinkage) * self.eigvals + shrinkage * (trace_cov / self.dim) * np.ones(self.dim)

        # rescale
        if rescale is not None:
            ratio = (self.scale / rescale) ** 2

            self.cov *= ratio
            self.eigvals *= ratio
            self.scale = rescale

        # restrict max length
        base_length = chi.mean(self.dim) + 2.0 * chi.std(self.dim)
        max_eigval = (bound_size / base_length) ** 2
        self.eigvals = np.clip(self.eigvals, EPS, max_eigval)

        # computing
        with warnings.catch_warnings(record=True) as w:
            self.cov = np.dot(self.eigvecs, np.diag(self.eigvals)).dot(self.eigvecs.T)
            
            #inv cov
            self.inv_cov = np.dot(self.eigvecs, np.diag(self.eigvals ** -1)).dot(self.eigvecs.T)

            # inv sqrt cov
            self.invsqrt_cov = np.dot(self.eigvecs, np.diag(self.eigvals ** -0.5)).dot(self.eigvecs.T)

            # sqrt cov
            self.sqrt_cov = np.dot(self.eigvecs, np.diag(self.eigvals ** 0.5)).dot(self.eigvecs.T)
        
            # reverse projection matrix
            self.rev = np.dot(np.diag(self.eigvals ** -0.5), self.eigvecs.T)

            # handle warnings
            if len(w) > 0:
                print("Eigvals: ", self.eigvals)
                print("Sigma: ", self.scale)
                raise Exception("Negative eigval")
        
    def sample(self, num, remap=None):
        if not self.decomposed:
            self.decompose()

        bias = np.random.normal(size=[num, self.dim])
        amp_bias = self.scale * (self.eigvals ** 0.5)[np.newaxis,:] * bias
        rot_bias = np.dot(amp_bias, self.eigvecs.T)
        samples = self.shift[np.newaxis,:] + rot_bias
        if remap is not None:
            samples = remap(samples)

        return samples

    def dispersion(self, X):
        x = X.reshape(-1, self.dim)
        y = x - self.shift[np.newaxis, :]
        z = np.dot(y / self.scale, self.invsqrt_cov)
        dens = np.sum(z ** 2, axis=1)
        if len(X.shape) == 1:
            dens = dens[0]
        return dens

