import numpy as np

from .base import BaseFirework
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

EPS = 1e-8

class SSPFirework(BaseFirework):
    """ Fireworks with Covariance Matrix Adaption used in Search Space Partition Method """
    def __init__(self, 
                 idv, 
                 val, 
                 mvnd, 
                 nspk,
                 mu_ratio=0.5,
                 weights='Recombination',
                 lr=1.0,
                 lb=-float('inf'), 
                 ub=float('inf')):
        super().__init__(idv, val, lb=lb, ub=ub)

        # states
        self.mvnd = mvnd
        self.new_mvnd = None

        self.nspk = nspk
        self.new_nspk = None

        self.ps = np.zeros(self.dim)
        self.new_ps = None

        self.pc = np.zeros(self.dim)
        self.new_pc = None

        self.num_iter = 0

        self.lowest_sample_mean = val
        self.not_improved_mean = 0
        self.not_improved_fit = 0

        # parameters
        self.mu_ratio = mu_ratio
        self.weights = weights
        self.lr = lr

    def eval(self, e):
        self.spk_fit = e(self.spk_pop)

    def select(self):
        tot_pop = np.vstack([self.idv, self.spk_pop])
        tot_fit = np.concatenate([np.array([self.val]), self.spk_fit])
        self.new_idv, self.new_val = opt.elite_select(tot_pop, tot_fit)

    def explode(self):
        self.spk_pop = opt.gaussian_explode(self.mvnd, self.nspk, remap=self.remap)
        
    def adapt(self, mu_ratio=None, weights=None, lr=None):
        # default parameters
        if mu_ratio is None:
            mu_ratio = self.mu_ratio
        if weights is None:
            weights = self.weights
        if lr is None:
            lr = self.lr

        # alias
        lam, dim = self.spk_pop.shape
        shift = self.mvnd.shift
        scale = self.mvnd.scale
        cov = self.mvnd.cov
        num_iter = self.num_iter
        mu = int(np.ceil(lam * mu_ratio))

        # sort sparks
        sort_idx = np.argsort(self.spk_fit)
        self.spk_pop = self.spk_pop[sort_idx]
        self.spk_fit = self.spk_fit[sort_idx]

        # compute weights
        if isinstance(weights, np.ndarray):
            w = weights
        
        elif weights == 'Recombination':
            w = np.log(mu + 0.5) - np.log(1+np.arange(lam))
            w[w<0] = 0
            w = w / np.sum(w)

        elif weights == 'Mean':
            w = np.zeros(lam)
            w[:mu] = 1.0 / mu
        
        else:
            raise Exception("Weights method '{}' not implemented.".format(weights))

        # compute parameters
        mueff = np.sum(w)**2/np.sum(w**2)

        cc = (4+mueff/dim) / (dim+4+2*mueff/dim)
        cs = (mueff+2)/(dim+mueff+5)
        ccn = (cc*(2-cc)*mueff)**0.5 / scale
        csn = (cs*(2-cs)*mueff)**0.5 / scale

        hsig = int((np.sum(self.ps**2)/dim/(1-(1-cs)**(2*num_iter+1)))<2+4./(dim+1))
        c1 = 2/((dim+1.3)**2+mueff)
        c1a = c1*(1-(1-hsig**2) * cc * (2-cc))
        cmu = min([1-c1, 2*(mueff-2+1/mueff) / ((dim+2)**2 + mueff)])

        # adapt shift
        new_shift = np.sum(w[:,np.newaxis] * self.spk_pop, axis=0)

        # adapt evolution path
        y = new_shift - shift
        z = np.dot(y, self.mvnd.invsqrt_cov)

        new_pc = (1-cc) * self.pc + ccn * hsig * y
        new_ps = (1-cs) * self.ps + csn * z

        # adapt cov
        new_cov = EPS * np.eye(dim) + cov * (1 - EPS - c1a - cmu * sum(w))
        new_cov += c1 * np.dot(new_pc[:,np.newaxis], new_pc[np.newaxis,:])
          
        bias = self.spk_pop - shift[np.newaxis,:]
        covs = np.matmul(bias[:,:,np.newaxis], bias[:,np.newaxis,:])
        for i in range(lam):
            if w[i] < 0:
                raise Exception("Negative weight found")
            new_cov += (w[i] * cmu / scale ** 2) * covs[i,:,:]
        
        # adapt scale
        damps = 2 * mueff / lam + 0.3 + cs
        cn, sum_square_ps = cs/damps, np.sum(new_ps ** 2)
        #new_scale = scale * np.exp(min(1, cn * (sum_square_ps / dim - 1) / 2))

        new_scale = scale * np.exp(0.5 * min(1, cn * (sum_square_ps / dim - 1) / 2))

        # slow down by lr
        lr = 1.0
        new_shift = (1.0 - lr) * shift + lr * new_shift
        new_scale = np.exp((1.0 - lr) * np.log(scale) + lr * np.log(new_scale))
        new_cov = (1.0 - lr) * cov + lr * new_cov

        self.new_mvnd = MVND(new_shift, new_scale, new_cov)
        self.new_mvnd.decompose()

        # mean distance to mean
        new_bias = self.spk_pop[:mu,:] - new_shift[np.newaxis,:]
        mean_scale = np.mean(np.linalg.norm(new_bias, axis=1))

        # update new mvnd and evolution paths
        self.ad_mvnd = MVND(new_shift, new_scale, new_cov)
        self.ad_mvnd.decompose(force_positive=True, shrinkage=1e-4, rescale=mean_scale)
        
        self.new_mvnd = self.ad_mvnd
        self.new_ps = new_ps
        self.new_pc = new_pc

        # adapt other states
        self.num_iter += 1
        
        if self.val - self.new_val > EPS:
            self.not_improved_fit = 0
        else:
            self.not_improved_fit += 1

        if np.mean(self.spk_fit) < self.lowest_sample_mean - EPS:
            self.lowest_sample_mean = np.mean(self.spk_fit)
            self.not_improved_mean = 0
        else:
            self.not_improved_mean += 1

    def update(self):
        # update population
        self.idv = self.new_idv
        self.val = self.new_val

        # update distribution
        self.mvnd = self.new_mvnd

        # update states
        self.ps = self.new_ps
        self.pc = self.new_pc

    def evolve(self, e):
        self.explode()
        self.eval(e)
        self.select()
        self.adapt()
        self.update()
