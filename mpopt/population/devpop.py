import sys
import numpy as np

from .base import BaseEDAFirework
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

# logging
import logging
fmt = "%(asctime)s - PopLogger - %(levelname)s: %(message)s"
#logging.basicConfig(level=logging.DEBUG, format=fmt, stream=sys.stdout)
logging.basicConfig(level=logging.WARNING, format=fmt, stream=sys.stdout)
logger = logging.getLogger(__name__)

EPS = 1e-8


class HCFirework(BaseEDAFirework):
    """ Firework for HCFWA """
    def __init__(self,
                 mvnd,
                 nspk,
                 mu_ratio=0.5,
                 weights='Recombination',
                 lb=-float('inf'), 
                 ub=float('inf')):
        super().__init__(mvnd, lb=lb, ub=ub)

        # states of population
        self.mvnd = mvnd
        self.ad_mvnd = None
        self.cp_mvnd = None
        self.new_mvnd = None

        self.nspk = nspk
        self.new_nspk = None

        self.ps = np.zeros(self.dim)
        self.new_ps = None

        self.pc = np.zeros(self.dim)
        self.new_pc = None

        # states of history
        self.best_idv = None
        self.best_val = None

        self.num_iter = 0
        self.not_improved_count = 0

        # parameters
        self.mu_ratio = mu_ratio
        self.weights = weights
    
    def eval(self, e):
        self.spk_fit = e(self.spk_pop)
    
    def remap(self, samples):
        return opt.mirror_map(samples, lb=self.lb, ub=self.ub)
    
    def select(self):
        if self.best_idv is None:
            tot_pop = self.spk_pop
            tot_fit = self.spk_fit
        else:
            tot_pop = np.vstack([self.best_idv, self.spk_pop])
            tot_fit = np.concatenate([np.array([self.best_val]), self.spk_fit])
        
        new_best_idv, new_best_val = opt.elite_select(tot_pop, tot_fit)
        
        if self.best_val is not None and self.best_val - new_best_val > EPS:
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1

        self.best_idv, self.best_val = new_best_idv, new_best_val

    def explode(self):
        self.spk_pop = opt.gaussian_explode(self.mvnd, self.nspk, remap=self.remap)

    def adapt(self, cmu=None):
        # alias
        lam, dim = self.spk_pop.shape
        shift = self.mvnd.shift
        scale = self.mvnd.scale
        cov = self.mvnd.cov
        mu = int(np.ceil(lam * self.mu_ratio))

        # sort sparks
        sort_idx = np.argsort(self.spk_fit)
        self.spk_pop = self.spk_pop[sort_idx]
        self.spk_fit = self.spk_fit[sort_idx]

        # compute weights
        w = np.log(mu + 0.5) - np.log(1+np.arange(lam))
        w[w<0] = 0
        w = w / np.sum(w)

        # compute parameters
        mueff = np.sum(w)**2/np.sum(w**2)
        cs = (mueff+2)/(dim+mueff+5)
        csn = (cs*(2-cs)*mueff)**0.5 / scale
        cmu = min([0.8, 2*(mueff-2+1/mueff) / ((dim+2)**2 + mueff)]) if cmu is None else cmu

        # adapt shift
        new_shift = np.sum(w[:,np.newaxis] * self.spk_pop, axis=0)

        # adapt evolution path
        y = new_shift - shift
        z = np.dot(y, self.mvnd.invsqrt_cov)

        new_ps = (1-cs) * self.ps + csn * z
        new_cov = EPS * np.eye(dim) + cov * (1 - EPS - cmu * sum(w))

        bias = self.spk_pop - shift[np.newaxis,:]
        covs = np.matmul(bias[:,:,np.newaxis], bias[:,np.newaxis,:])
        for i in range(lam):
            if w[i] < 0:
                raise Exception("Negative weight found")
            new_cov += (w[i] * cmu / scale ** 2) * covs[i,:,:]
        
        # adapt scale
        damps = 2 * mueff / lam + 0.3 + cs
        cn, sum_square_ps = cs/damps, np.sum(new_ps ** 2)
        new_scale = scale * np.exp(min(1, cn * (sum_square_ps / dim - 1) / 2))

        # mean distance to mean
        new_bias = self.spk_pop[:mu,:] - new_shift[np.newaxis,:]
        mean_scale = np.mean(np.linalg.norm(new_bias, axis=1))

        # update new mvnd and evolution paths
        self.ad_mvnd = MVND(new_shift, new_scale, new_cov)
        try:
            self.ad_mvnd.decompose(force_positive=True, shrinkage=1e-4, rescale=mean_scale)
        except Exception as e:
            logger.debug("-------------------------------------------------------")
            logger.debug("Original Cor Matrix:", self.mvnd.cov)
            logger.debug("Original Eigvals :", self.mvnd.eigvals)
            logger.debug("Original Scale   :", self.mvnd.scale)
            raise e

        self.new_mvnd = self.ad_mvnd
        self.new_ps = new_ps

        # adapt other states
        self.num_iter += 1
    
    def update(self):
        self.mvnd = self.new_mvnd
        self.ps = self.new_ps
        self.pc = self.new_pc

    def evolve(self, e):
        self.explode()
        self.eval(e)
        self.select()
        self.adapt()
        self.update()

class RankOneHCFirework(BaseEDAFirework):
    """ Firework for HCFWA """
    def __init__(self,
                 mvnd,
                 nspk,
                 mu_ratio=0.5,
                 weights='Recombination',
                 lb=-float('inf'), 
                 ub=float('inf')):
        super().__init__(mvnd, lb=lb, ub=ub)

        # states of population
        self.mvnd = mvnd
        self.ad_mvnd = None
        self.cp_mvnd = None
        self.new_mvnd = None

        self.nspk = nspk
        self.new_nspk = None

        self.ps = np.zeros(self.dim)
        self.new_ps = None

        self.pc = np.zeros(self.dim)
        self.new_pc = None

        # states of history
        self.best_idv = None
        self.best_val = None

        self.num_iter = 0
        self.not_improved_count = 0

        # parameters
        self.mu_ratio = mu_ratio
        self.weights = weights
    
    def eval(self, e):
        self.spk_fit = e(self.spk_pop)
    
    def remap(self, samples):
        return opt.mirror_map(samples, lb=self.lb, ub=self.ub)
    
    def select(self):
        if self.best_idv is None:
            tot_pop = self.spk_pop
            tot_fit = self.spk_fit
        else:
            tot_pop = np.vstack([self.best_idv, self.spk_pop])
            tot_fit = np.concatenate([np.array([self.best_val]), self.spk_fit])
        
        new_best_idv, new_best_val = opt.elite_select(tot_pop, tot_fit)
        
        if self.best_val is not None and self.best_val - new_best_val > EPS:
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1

        self.best_idv, self.best_val = new_best_idv, new_best_val

    def explode(self):
        self.spk_pop = opt.gaussian_explode(self.mvnd, self.nspk, remap=self.remap)

    def adapt(self, cmu=None):
        # alias
        lam, dim = self.spk_pop.shape
        shift = self.mvnd.shift
        scale = self.mvnd.scale
        cov = self.mvnd.cov
        num_iter = self.num_iter
        mu = int(np.ceil(lam * self.mu_ratio))

        # sort sparks
        sort_idx = np.argsort(self.spk_fit)
        self.spk_pop = self.spk_pop[sort_idx]
        self.spk_fit = self.spk_fit[sort_idx]

        # compute weights
        w = np.log(mu + 0.5) - np.log(1+np.arange(lam))
        w[w<0] = 0
        w = w / np.sum(w)

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
        new_scale = scale * np.exp(min(1, cn * (sum_square_ps / dim - 1) / 2))

        # mean distance to mean
        new_bias = self.spk_pop[:mu,:] - new_shift[np.newaxis,:]
        mean_scale = np.mean(np.linalg.norm(new_bias, axis=1))

        # update new mvnd and evolution paths
        self.ad_mvnd = MVND(new_shift, new_scale, new_cov)
        try:
            self.ad_mvnd.decompose(force_positive=True, shrinkage=1e-4, rescale=mean_scale)
        except Exception as e:
            logger.debug("-------------------------------------------------------")
            logger.debug("Original Cor Matrix:", self.mvnd.cov)
            logger.debug("Original Eigvals :", self.mvnd.eigvals)
            logger.debug("Original Scale   :", self.mvnd.scale)
            raise e

        self.new_mvnd = self.ad_mvnd
        self.new_ps = new_ps
        self.new_pc = new_pc

        # adapt other states
        self.num_iter += 1
    
    def update(self):
        self.mvnd = self.new_mvnd
        self.ps = self.new_ps
        self.pc = self.new_pc

    def evolve(self, e):
        self.explode()
        self.eval(e)
        self.select()
        self.adapt()
        self.update()
