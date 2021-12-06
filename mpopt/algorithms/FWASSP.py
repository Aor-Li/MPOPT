import os
import time
import numpy as np

from ..population.fireworks import SSPFirework
from ..operator import operator as opt
from ..tools.distribution import MultiVariateNormalDistribution as MVND

EPS = 1e-6

class FWASSP(object):

    def __init__(self):
        # params
        self.fw_size = None
        self.sp_size = None
        self.init_scale = None
        self.max_not_improved = None

        # problem related params
        self.dim = None
        self.lb = None
        self.ub = None

        # population
        self.fireworks = None

        # firework parameters   
        self.spk_size = None
        self.init_scale = None
        self.mu_ratio = None
        self.weight_method = None
        self.adapt_lr = None
        self.coop_lr = None

        # states
        self.restart_status = None
        self.feature_points = None

        # load default params
        self.set_params(self.default_params())

    def default_params(self, benchmark=None):
        params = {}
        params['fw_size'] = 5
        params['sp_size'] = 300
        params['max_not_improved'] = 20

        # fireworks parameters
        params['spk_size']  = 60
        params['init_scale'] = 5
        params['mu_ratio'] = 0.5
        params['weight_method'] = 'Recombination'
        params['adapt_lr'] = 0.5
        params['coop_lr'] = 0.5
        
        return params

    def set_params(self, params):
        for param in params:
            setattr(self, param, params[param])
    
    def optimize(self, e):
        self.init(e)
        
        while not e.terminate():

            # independent cma evolving
            for fw in self.fireworks:
                fw.explode()
                fw.eval(e)
                fw.select()
                fw.adapt()

            # restart
            self.restart(e)

            # cooperate
            self.cooperate()

            # update
            self.update()

        return e.best_y
        
    def init(self, e):

        # record problem related params
        self.dim = opt.dim = e.obj.dim
        self.lb = opt.lb = e.obj.lb
        self.ub = opt.ub = e.obj.ub

        # init random seed
        self.seed = int(os.getpid()*time.time() % 1e8)
        np.random.seed(self.seed)

        # init states
        self.restart_status = np.array([False] * self.fw_size)
        self.feature_points = [list() for _ in range(self.fw_size)]

        # init population
        init_pop = np.random.uniform(self.lb, self.ub, [self.fw_size, self.dim])
        init_fit = e(init_pop)

        shifts = init_pop.copy()
        scales = [self.init_scale] * self.fw_size
        covs = [np.eye(self.dim) for _ in range(self.fw_size)]
        mvnds = [MVND(shifts[_,:], scales[_], covs[_]) for _ in range(self.fw_size)]

        for mvnd in mvnds:
            mvnd.decompose()

        self.fireworks = [SSPFirework(init_pop[i,:], 
                                      init_fit[i], 
                                      mvnds[i], 
                                      self.spk_size,
                                      mu_ratio=self.mu_ratio,
                                      weights=self.weight_method,
                                      lr=self.adapt_lr,
                                      lb=self.lb,
                                      ub=self.ub) for i in range(self.fw_size)]

    def restart(self, e):
        """ Decide fireworks to restart """
        # restart status
        self.restart_status = np.array([False] * self.fw_size)

        """ Independent Restart """
        for idx, fw in enumerate(self.fireworks):
            # 1. Low value variance
            if np.std(fw.spk_fit) < 1e-6:
                self.restart_status[idx] = True
            
            # 2. Low position variance
            if fw.new_mvnd.scale * (np.max(fw.new_mvnd.eigvals) ** 0.5) < 1e-6:
                self.restart_status[idx] = True
            
            # 3. Max not improved
            if fw.not_improved_fit > self.max_not_improved:
                self.restart_status[idx] = True

        """ Collaborated Restart """
        """
        # 4. Loser-out tournament
        left_iter = int(e.max_eval / self.spk_size)
        for i, fw in enumerate(self.fireworks):
            # ignore fireworks at first explosion
            if fw.val is None:
                continue
            
            # ignore fireworks not improved
            improve = fw.val - fw.new_val
            if improve < 0:
                continue

            # restart if not improved fast enough
            if improve * left_iter < fw.new_val - e.cur_y:
                self.restart_status[i] = True

        #print("r4  ", self.restart_status)
        """
        
        # 5. Covered by better firework
        sort_idx = np.argsort([fw.val for fw in self.fireworks])
        for i in range(len(sort_idx)):
            for j in range(i+1, len(sort_idx)):
                if self.restart_status[sort_idx[i]] or self.restart_status[sort_idx[j]]:
                    continue
                
                # alias
                fw1 = self.fireworks[sort_idx[i]]
                fw2 = self.fireworks[sort_idx[j]]
                mvnd1 = fw1.new_mvnd
                mvnd2 = fw2.new_mvnd

                dim = self.dim
                lam = int(fw2.nspk * fw2.mu_ratio)

                # check cover rate
                ys = (fw2.spk_pop[:lam,:] - mvnd1.shift[np.newaxis]) / mvnd1.scale
                zs = np.dot(ys, mvnd1.invsqrt_cov)
                r2 = np.sum(zs ** 2, axis=1)
                d2 = dim * (1 - 1/(4*dim) + 1/(21 * dim**2)) ** 2
                
                cover = r2 < d2
                cover_rate = np.sum(cover) / lam

                # restart if cover by 85%
                if cover_rate > 0.85:
                    self.restart_status[sort_idx[j]] = True
        
        #print("r5  ", self.restart_status)

        """ Restart Fireworks """
        for idx in range(self.fw_size):
            if not self.restart_status[idx]: 
                continue

            # generate new individual
            idv = np.random.uniform(self.lb, self.ub, [self.dim])            
            val = e(idv[np.newaxis,:])[0]

            shift = idv.copy()
            scale = self.init_scale
            cov = np.eye(self.dim)
            mvnd = MVND(shift, scale, cov)
            mvnd.decompose()

            self.fireworks[idx] = SSPFirework(idv, val, mvnd, self.spk_size, 
                                              mu_ratio=self.mu_ratio, 
                                              weights=self.weight_method,
                                              lr=self.adapt_lr,
                                              lb=self.lb, 
                                              ub=self.ub)

            # prepare for next generation
            fw = self.fireworks[idx]
            fw.new_idv = fw.idv
            fw.new_val = fw.val
            fw.new_mvnd = fw.mvnd
            fw.new_ps = fw.ps
            fw.new_pc = fw.pc
        
    def cooperate(self):
        # alias
        dim = self.dim
        
        """ Preparing """
        poss = np.array([fw.new_mvnd.shift for fw in self.fireworks])

        # pair-wise distance
        dist = np.zeros((self.fw_size, self.fw_size))
        for i in range(self.fw_size):
            for j in range(i+1, self.fw_size):
                d = np.sqrt(np.sum((poss[i,:] - poss[j,:]) ** 2))
                dist[i,j] = dist[j,i] = d
        
        # Unit vector
        unit_vec = np.zeros((self.fw_size, self.fw_size, self.dim))
        for i in range(self.fw_size):
            for j in range(self.fw_size):
                if i == j: continue
                unit_vec[i,j,:] = (poss[j,:] - poss[i,:]) / dist[i,j]

        # expected radius on direction
        expr = np.zeros((self.fw_size, self.fw_size))
        p0 = dim * (1 - 1/(4*dim) + 1/(21 * dim**2)) ** 2
        for i in range(self.fw_size):
            for j in range(self.fw_size):
                if i == j: continue
                pj = self.fireworks[i].new_mvnd.dispersion(poss[j,:])
                expr[i,j] = dist[i,j] * np.sqrt(p0 / pj)

        # firework rank
        rank = np.empty(self.fw_size)
        rank[np.argsort([fw.val for fw in self.fireworks])] = np.arange(self.fw_size)

        """ Get feature points """
        feature_points = [list() for _ in range(self.fw_size)]
        feature_direct_expr = [list() for _ in range(self.fw_size)]

        # obtain feature points
        for i in range(self.fw_size):
            for j in range(i+1, self.fw_size):
                comp = self.fuzzy_compare(self.fireworks[i], self.fireworks[j])

                r1 = dist[i,j] - expr[j,i]
                r2 = dist[j,i] - expr[i,j]
                mr1 = (expr[i,j] + dist[i,j] - expr[j,i]) / 2
                mr2 = (expr[j,i] + dist[j,i] - expr[i,j]) / 2

                # clipping
                abs_r1, abs_r2, abs_mr1, abs_mr2 = np.abs(r1), np.abs(r2), np.abs(mr1), np.abs(mr2)
                r1 *= np.clip(abs_r1, 0.5*expr[i,j], 2*expr[i,j]) / abs_r1
                r2 *= np.clip(abs_r2, 0.5*expr[j,i], 2*expr[j,i]) / abs_r2
                mr1 *= np.clip(abs_mr1, 0.5*expr[i,j], 2*expr[i,j]) / abs_mr1
                mr2 *= np.clip(abs_mr2, 0.5*expr[j,i], 2*expr[j,i]) / abs_mr2

                fp1 = poss[i,:] + unit_vec[i,j,:] * r1
                fp2 = poss[j,:] + unit_vec[j,i,:] * r2
                fp3 = poss[i,:] + unit_vec[i,j,:] * mr1
                fp4 = poss[j,:] + unit_vec[j,i,:] * mr2

                if comp == 1:
                    feature_points[j].append(fp2)
                    feature_direct_expr[j].append(expr[j,i])
                elif comp == -1:
                    feature_points[i].append(fp1)
                    feature_direct_expr[i].append(expr[i,j])
                else:
                    feature_points[i].append(fp3)
                    feature_direct_expr[i].append(expr[i,j])
                    feature_points[j].append(fp4)
                    feature_direct_expr[j].append(expr[j,i])

        """ Clip feature points in bound """
        for idx, fw in enumerate(self.fireworks):
            for j, fp in enumerate(feature_points[idx]):
                shift = fw.new_mvnd.shift
                vec = fp - shift
                out_ub = fp > self.ub
                out_lb = fp < self.lb

                if np.any(out_ub):
                    ubr = min(((self.ub * np.ones(self.dim))[out_ub] - shift[out_ub]) / vec[out_ub])
                else:
                    ubr = 1
                if np.any(out_lb):
                    lbr = min(((self.lb * np.ones(self.dim))[out_lb] - shift[out_lb]) / vec[out_lb])
                else:
                    lbr = 1

                feature_points[idx][j] = fw.mvnd.shift + min(ubr, lbr) * vec

        """ Select feature points """
        max_fps = 2
        for idx, fw in enumerate(self.fireworks):
            lam = len(feature_points[idx])
            if lam <= max_fps:
                continue
            else:
                fps = np.array(feature_points[idx])
                vec = fps - fw.new_mvnd.shift[np.newaxis,:]
                vec_norm = np.sqrt(np.sum(vec**2, axis=1))
                sort_idx = np.argsort(vec_norm/np.array(feature_direct_expr[idx]))

                if idx == 0:
                    feature_points[idx] = [fps[_] for _ in sort_idx[lam-max_fps:]]
                else:
                    feature_points[idx] = [fps[_] for _ in sort_idx[:max_fps]]

        self.feature_points = feature_points

        """ Cooperate fireworks """
        for i in range(self.fw_size):
            if len(feature_points[i]) == 0:
                continue
        
            # alias
            fw = self.fireworks[i]
            dim = self.dim
            fps = np.array(feature_points[i]).reshape(-1, dim)
            lam = fps.shape[0]
            lr = self.coop_lr

            # fit feature points
            new_shift = fw.new_mvnd.shift
            new_scale = fw.new_mvnd.scale

            ys = (fps - new_shift[np.newaxis,:]) / new_scale
            zs = np.matmul(ys, fw.new_mvnd.invsqrt_cov)

            d2 = dim * (1 - 1/(4*dim) + 1/(21 * dim**2)) ** 2
            r2 = np.sum(zs ** 2, axis=1)

            a = 1
            b = 1/d2 - (1/r2) * a

            new_cov = EPS * np.eye(dim)
            new_cov += (1 - lr + lr * np.mean(a) - EPS) * fw.new_mvnd.cov

            covs = np.matmul(ys[:,:,np.newaxis], ys[:,np.newaxis,:])
            for j in range(lam):
                new_cov += (lr * b[j]) / lam * covs[j,:,:]

            fw.new_mvnd = MVND(new_shift, new_scale, new_cov)
            fw.new_mvnd.decompose()

    def update(self):
        for fw in self.fireworks:
            fw.update()

    def fuzzy_compare(self, fw1, fw2):
        if fw1.spk_fit is None and fw2.spk_fit is None:
            return 0
        elif fw1.spk_fit is None:
            return -1
        elif fw2.spk_fit is None:
            return 1
        else:
            if max(fw1.spk_fit) < min(fw2.spk_fit):
                return 1
            elif min(fw1.spk_fit) > max(fw2.spk_fit):
                return -1
            else:
                return 0

