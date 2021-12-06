""" General Operators for Swarms """
import sys
import numpy as np
from numpy.lib.ufunclike import isposinf

from mpopt.tools.distribution import MultiVariateNormalDistribution as MVND

# logging
import logging
fmt = "[%(asctime)s]-[OptLogger]-[%(levelname)s]-[%(filename)s]-[%(lineno)s]:%(message)s"
logging.basicConfig(level=logging.CRITICAL, format=fmt, stream=sys.stdout)
logger = logging.getLogger(__name__)

lb = -float('inf')
ub = float('inf')


""" Mapping Rules """
# Random Re-map
def random_map(samples, lb=-float('inf'), ub=float('inf')):
    in_bound = (samples > lb) * (samples < ub)
    if not in_bound.all():
        rand_samples = np.random.uniform(lb, ub, samples.shape)
        samples = in_bound * samples + (1 - in_bound) * rand_samples
    return samples

# Mirror Re-map
def mirror_map(samples, lb=-float('inf'), ub=float('inf')):

    if not (np.isneginf(lb) or np.isposinf(ub)):

        if lb - np.min(samples) > 10 * (ub - lb):
            logger.warning("Sample is {:.2e} times of the bound size below feasible space).".format(
                (lb - np.min(samples)) / (ub - lb)
            ))

        if np.max(samples) - ub > 10 * (ub - lb):
            logger.warning("Sample is {:.2e} times of the bound size above feasible space).".format(
                (np.max(samples) - ub) / (ub - lb)
            ))

    while True:
        sample_ub = samples > ub
        sample_lb = samples < lb
        if sample_ub.any():
            samples[sample_ub] = 2*ub - samples[sample_ub]
        if sample_lb.any():
            samples[sample_lb] = 2*lb - samples[sample_lb]
        if not (sample_ub.any() or sample_lb.any()):
            break
    return samples

# Mod Re-map
def mod_map(samples):
    pass


""" Explosion Methods """
def box_explode(idv, amp, num_spk, remap=random_map):
    dim = idv.shape[-1]
    bias = np.random.uniform(-1, 1, [num_spk, dim])
    spks = idv[np.newaxis, :] + amp * bias
    spks = remap(spks)
    return spks

def gaussian_explode(mvnd, nspk, remap=random_map):
    return mvnd.sample(nspk, remap)
    

""" Mutation Methods """
# Guided Mutation
def guided_mutate(idv, spk_pop, spk_fit, gm_ratio, remap=random_map):
    num_spk = spk_pop.shape[0]
    
    top_num = int(num_spk * gm_ratio)
    sort_idx = np.argsort(spk_fit)
    top_idx = sort_idx[:top_num]
    btm_idx = sort_idx[-top_num:]

    top_mean = np.mean(spk_pop[top_idx,:], axis=0)
    btm_mean = np.mean(spk_pop[btm_idx,:], axis=0)
    delta =  top_mean - btm_mean

    ms = remap(idv + delta).reshape(1, -1)
    
    return ms

""" Selection Methods """
# Elite Selection
def elite_select(pop, fit, topk=1):
    if topk == 1:
        min_idx = np.argmin(fit)
        return pop[min_idx, :], fit[min_idx]
    else:
        sort_idx = np.argsort(fit)
        top_idx = sort_idx[:topk]
        return pop[top_idx, :], fit[top_idx]

def paired_select(pop, fit, new_pop, new_fit, return_indexes=False):
    indexes = np.where(fit > new_fit)[0]
    pop[indexes] = new_pop[indexes]
    if return_indexes:
        return pop, indexes
    else:
        return pop


""" Differential Evolution Methods """
def current_to_pbest_mutation(pop, fit, f, p, archive=[], remap=mirror_map):
    
    # prepare
    dim = pop.shape[1]
    pop_size = pop.shape[0]
    
    archive = np.array(archive).reshape(-1, dim)
    archive_size = archive.shape[0]

    if pop_size < 4:
        raise Exception("Population size too small for DE mutation.")
    
    # find pbest
    p_best = []
    for p_i in p:
        best_index = np.argsort(fit)[:max(2, int(round(p_i*pop_size)))]
        p_best.append(np.random.choice(best_index))
    p_best = np.array(p_best)

    # find random parents
    choices = np.indices((pop_size, pop_size))[1]
    mask = np.ones(choices.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    choices = choices[mask].reshape(pop_size, pop_size - 1)
    parents = np.array([np.random.choice(row, 2, replace=False) for row in choices])

    # mutate
    mutated = pop + f * (pop[p_best] - pop)

    # get X_{r2, G} from pop or archive
    r =  archive_size / (archive_size + pop_size)
    if np.random.rand() < r:
        x2 = archive[np.random.choice(archive_size)]
        mutated += f * (pop[parents[:, 0]] - x2)
    else:
        mutated += f * (pop[parents[:, 0]] - pop[parents[:, 1]])

    mutated = remap(mutated)
    return mutated

def crossover(pop, mutated, cr):
    chosen = np.random.rand(*pop.shape)
    j_rand = np.random.randint(0, pop.shape[1])
    chosen[j_rand::pop.shape[1]] = 0
    return np.where(chosen <= cr, mutated, pop)

""" Others """
def rank(vals):
    """ Rank the values """
    ranks = np.empty(len(vals))
    ranks[np.argsort(vals)] = np.arange(vals)
    return ranks