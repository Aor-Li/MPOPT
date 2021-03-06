#!/usr/bin/env python
""" Script for testing algorithms on given benchmarks """

import os
import time
import json
import copy
import argparse
import importlib
import numpy as np
import pickle as pkl
from icecream import ic
from functools import wraps
from multiprocessing import Pool

from mpopt.benchmarks.benchmark import Benchmark


def parsing():
    
    # experiment setting for benchmark testing
    parser = argparse.ArgumentParser(description="Benchmark testing for SIOA")

    # benchmark params
    # fmt: off
    parser.add_argument("--benchmark", 
                        "-b", 
                        help="Benchmark name")
    
    parser.add_argument("--dim", 
                        "-d", 
                        type=int, 
                        help="Benchmark dimension")

    parser.add_argument("--funcid",
                        "-f",
                        default=0,
                        type=int,
                        help="Benchmark function, 0 for all functions in benchmark")

    # agortihm params
    parser.add_argument("--alg", 
                        "-a", 
                        help="Algorithm name")

    parser.add_argument("--apd", 
                        "--algparams_dict", 
                        default="{}", 
                        help="Algorithm parameters diction.")

    # testing params
    parser.add_argument("--name", 
                        "-n", 
                        default="", 
                        help="Name of experiment")

    parser.add_argument("--rep", 
                        "-r", 
                        default=1, 
                        type=int, 
                        help="Repetition of each problem")
    
    parser.add_argument("--multiprocess",
                        "-m",
                        default=1,
                        type=int,
                        help="Number of threads for multiprocessing")
                        
    # results handling
    parser.add_argument("--traj_mod", 
                        "-t", 
                        default=0, 
                        type=int, 
                        help="Mod for recording fitness curve")

    parser.add_argument("--algorithm_log",
                        "-al",
                        action='store_true',
                        default=False,
                        help="Whether there is log generated by algorithm that should be stored.")

    parser.add_argument("--log_directory",
                        "-ld",
                        default='',
                        help="Specific logging directory")

    parser.add_argument("--overwrite",
                        action='store_true',
                        default=False,
                        help="Whether to overwrite record if it already exist.")

    # fmt: on
    return parser.parse_args()


def logging(opt):
    @wraps(opt)
    def opt_with_logging(func_id):
        start = time.time()
        res, err, log = opt(func_id)
        end = time.time()

        print(
            "Prob.{:<4}, res:{:.4e},\t, err:{:.4e},\t time:{:.3f}".format(
                func_id + 1, res, err, end - start
            )
        )

        return res, end - start, log

    return opt_with_logging


if __name__ == "__main__":

    args = parsing()

    # get benchmark
    benchmark = Benchmark(args.benchmark, args.dim)

    # get opimizer
    alg_mod = importlib.import_module("mpopt.algorithms." + args.alg)
    model = getattr(alg_mod, args.alg)
    optimizer = model()

    # get algorithm params and update to provided params
    params = optimizer.default_params(benchmark=benchmark)
    input_params = eval(args.apd)

    for param in input_params:
        if param in params:   
            params[param] = input_params[param]
        else:
            raise Exception("Unsupported parameter '{}' is provided!".format(param))


    # record name
    record_name = args.alg
    if args.name is not "":
        record_name = record_name + "_" + args.name
    record_name = record_name + ".json"

    # record path
    project_path = os.path.split(os.path.realpath(__file__))[0]
    if args.log_directory is not '':
        relative_path = "../logs/{}".format(args.log_directory)
    else:
        relative_path = "../logs/{}_{}D".format(benchmark.name, benchmark.dim)
    record_path = os.path.join(project_path, relative_path)
    
    # check whether the record already exist
    if not args.overwrite and os.path.exists(os.path.join(record_path, record_name)):
        print("Record [{}] in [{}] already exist, optimization abort.".format(record_name, record_path, ))
        exit()

    # opt function
    @logging
    def opt(idx):
        evaluator = benchmark.generate(idx, traj_mod=args.traj_mod)
        optimizer = model()
        optimizer.set_params(params)
        res = optimizer.optimize(evaluator)
        err = res - evaluator.obj.optimal_val

        # handle log
        log = {}
        if args.algorithm_log:
            if hasattr(optimizer, "log"):
                log = optimizer.log
            else:
                print("Warning: Opitmizer has no attribute 'log' for storing. Empty log is taken.")

        if args.traj_mod > 0:
            log["traj"] = copy.deepcopy(evaluator.traj)
        
        return res, err, log

    # store results
    res = np.empty((benchmark.num_func, args.rep))
    cst = np.empty((benchmark.num_func, args.rep))
    log = [[None for rep in range(args.rep)] for fid in range(benchmark.num_func)]

    for i in range(benchmark.num_func):

        if args.multiprocess > 1:
            # multiprocessing
            p = Pool(args.multiprocess)
            results = p.map(opt, [i] * args.rep)
            p.close()
            p.join()

            for j in range(args.rep):
                res[i, j], cst[i, j], log[i][j] = results[j][0], results[j][1], results[j][2]

        else:
            # sequential
            for j in range(args.rep):
                result = opt(i)
                res[i, j], cst[i, j], log[i][j] = result[0], result[1], result[2]

    # save
    record = {}
    record["experiment datetime"] = time.asctime()
    record["args"] = vars(args)
    record["params"] = params
    record["optimals"] = res.tolist()
    record["times"] = cst.tolist()

    # store result
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    with open(os.path.abspath(os.path.join(record_path, record_name)), "w") as f:
        json.dump(record, f)

    if args.traj_mod > 0:
        traj = [[log[i][j]["traj"] for j in range(args.rep)] for i in range(benchmark.num_func)]
        traj = [np.array(traj[_]).mean(axis=0).tolist() for _ in range(benchmark.num_func)]
        with open(os.path.abspath(os.path.join(record_path, "Traj2_"+record_name)), "w") as f:
            json.dump(traj, f)
