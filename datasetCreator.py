#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:53:14 2022

@author: s215863
"""
from DataGenerator import DataGenerator
import numpy as np
import os

def make_directory(name = None, sigma = None, numvars = None):
    if(name == None or sigma == None or numvars == None):
        raise ValueError("Inputs must be not none")
    base_dir = "data/{}/{}/{}".format(name, numvars,sigma)
    if(not os.path.isdir(base_dir)):
        os.makedirs(base_dir)
    return base_dir


if(__name__ == "__main__"):
    base_dir = "data"
    #try: 
    for sigma in [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]:
        funcs = [DataGenerator.lorenz96, DataGenerator.lotka_volterra, DataGenerator.chua]
        arguments = [(10,), (1.2, 0.2, 0.05, 1.1), (10.82, 14.286)]
        for fun, args in zip(funcs, arguments):
            generator = DataGenerator(fun, sigma = sigma)
            for p in [8, 12, 20, 48]:
                if(fun == DataGenerator.chua):
                    p = 3
                for i in range(5):
                    series_rk4, graph = generator.simulate(p=p, burn_in=500, T=10000, args=args)
                    # plt.matshow(graph)
                    # plt.plot(series_lsoda)
                    # plt.show()
                    base_dir = make_directory(name = fun.__name__, sigma = sigma, numvars = p)
                    np.save(os.path.join(base_dir, "rk4_base_{}.npy".format(i)), series_rk4)
                    np.save(os.path.join(base_dir, "causal_graph.npy"), graph)
    
    file = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/returns/random-rels_20_1_3_returns30007000.csv"
    gt = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/relationships/random-rels_20_1_3.csv"
    series, causality_graph = DataGenerator.finance(file, gt)
    np.save(os.path.join("data/finance", "finance.npy"), series)
    np.save(os.path.join("data/finance", "causal_graph.npy"), causality_graph)
    
