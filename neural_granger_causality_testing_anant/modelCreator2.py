#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:53:14 2022

@author: s215863
"""
from GVAR.datasets.lotkaVolterra.multiple_lotka_volterra import MultiLotkaVolterra
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from Metrics import Metrics
from DataGenerator import DataGenerator
from GVARTester import GVARTester, GVARTesterStable
from GVARTesterTRGC import GVARTesterTRGC
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
import os
from datasetCreator import make_directory
import torch
import gc

def make_experiment():
    i = 1
    while(os.path.isdir("experiments/exp{}".format(i))):
        i+=1
    os.makedirs("experiments/exp{}".format(i))
    return "experiments/exp{}".format(i)

if(__name__ == "__main__"):
    
    sigmas = [2, 5, 10]
    numvars = [8, 12, 20, 48]
    funcs = ["lorenz96", "chua"]
    for sigma in sigmas:
        c = False
        for numvar in numvars:
            for func in funcs:
                if(func == "chua" and c):
                    continue
                elif(func == "chua"):
                    numvar = 3
                    c = True
                base_dir = make_directory(name = func, sigma=sigma, numvars=numvar)
                series = np.load(os.path.join(base_dir, "ito_base_1.npy"))
                graph = np.load(os.path.join(base_dir, "causal_graph.npy"))
                n = 5000
                models = [GVARTesterTRGC(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
                while(len(models) > 0):
                    model = models[0]
                    directory = os.path.join(base_dir, type(model).__name__)
                    if(not os.path.isdir(directory)):
                        os.makedirs(directory)
                    model.trainInherit()
                    metrics = Metrics(model, graph, series, store_directory=directory)
                    metrics.test()
                    model.save(directory)
                    del models[0]
                    torch.cuda.empty_cache()
                    gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()