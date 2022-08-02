#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:53:14 2022

@author: s215863
"""
from Metrics import Metrics
from GVARTester import GVARTesterStable
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester
from bayesianNetwork import BNTester
import numpy as np
import os
from datasetCreator import make_directory
from GVARTesterTRGC import GVARTesterTRGC
import torch
import gc

def make_experiment():
    i = 1
    while(os.path.isdir("experiments/exp{}".format(i))):
        i+=1
    os.makedirs("experiments/exp{}".format(i))
    return "experiments/exp{}".format(i)

def verifyNewModels(numEpochs):
    sigmas = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
    numvars = [8, 12, 20, 48]
    funcs = ["lorenz96", "lotka_volterra", "chua"]
    for sigma in sigmas:
        for numvar in numvars:
            for func in funcs:
                if(func == "chua"):
                    numvar = 3
                base_dir = make_directory(name = func, sigma=sigma, numvars=numvar)
                series = np.zeros((6,6))
                n = 6
                models = [GVARTesterTRGC(series[:n], cuda=True)]
                #models = [GVARTesterTRGC(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
                for model in models:
                    directory = base_dir+"/"+type(model).__name__
                    if(not os.path.isdir(directory)):
                        continue
                    model.load(directory)
                    if(model.NUM_EPOCHS == numEpochs):
                        print("# layers:", model.senn.num_hidden_layer_size)
                        print("# per layer:", model.senn.hidden_layer_size)
                        
def verifyStateEquality():
    sigmas = [0, 0.05, 2, 5, 10]
    numvars = [8, 12, 20, 48]
    funcs = ["lorenz96"]
    for sigma in sigmas:
        for numvar in numvars:
            for func in funcs:            
                base_dir = make_directory(name = func, sigma=sigma, numvars=numvar)
                s = "rk4" if sigma==0 else "ito"
                series = np.load(os.path.join(base_dir, s+"_base_1.npy"))
                graph = np.load(os.path.join(base_dir, "causal_graph.npy"))
                n = 5000
                models = [GVARTesterTRGC(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
                while(len(models) > 0):
                    model = models[0]
                    directory = os.path.join(base_dir, type(model).__name__)
                    model.load(directory)
                    metrics = Metrics(model, graph, series, store_directory=directory)
                    prev_pred = np.load(os.path.join(directory, "pred.npy"))
                    print(np.allclose(prev_pred, metrics.pred_timeseries))
                    del models[0]
                    torch.cuda.empty_cache()
                    gc.collect()

def test2():
    sigmas = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
    numvars = [12]
    funcs = ["lorenz96"]
    for sigma in sigmas:
        c = False
        for numvar in numvars:
            for func in funcs:        
                if(func == "chua" and not c):
                    numvar = 3
                    c = True
                elif(func == "chua"):
                    continue
                base_dir = make_directory(name = func, sigma=sigma, numvars=numvar)
                s = "rk4" if sigma==0 or func == "lotka_volterra" else "ito"
                series = np.load(os.path.join(base_dir, s+"_base_2.npy"))
                graph = np.load(os.path.join(base_dir, "causal_graph.npy"))
                n = 5000
                models = [GVARTesterTRGC(series[:n], cuda=True)]
                #models = [GVARTesterTRGC(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
                while(len(models) > 0):
                    model = models[0]
                    directory = os.path.join(base_dir, type(model).__name__)+"2"
                    try:
                        model.load(directory)
                    except:
                        continue
                    metrics = Metrics(model, graph, series, store_directory=directory+"/unseen")
                    metrics.test()
                    del models[0]
                    torch.cuda.empty_cache()
                    gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
                


if(__name__ == "__main__"):
    test2()