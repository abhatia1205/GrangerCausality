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
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
import os
import shutil

def make_experiment():
    i = 1
    while(os.path.isdir("experiments/exp{}".format(i))):
        i+=1
    os.makedirs("experiments/exp{}".format(i))
    return "experiments/exp{}".format(i)

if(__name__ == "__main__"):
    base_dir = make_experiment()
    #try:
    for sigma in [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]:
        lorenz_generator = DataGenerator(DataGenerator.lorenz96, sigma = sigma)
        series_lsoda, series_rk4, graph = lorenz_generator.simulate(p=12, burn_in=200, T=2000, args=(10,))
        plt.matshow(graph)
        plt.plot(series_lsoda)
        plt.show()
        
        np.save(os.path.join(base_dir, "lsoda_base.npy"), series_lsoda)
        np.save(os.path.join(base_dir, "rk4_base.npy"), series_rk4)
        
        
        series = series_lsoda
        n = int(0.8*len(series))
        models = [GVARTesterStable(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
        for model in models:
            directory = os.path.join(base_dir, str(sigma), type(model).__name__)
            os.makedirs(directory)
            model.trainInherit()
            metrics = Metrics(model, graph, series, store_directory=directory)
            metrics.vis_pred(start = n)
            metrics.vis_causal_graphs()
            metrics.prec_rec_acc_f1()
    # except:
    #     shutil.rmtree(base_dir, ignore_errors = True)
