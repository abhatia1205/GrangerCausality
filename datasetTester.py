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

if(__name__ == "__main__"):
    for sigma in [0.05, 0.1, 0.25, 0.5, 1, 2, 5]:
        lorenz_generator = DataGenerator(DataGenerator.lorenz96, sigma = 3)
        series, series2, graph = lorenz_generator.simulate(p=12, burn_in=200, T=200, args=(10,))
        plt.matshow(graph)
        plt.plot(series2)
        plt.show()
        
        series = series2
        n = int(0.8*len(series))
        models = [GVARTesterStable(series[:n], cuda = False), TCDFTester(series[:n], cuda = False)]
        for model in models:
            model.trainInherit()
            metrics = Metrics(model, graph, series)
            metrics.vis_pred(start = n)
            metrics.vis_causal_graphs()
            metrics.prec_rec_acc_f1()
