## -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:30:46 2022
ddd
@author: anant
"""

#import sys
#import os
#packages = ["C:\\Users\\anant\\anaconda3\\envs\\granger\\Neural-GC",
#            "C:\\Users\\anant\\anaconda3\\envs\\granger\\GVAR"]
#for package in packages:
#    if(package not in sys.path):
#        sys.path.append(package)

from Metrics import Metrics
from DataGenerator import DataGenerator
from GVARTester import GVARTester, GVARTesterStable
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
import matplotlib.pyplot as plt

if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.chua)
    #series, causality_graph = lorenz_generator.create_series([[0.64994384, 0.01750787, 0.72402577, 0.14358566, 0.502893]], F = 8)
    series, causality_graph = lorenz_generator.simulate(p=3, T=2000, args=(10.82, 13.286))
    file = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/returns/random-rels_20_1_3_returns30007000.csv"
    gt = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/relationships/random-rels_20_1_3.csv"
    #series, causality_graph = DataGenerator.finance(file, gt)
    n = int(0.8*len(series))
    lstmTester = GVARTesterStable(series[:n], cuda = False)
    lstmTester2 = TCDFTester(series[:n], cuda = False)
    #lstmTester.train()
    lstmTester2.trainInherit()
    lstmTester.trainInherit()
    metrics = Metrics(lstmTester, causality_graph, series)
    metrics.vis_pred(start = n)
    metrics.vis_causal_graphs()
    metrics.prec_rec_acc_f1()
    
    metrics2 = Metrics(lstmTester2, causality_graph, series)
    metrics2.vis_pred(start=n)
    metrics2.vis_causal_graphs()
    metrics2.prec_rec_acc_f1()

    
    print("Done")