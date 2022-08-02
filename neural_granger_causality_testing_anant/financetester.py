#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:30:24 2022

@author: s215863
"""

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
from bayesianNetwork import BNTester
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
import os
import torch
import gc

def get_finance_data():
    base_dir = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/returns/"
    gt_base = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/relationships"
    for fname in os.listdir(base_dir):
        if( "random-rels" in fname and  fname.index("random-rels") == 0):
            file = os.path.join(base_dir, fname)
            corres_index = fname.index("_returns")
            gt_suff = fname[:corres_index]+".csv"
            gt = os.path.join(gt_base, gt_suff)
            series, graph =  DataGenerator.finance(file, gt)
            yield series, graph, fname[:corres_index]

if(__name__ == "__main__"):
    

    gen = get_finance_data()
    if(not os.path.isdir("data2/finance")):
        os.makedirs("data2/finance")
    finance_dir = "data2/finance"
    for series, graph, dir_name in gen:
        base_dir = os.path.join(finance_dir, dir_name)
        n = int(0.8*len(series))
        models = [BNTester(series[:n], cuda = True),GVARTesterTRGC(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
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