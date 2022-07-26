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
import pickle
from datasetCreator import make_directory

def make_experiment():
    i = 1
    while(os.path.isdir("experiments/exp{}".format(i))):
        i+=1
    os.makedirs("experiments/exp{}".format(i))
    return "experiments/exp{}".format(i)

if(__name__ == "__main__"):
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
                models = [GVARTesterStable(series[:n], cuda = True), TCDFTester(series[:n], cuda = True), cLSTMTester(series[:n], cuda=True)]
                for model in models:
                    directory = base_dir+"/"+type(model).__name__
                    if(not os.path.isdir(directory)):
                        continue
                    model.load(directory)
                    if(model.NUM_EPOCHS == 1500):
                        print(directory)
