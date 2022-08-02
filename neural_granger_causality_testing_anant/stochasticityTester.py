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
from bayesianNetwork import BNTester
import os
import shutil
import pickle
from datasetCreator import make_directory
import re


def parseNumerics(directory):
    print(directory)
    f = open(directory+"/numerics.txt", 'r').read().replace("F1", "F")
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    numbers = rx.findall(f)
    numbers = np.array([float(number) for number in numbers])
    print(numbers)
    ret_dict = {"Precision":numbers[0], "Recall":numbers[1], "Accuracy":numbers[2], "F1 score":numbers[3], "ADF Pval":numbers[-3], "KPSS Pval": numbers[-2], "LBox Pval":numbers[-1]}
    losses = ["Error std", "Percentage std to orig", "MSE Loss from orig", "PErcentage MSE from orig"]
    newVals=  np.mean(np.reshape(numbers[4:-3],(4,-1) ), axis = 1)
    for loss, val in zip(losses, newVals):
        ret_dict[loss] = val
    return ret_dict

def make_sigma_graph(numvar = None, func = None, graphName = None):
    def plot(numvar, func, graphName):
        data = []
        for model in models:
            vals = []
            for sigma in sigmas:
                directory = make_directory(name = func, sigma=sigma, numvars=numvar) + "/" +model +"/unseen"
                vals.append(gens[directory][graphName])
            data.append(vals)
        plt.plot(np.array(data).transpose())
        plt.title(graphName+" overtime for {} data with {} variables".format(func, numvar))
        plt.legend(models)
        plt.ylabel(graphName)
        plt.xlabel("White noise variance")
        plt.xticks(ticks=range(len(sigmas)), labels = sigmas)
        plt.show()
    numvars = [numvar] if numvar != None else [8, 12, 20]
    funcs = [func] if func != None else ["lorenz96", "lotka_volterra"]
    models = ["GVARTesterTRGC", "TCDFTester", "cLSTMTester", "BNTester"]
    sigmas = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
    losses = [graphName] if graphName != None else ["Precision", "Accuracy", "Recall", "F1 score", "Error std", "Percentage std to orig",
                      "MSE Loss from orig", "PErcentage MSE from orig", "ADF Pval", "KPSS Pval", "LBox Pval"]
    directories = [make_directory(name = func, sigma=sigma, numvars=numvar) + "/" +model + "/unseen"
                   for sigma in sigmas for model in models for func in funcs for numvar in numvars]
    gens = {directory:parseNumerics(directory) for directory in directories}
    
    for numvar in numvars:
        for func in funcs:
            for graphName in losses:
                plot(numvar, func, graphName)

def make_numvar_graph(sigma = None, func = None, graphName = None):
    def plot(numvar, func, graphName):
        data = []
        numvars = [8,12,20]
        for model in models:
            vals = []
            for numvar in numvars:
                directory = make_directory(name = func, sigma=sigma, numvars=numvar) + "/" +model
                vals.append(gens[directory][graphName])
            data.append(vals)
        plt.plot(np.array(data).transpose())
        plt.title(graphName+" overtime for {} data with {} stochasticity".format(func, numvar))
        plt.legend(models)
        plt.xticks(ticks=range(len(numvars)), labels = numvars)
        plt.ylabel(graphName)
        plt.xlabel("Num Vars")
        plt.show()
    
    sigmas = [sigma] if sigma != None else [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
    funcs = [func] if func != None else ["lorenz96", "lotka_volterra"]
    models = ["GVARTesterTRGC", "TCDFTester", "cLSTMTester", "BNTester"]
    numvars = [8,12,20]
    losses = [graphName] if graphName != None else ["Precision", "Recall", "Accuracy", "F1 score", "Error std", "Percentage std to orig",
                      "MSE Loss from orig", "PErcentage MSE from orig", "ADF Pval", "KPSS Pval", "LBox Pval"]
    directories = [make_directory(name = func, sigma=sigma, numvars=numvar) + "/" +model +"/unseen"
                   for sigma in sigmas for model in models for func in funcs for numvar in numvars]
    gens = {directory:parseNumerics(directory) for directory in directories}
    
    for sigma in sigmas:
        for func in funcs:
            for graphName in losses:
                plot(sigma, func, graphName)
                
            

if(__name__ == "__main__"):
    make_numvar_graph(func = "lorenz96", sigma = 0)
                    
                    
                        
