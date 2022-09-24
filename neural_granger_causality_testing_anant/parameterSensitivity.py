#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:30:07 2022

@author: s215863
"""

from GVAR.datasets.lotkaVolterra.multiple_lotka_volterra import MultiLotkaVolterra
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from Metrics import Metrics
from DataGenerator import DataGenerator
from GVARTesterTRGC import GVARTesterTRGC
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
from bayesianNetwork import BNTester
import os
import shutil
import pickle
from datasetCreator import make_directory
import re
import torch
import gc
from datasetCreator import make_directory
from stochasticityTester import parseNumerics

lorenz_generator = DataGenerator(DataGenerator.lorenz96)
series, causality_graph = lorenz_generator.simulate(p=12, T=5000, args=(10,))

def GVARSensitivity():
    n = int(0.8*len(series))
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.]:
        for lmbd in [0.05, 0.1, 0.15, 0.2]:
            for gmma in [0.05, 0.1, 0.15, 0.2]:
                lstmTester = GVARTesterTRGC(series[:n], cuda = True)
                lstmTester.alpha = alpha
                lstmTester.lmbd = lmbd
                lstmTester.gmma = gmma
                lstmTester.trainInherit()
                torch.cuda.empty_cache()
                gc.collect()
                metrics = Metrics(lstmTester, causality_graph, series, store_directory="gvarSensitivity/{}/{}/{}".format(alpha, lmbd, gmma))
                metrics.vis_pred(start = n)
                metrics.vis_causal_graphs()
                metrics.prec_rec_acc_f1()

def TCDFSensitivity():
    n = int(0.8*len(series))
    for sig in np.arange(0.1, 1, 0.1):
        lstmTester = TCDFTester(series[:n], cuda = True, significance = sig)
        lstmTester.trainInherit()
        torch.cuda.empty_cache()
        gc.collect()
        metrics = Metrics(lstmTester, causality_graph, series, store_directory="TCDFSensitivity/{}".format(sig))
        metrics.vis_pred(start = n)
        metrics.vis_causal_graphs()
        metrics.prec_rec_acc_f1()

def cLSTMSensitivity():
    n = int(0.8*len(series))
    for lam in [0.025, 0.05, 0.1, 0.2, 0.4]:
        for lam_ridge in [0.025, 0.05, 0.1, 0.2, 0.4]:
            lstmTester = cLSTMTester(series[:n], cuda = True, lam_ridge=lam_ridge, lam=lam)
            lstmTester.lr = 0.0005
            lstmTester.trainInherit()
            torch.cuda.empty_cache()
            gc.collect()
            metrics = Metrics(lstmTester, causality_graph, series, store_directory="cLSTMSensitivity/{}/{}".format(lam, lam_ridge))
            metrics.vis_pred(start = n)
            metrics.vis_causal_graphs()
            metrics.prec_rec_acc_f1()

def BNSensitivity():
    n = int(0.8*len(series))
    for dropout in np.arange(0.1,1,0.1):
        for sig in [0, 0.05, 0.1, 0.2, 0.4, 0.8]:
            lstmTester = BNTester(series[:n], cuda = True, large = True, dropout=dropout, sig=sig)
            lstmTester.lr = 0.005
            lstmTester.trainInherit()
            torch.cuda.empty_cache()
            gc.collect()
            metrics = Metrics(lstmTester, causality_graph, series, store_directory="BNSensitivity/{}/{}".format(dropout, sig))
            metrics.vis_pred(start = n)
            metrics.vis_causal_graphs()
            metrics.prec_rec_acc_f1()


def plot(data, l, xlabel, ylabel, title):
    plt.plot(np.array(data).transpose())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(ticks=range(len(l)), labels = [float('%.1g'%x) for x in l])
    plt.legend(["Precision", "Accuracy", "Recall", "F1 score"])
    plt.gca().set_ylim(top=1.001)
    plt.show()

def GVAR_graph(alpha = 0.5, lmbd = 0.1, gmma = 0.1, graphName="Errors"):
    def plot_param(var, l):
        final_vals = []
        for loss in losses:
            vals = []
            for param in l:
                alph = param if var == "alpha" else alpha
                lmb = param if var == "lmbd" else lmbd
                gmm = param if var == "gmma" else gmma
                directory = "gvarSensitivity/{}/{}/{}".format(alph, lmb, gmm)
                vals.append(gens[directory][loss])
            final_vals.append(vals)
        paramString1 = "alpha = {}".format(alpha)
        paramString2 = "lmbd = {}".format(lmbd)
        paramString3 = "gmma = {}".format(gmma)
        if(var == "alpha"):
            choices = (paramString2, paramString3)
        elif(var == "lmbd"):
            choices = (paramString1, paramString3)
        else:
            choices = (paramString1, paramString2)
        paramString = "{} and {}".format(choices[0], choices[1])
        title = " GVAR {} with respect to {} with {}".format(graphName, var, paramString)
        plot(final_vals, l, var.capitalize(), graphName, title)
    
    alphas = [0.3, 0.4, 0.5, 0.6, 0.0]
    lmbds  = [0.05, 0.1, 0.15, 0.2]
    gmmas  = [0.05, 0.1, 0.15, 0.2]
    losses = ["Precision", "Accuracy", "Recall", "F1 score"]
    directories = ["gvarSensitivity/{}/{}/{}".format(alpha, lmbd, gmma)
                   for alpha in alphas for gmma in gmmas for lmbd in lmbds]
    gens = {directory:parseNumerics(directory) for directory in directories}
    plot_param("alpha", alphas)
    plot_param("lmbd", lmbds)
    plot_param("gmma", gmmas)

def BN_graph(dropout = 0.5, sig = 0, graphName="Errors"):
    def plot_param(var, l):
        final_vals = []
        for loss in losses:
            vals = []
            for param in l:
                p1 = param if var == "dropout" else dropout
                p2 = param if var == "sig" else sig
                directory = "BNSensitivity/{}/{}".format(p1, p2)
                vals.append(gens[directory][loss])
            final_vals.append(vals)
        paramString1 = "dropout = {}".format(dropout)
        paramString2 = "sig = {}".format(sig)
        paramString = paramString2 if var == "dropout" else paramString1
        title = "BN {} with respect to {} with {}".format(graphName, var, paramString)
        plot(final_vals, l, var.capitalize(), graphName, title)
    
    dropouts = np.arange(0.1,1,0.1)
    sigs = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
    losses = ["Precision", "Accuracy", "Recall", "F1 score"]
    directories = ["BNSensitivity/{}/{}".format(dropout, sig)
                   for sig in sigs for dropout in dropouts]
    gens = {directory:parseNumerics(directory) for directory in directories}
    
    plot_param("dropout", dropouts)
    plot_param("sig", sigs)

def cLSTM_graph(lm = 0.1, lm_ridge = 0.1, graphName="Errors"):
    def plot_param(var, l):
        final_vals = []
        for loss in losses:
            vals = []
            for param in l:
                p1 = param if var == "lm" else lm
                p2 = param if var == "lm_ridge" else lm_ridge
                directory = "cLSTMSensitivity/{}/{}".format(p1, p2)
                vals.append(gens[directory][loss])
            final_vals.append(vals)
        paramString1 = "lm = {}".format(lm)
        paramString2 = "lm_ridge = {}".format(lm_ridge)
        paramString = paramString2 if var == "lm" else paramString1
        title = "cLSTM {} with respect to {} with {}".format(graphName, var, paramString)
        plot(final_vals, l, var.capitalize(), graphName, title)
    
    params = [0.025, 0.05, 0.1, 0.2, 0.4]
    losses = ["Precision", "Accuracy", "Recall", "F1 score"]
    directories = ["cLSTMSensitivity/{}/{}".format(lm, lm_ridge)
                   for lm in params for lm_ridge in params]
    gens = {directory:parseNumerics(directory) for directory in directories}
    
    plot_param("lm", params)
    plot_param("lm_ridge", params)

def TCDF_graph(graphName="Erros"):
    
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    losses = ["Precision", "Accuracy", "Recall", "F1 score"]
    final_vals = []
    for loss in losses:
        vals = []
        for param in params:
            directory = "TCDFSensitivity/{}".format(param)
            numerics = parseNumerics(directory)
            vals.append(numerics[loss])
        final_vals.append(vals)
    title = "TCDF {} with respect to significance".format(graphName)
    plot(final_vals, params, "Significance", graphName, title)

if(__name__ == "__main__"):
    GVAR_graph()
    BN_graph()
    cLSTM_graph()
    TCDF_graph()
            