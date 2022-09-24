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
from GVARTesterTRGC import GVARTesterTRGC
from cLSTMTester import cLSTMTester
from TCDFTester import TCDFTester, TCDFTesterOrig
import numpy as np
import matplotlib.pyplot as plt
from bayesianNetwork import BNTester
import torch
import gc
from torch.autograd import Variable
import random
import pandas as pd
import os
from DataGenerator import DataGenerator
from GVAR.utils import construct_training_dataset
import types

LAYER = 1

def spatial_preprocess(directory):
    d = {'Edge Velocity':[], "Arp":[], "Actin":[]}
    for filepath in os.listdir(directory):
        if('txt' in filepath):
            df = pd.read_csv(os.path.join(directory, filepath)).to_numpy().transpose()[1:, :82] #cut to 82 windows since 80 is a nuce number +2 for padding LSTM
            if('actin' in filepath):
                d['Actin'].append(df)
            elif('arp' in filepath):
                d['Arp'].append(df)
            else:
                d['Edge Velocity'].append(df)
    return [np.array(d["Edge Velocity"]), 
            np.array(d["Actin"]), np.array(d["Arp"])]

def getNeighbors(arr, window, layer):
    edgeVelocities = np.array([arr[0][0][:, window+i] for i in [-1,0,1]])
    arps = []
    actins = []
    for layer_delt, window_delt in [(-1,0), (0, -1), (0,0), (0,1), (1,0)]:
        layer_temp = layer+layer_delt
        window_temp = window+window_delt
        if(layer_temp >0 and layer_temp < 5):
            actins.append(arr[1][layer_temp][:, window_temp])
            arps.append(arr[2][layer_temp][:, window_temp])
    arps = np.array(arps)
    actins = np.array(actins)
    final = np.concatenate((arps, actins, edgeVelocities)).transpose()
    final_normal = DataGenerator.normalize(final)
    final_normal[~np.isfinite(final_normal)] = 0
    return final_normal

def constructDataset(self, arr = None):
    if(arr is None):
        arr = self.X       
    numWindows = len(arr[0][0][0]) - 2 # array includes 42 windows, but we only use middle 80 for padding
    timeLen = len(arr[0][0])
    print(numWindows, timeLen)
    assert numWindows == 80 and timeLen == 300 #hardcoded for Jungsiks data
    predictors = []
    responses = []
    for window in range(1, 81):
        data = getNeighbors(arr, window, self.layer)
        pred, resp, _ = construct_training_dataset(data, order=self.context)
        predictors.append(pred)
        responses.append(resp)
    x = np.concatenate(predictors)
    y = np.concatenate(responses)
    return x, y

def constructDatasetGVAR(self, arr = None):
    if(arr is None):
        arr = self.X       
    numWindows = len(arr[0][0][0]) - 2 # array includes 42 windows, but we only use middle 80 for padding
    timeLen = len(arr[0][0])
    print(numWindows, timeLen)
    assert numWindows == 80 and timeLen == 300 #hardcoded for Jungsiks data
    predictors = []
    responses = []
    predictors_tr = []
    responses_tr = []
    for window in range(1, 81):
        data = getNeighbors(arr, window, self.layer)
        pred, resp, _ = construct_training_dataset(data, order=self.order)
        predictors.append(pred)
        responses.append(resp)
        pred, resp, _ = construct_training_dataset(data=np.flip(data, axis=1), order=self.order)
        predictors_tr.append(pred)
        responses_tr.append(resp)
    x = np.concatenate(predictors)
    y = np.concatenate(responses)
    yield x, y
    x = np.concatenate(predictors_tr)
    y = np.concatenate(responses_tr)
    yield x, y

def batchGenerator(self, X, Y):
    l = list(range(len(X)))
    random.shuffle(l)
    batches = np.array(l).reshape(( 10, int(len(l)/10)))
    for batch in batches:
        x= X[batch, :, :]
        y= Y[batch, :]
        x = Variable(torch.tensor(x, dtype=torch.float)).float().to(self.device).cuda()
        y = Variable(torch.tensor(y, dtype=torch.float)).float().to(self.device).cuda()
        yield x, y

def batchGeneratorGVAR(self, X, Y):
    l = list(range(len(X)))
    random.shuffle(l)
    batches = np.array(l).reshape(( 10, int(len(l)/10)))
    for batch in batches:
        x= X[batch, :, :]
        x1=x[1:, :, :]
        y= Y[batch, :]
        yield (x, x1), y

def plot(truth, layer, directory):
    n = 5 if layer == 1 or layer == 4 else 6
    actins = ["Act {}".format(i) for i in range(1,n)]
    mutants = ["Arp {}".format(i) for i in range(1,n)]
    EV = ["EV {}".format(i) for i in range(3)]
    ticks = actins+mutants+EV
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(truth, interpolation='nearest')
    fig.colorbar(cax)
    xaxis = np.arange(len(ticks))
    ax.set_xticks(xaxis)
    ax.set_yticks(xaxis)
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)
    ax.set_ylabel("Causal series")
    ax.set_xlabel("Affected Series")

    plt.show()
    plt.savefig(directory+"/graph.jpg")

def makeGCgraph2(self, x, singular_iteration, make_final_graph,plotSwitch = True, argument = 0.5):     
    self.model.eval()
    with torch.no_grad():
        predictors, responses = self.preprocess_data(arr=x)
        X = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device).cuda()
        truth = []
        for i in range(50):
            truth.append(self.model.forward(X))
            # torch.cuda.empty_cache()
            # gc.collect()
        truth = [x.cpu().numpy() for x in truth]
        truth = singular_iteration(truth)
        permuted = []
        for idx in range(self.numVars):
            newX = X.clone().cpu().numpy()
            random.shuffle(newX[:, :, idx])
            newX = torch.from_numpy(newX).cuda()
            prediction = []
            for i in range(50):
                prediction.append(self.model.forward(newX))
                # torch.cuda.empty_cache()
                # gc.collect()
            prediction = [x.cpu().numpy() for x in prediction]
            prediction = singular_iteration(prediction)
            permuted.append(prediction)
      
        self.graph_est = make_final_graph(truth, np.array(permuted), q = argument)
        plot(self.graph_est)
        return self.graph_est

if(__name__ == "__main__"):
    
    while(LAYER < 5):
        data_dir = "/home2/s215863/Documents/jungsikVASP"
        save_dir = "jungSikModelsOrig/{}".format(LAYER)
        arr = spatial_preprocess(data_dir)
        numVars = 11 if LAYER == 1 or LAYER == 4 else 13
        lstmtester=GVARTesterTRGC(arr, numvars = numVars, cuda=True)
        lstmtester.preprocess_data = types.MethodType(constructDatasetGVAR, lstmtester)
        lstmtester.order = 30
        lstmtester.layer = LAYER
        lstmtester.lr = 0.002
        lstmtester.batch_generator = types.MethodType(batchGeneratorGVAR, lstmtester)
        lstmtester.NUM_EPOCHS = 3000
        lstmtester.trainInherit()
        lstmtester.save(save_dir+"/"+type(lstmtester).__name__)
        #lstmtester.make_GC_graph2 = types.MethodType(makeGCgraph2, lstmtester)
        plot(lstmtester.make_GC_graph(), LAYER, save_dir+"/"+type(lstmtester).__name__)
        plt.plot([x.cpu().detach().numpy() for x in lstmtester.history])
        
        actual = getNeighbors(arr, 40, LAYER)
        graph = np.diag((numVars, numVars))
        pred = lstmtester._predict(actual)
        error = actual-pred
        fig, axarr = plt.subplots(3, 1, figsize=(10, 5))
        axarr[0].plot(actual)
        axarr[0].set_title('Actual timeseries')
        axarr[1].plot(pred)
        axarr[1].set_title('Predicted timeseries:GVARTester')
        axarr[2].plot(error)
        axarr[2].set_title('Error timeseries')
        plt.show()
        
        lstmtester=BNTester(arr, numvars = numVars, cuda=True)
        lstmtester.preprocess_data = types.MethodType(constructDataset, lstmtester)
        lstmtester.context = 30
        lstmtester.layer = LAYER
        lstmtester.lr = 0.002
        lstmtester.batch_generator = types.MethodType(batchGenerator, lstmtester)
        lstmtester.NUM_EPOCHS = 3000
        lstmtester.trainInherit()
        lstmtester.save(save_dir+"/"+type(lstmtester).__name__)
        #lstmtester.make_GC_graph2 = types.MethodType(makeGCgraph2, lstmtester)
        plot(lstmtester.make_GC_graph(), LAYER, save_dir+"/"+type(lstmtester).__name__)
        plt.plot(lstmtester.history)
        
        actual = getNeighbors(arr, 40, LAYER)
        graph = np.diag((numVars, numVars))
        pred = lstmtester._predict(actual)
        error = actual-pred
        fig, axarr = plt.subplots(3, 1, figsize=(10, 5))
        axarr[0].plot(actual)
        axarr[0].set_title('Actual timeseries')
        axarr[1].plot(pred)
        axarr[1].set_title('Predicted timeseries:GVARTester')
        axarr[2].plot(error)
        axarr[2].set_title('Error timeseries')
        plt.show()
        LAYER += 1