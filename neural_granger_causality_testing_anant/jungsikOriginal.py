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

layer = 1

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
    numWindows = len(arr[0][0][0]) - 2 # array includes 82 windows, but we only use middle 80 for padding
    timeLen = len(arr[0][0])
    assert numWindows == 80 and timeLen == 300 #hardcoded for Jungsiks data
    predictors = []
    responses = []
    for window in range(1, 81):
        final_normal = getNeighbors(arr, window, layer)
        pred, resp, _ = construct_training_dataset(final_normal, order=self.context)
        predictors.append(pred)
        responses.append(resp)
    x = np.concatenate(predictors)
    y = np.concatenate(responses)
    return x, y

def batchGenerator(self, X, Y):
    l = list(range(len(X)))
    random.shuffle(l)
    batches = np.array(l).reshape(( -1, 2320))
    for batch in batches:
        x= X[batch, :, :]
        y= Y[batch, :]
        x = Variable(torch.tensor(x, dtype=torch.float)).float().to(self.device).cuda()
        y = Variable(torch.tensor(y, dtype=torch.float)).float().to(self.device).cuda()
        yield x, y

def makeGCgraph2(self, x, singular_iteration, make_final_graph,plotSwitch = False, argument = 0.5):
    def plot(truth, title):
        if(not plotSwitch):
            return
        mean_over_time = np.mean(truth, axis=0)
        std_over_time = np.std(truth, axis=0)
        lower = mean_over_time - 3*std_over_time
        higher = mean_over_time + 3*std_over_time
        fig, axarr = plt.subplots(3,4, figsize=(10, 5))
        fig.suptitle(title)
        for i in range(self.numVars):
            x,y = i//4, i%4
            axarr[x,y].plot(higher[:300,i], "g")
            axarr[x,y].plot(lower[:300,i], "r")
            axarr[x,y].set_title('Var {}'.format(i))
        plt.show()
        
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
        plot(truth, "Truthfull predictions")
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
            plot(prediction, "Permuted variable: {}".format(idx))
            prediction = singular_iteration(prediction)
            permuted.append(prediction)
      
        self.graph_est = make_final_graph(truth, np.array(permuted), q = argument)
        self.graph_est_done = True
        return self.graph_est

if(__name__ == "__main__"):
    
    data_dir = "/home2/s215863/Desktop/Granger Causality/JungsikData/Cell 2 LFD Data"
    arr = spatial_preprocess(data_dir)
    numVars = 13 if layer == 2 or layer == 3 else 11
    lstmtester=BNTester(arr, numVars = numVars, cuda=True, large=True)
    lstmtester.preprocess_data = types.MethodType(constructDataset, lstmtester)
    lstmtester.context = 30
    lstmtester.NUM_EPOCHS = 1000
    lstmtester.batch_generator = types.MethodType(batchGenerator, lstmtester)
    lstmtester.trainInherit()
    f = lstmtester.make_GC_graph2
    lstmtester.make_GC_graph2 = types.MethodType(makeGCgraph2, lstmtester)
    
    actual = getNeighbors(arr, 40, layer)
    graph = np.diag((numVars, numVars))
    pred = lstmtester._predict(actual, lstmtester.context)
    error = actual-pred
    fig, axarr = plt.subplots(3, 1, figsize=(10, 5))
    axarr[0].plot(actual)
    axarr[0].set_title('Actual timeseries')
    axarr[1].plot(pred)
    axarr[1].set_title('Predicted timeseries: BNTester')
    axarr[2].plot(error)
    axarr[2].set_title('Error timeseries')
    plt.show()