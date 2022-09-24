 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 15:53:30 2022

@author: s215863
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:59:52 2022

@author: anant
"""

from ModelInterface import ModelInterface
import scipy
from NeuralGC.models.clstm import cLSTM, train_model_ista, arrange_input, regularize, ridge_regularize, prox_update
import torch
import numpy as np
import torch.nn as nn
import os
import pandas as pd
from GVAR.utils import construct_training_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MSELoss
import gc
import random
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from Metrics import Metrics
from GVARTesterTRGC import GVARTesterTRGC
from scipy.stats import ttest_ind_from_stats

class GCLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):

        super(GCLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = 128 # number of encoder cells (from paper)
        self.hidden_size_2 = 32 # number of decoder cells (from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda")
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = output[:, -1, :] # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred
        
    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(self.device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1)).to(self.device)
        return hidden_state, cell_state
    
    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(self.device)
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2)).to(self.device)
        return hidden_state, cell_state
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()


class ConcatGC(nn.Module):
    def __init__(self, n_features, output_length, batch_size, cuda=True):
        super().__init__()
        self.networks = nn.ModuleList([GCLSTM(n_features, output_length, batch_size) for i in range(n_features)])
        self.networks = self.networks.cuda() if cuda else self.networks
    
    def forward(self, X):
        self.networks.train()
        pred = [network.forward(X)
                for network in self.networks]
        pred = torch.cat(pred, dim=1)
        return pred
    
    def calculate_losses(self, idx, X, Y, loss):
        pred = self.networks[idx](X)
        return loss(pred, Y[:, :, idx])


class GCTester(ModelInterface):
    
    def __init__(self, X, numVars = None, large=False, cuda = True, dropout=0.6, sig=0):
        
        super(GCTester, self).__init__(cuda)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.X = X
        self.numVars = self.X.shape[1] if numVars is None else numVars
        self.model = ConcatGC(self.numVars, 1, 1)
        self.model = self.model.cuda() if cuda else self.model
        self.parameters = self.model.parameters()
        self.base_loss = MSELoss()
        self.graph_est_done = False
        
        self.context = 10
        self.base_loss = nn.MSELoss(reduction='mean')
        self.std_significance = sig
    
    def _predict(self, x_test, context = 10):
        predictors, responses, _ = construct_training_dataset(x_test, self.context)
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        print("inputs: ", inputs.shape)
        print(self.device)

        # Get the forecasts and generalised coefficients
        preds = self.model.forward(inputs)
        preds = preds.cpu().detach().numpy() if self.cuda else preds.detach().numpy()
        print(preds.shape)
        return np.concatenate((x_test[:self.context, :], preds), axis=0)
    
    def preprocess_data(self):
        predictors, responses, _ = construct_training_dataset(self.X, self.context)
        return predictors, responses
    
    def batch_generator(self, X, Y):
        x = Variable(torch.tensor(X, dtype=torch.float)).float().to(self.device).cuda()
        y = Variable(torch.tensor(Y, dtype=torch.float)).float().to(self.device).cuda()
        yield x,y
    
    def modelforward(self, X):
        return self.model.forward(X)
    
    def lossfn(self, pred, Y):
        targets = Variable(torch.tensor(Y, dtype=torch.float)).float().to(self.device)
        # print("Targets: ", targets, targets.shape)
        # print("Pred: ",pred, targets.shape)
        # print("Diff: ", targets - pred, targets.shape)
        #input("Looks good?")
        base_loss = sum([F.mse_loss(pred[:,i], targets[:,i]) for i in range(self.numVars)])
        return base_loss
                         
    def make_GC_graph(self, x,plotSwitch = False, argument = 0):
        def plot(truth, title):
            if(not plotSwitch):
                return
            plt.plot(truth)
            plt.title(title)
            plt.show()
            
        self.model.eval()
        with torch.no_grad():
            predictors, responses, _ = construct_training_dataset(x, self.context)
            X = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device).cuda()
            truth= x[self.context:]
            full = self.model.forward(X).cpu().numpy()
            plot(full, "Truthfull predictions")
            permuted = []
            for idx in range(self.numVars):
                newX = X.clone().cpu().numpy()
                random.shuffle(newX[:, :, idx])
                newX = torch.from_numpy(newX).cuda()
                prediction = self.model.forward(newX).cpu().numpy()
                plot(prediction, "Permuted variable: {}".format(idx))
                permuted.append(prediction)
            permuted_error = np.array([np.sum(np.square(x[:100]- truth[:100]), axis=0)  for x in permuted])
            true_error = np.sum(np.square(full[:100]-truth[:100]), axis=0)
            self.graph_est = np.zeros((self.numVars, self.numVars))
            self.true_graph = np.zeros((self.numVars, self.numVars))
            
            for pred_var in range(self.numVars):
                for cause_var in range(self.numVars):
                    reduced = permuted_error[cause_var][pred_var]
                    unreduced = true_error[pred_var]
                    print(reduced, unreduced)
                    n = 100
                    p = 10
                    stat= ((reduced-unreduced)/p)/(unreduced/(n-2*p-1))
                    self.true_graph[cause_var][pred_var] = scipy.stats.f.cdf(stat, n, n-2*p-1)
                    self.graph_est[cause_var][pred_var] = (scipy.stats.f.cdf(stat, n, n-2*p-1) < 0.05)*1.0
            self.graph_est_done = True
            return self.graph_est
        
    
    def make_causal_estimate(self):
        return np.ones((self.numVars, self.numVars))
    
    def predict(self, x_test):
        pred_series = self._predict(x_test)
        graph = self.make_GC_graph(x_test)
        return graph, pred_series
    

if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.lorenz96)
    series, causality_graph = lorenz_generator.integrate(p=12, T=3000, args=(10,))#1.2,.2,0.05,1.1))
    #_, series2, causality_graph = lorenz_generator.simulate(p=8, T=500, args= (10,))#82, 13.286))
    file = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/returns/random-rels_20_1_3_returns30007000.csv"
    gt = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/relationships/random-rels_20_1_3.csv"
    
    n = int(0.8*len(series))
    print(n)
    lstmTester = GCTester(series[:n], cuda = True)
    lstmTester.NUM_EPOCHS = 1000
    lstmTester.trainInherit()
    torch.cuda.empty_cache()
    gc.collect()
    metrics = Metrics(lstmTester, causality_graph, series)
    metrics.vis_pred(start = n)
    metrics.vis_causal_graphs()
    metrics.prec_rec_acc_f1()
        
        
        
        
        
