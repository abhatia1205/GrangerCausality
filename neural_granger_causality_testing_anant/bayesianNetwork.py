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

class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):

        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size # user-defined

        self.hidden_size_1 = 128 # number of encoder cells (from paper)
        self.hidden_size_2 = 32 # number of decoder cells (from paper)
        self.stacked_layers = 2 # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.6# arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features, 
                             self.hidden_size_1, 
                             num_layers=self.stacked_layers,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size_1, output_length)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda")
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
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

class ConcatBN(nn.Module):
    def __init__(self, n_features, output_length, batch_size, cuda = True):
        super().__init__()
        self.networks = nn.ModuleList([BayesianLSTM(n_features, output_length, batch_size) for i in range(n_features)])
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


class BNTester(ModelInterface):
    
    def __init__(self, X, cuda = True):
        
        super(BNTester, self).__init__(cuda)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.X = X
        self.numVars = self.X.shape[1]
        self.model = ConcatBN(self.numVars, 1, 1)
        self.model = self.model.cuda() if cuda else self.model
        self.parameters = self.model.parameters()
        self.base_loss = MSELoss()
        self.graph_est_done = False
        
        self.context = 10
        self.base_loss = nn.MSELoss(reduction='mean')
        self.lam = 0
        self.lam_ridge = 0
    
    def create_sliding_window(data, sequence_length, stride=1):
        X_list, y_list = [], []
        for i in range(len(data)):
          if (i + sequence_length) < len(data):
            X_list.append(data[i:i+sequence_length:stride, :])
            y_list.append(data[i+sequence_length])
        return np.array(X_list), np.array(y_list)
    
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
        print("Shape: ", predictors.shape, responses.shape)
        print("Predictors: ", predictors[0])
        print("Responses: ", responses[0])
        print("Orig: ", self.X[:12])
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
    
    def make_GC_graph(self, x, plotSwitch = False, q = 0.5):
        if(self.graph_est_done):
            return self.graph_est
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
            predictors, responses, _ = construct_training_dataset(x, self.context)
            X = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device).cuda()
            truth = []
            for i in range(50):
                truth.append(self.model.forward(X))
                torch.cuda.empty_cache()
                gc.collect()
            truth = [x.cpu().numpy() for x in truth]
            plot(truth, "Truthfull predictions")
            truth = np.mean(np.std(truth, axis=0), axis=0)
            permuted_means = []
            for idx in range(self.numVars):
                newX = X.clone().cpu().numpy()
                random.shuffle(newX[:, :, idx])
                newX = torch.from_numpy(newX).cuda()
                prediction = []
                for i in range(50):
                    prediction.append(self.model.forward(newX))
                    torch.cuda.empty_cache()
                    gc.collect()
                prediction = [x.cpu().numpy() for x in prediction]
                plot(prediction, "Permuted variable: {}".format(idx))
                prediction = np.mean(np.std(prediction, axis=0), axis=0)
                permuted_means.append((prediction-truth)/truth)
            self.true_graph = np.abs(np.array(permuted_means))
            mean = np.mean(self.true_graph)
            self.graph_est = (self.true_graph >= mean)*1.0
            self.graph_est.transpose()
            self.graph_est_done = True
            return self.graph_est
    
    def E_n_Var_t(arr):
        return np.mean(np.std(arr, axis=0), axis=0)
    
    def E_t_Var_n(arr):
        return np.mean(np.std(arr, axis=1), axis=0)
    
    def mean_threshold(truth, permuted):
        permuted_means = (permuted-truth)/truth
        true_graph = np.abs(np.array(permuted_means))
        mean = np.mean(true_graph)
        graph_est = (true_graph >= mean)*1.0
        return graph_est.transpose()
    
    def ESD_t_Var_n(arr):
        return np.array([np.mean(np.std(arr, axis=1), axis=0), np.std(np.std(arr, axis=1), axis=0)])
    
    def ESD_n_Var_t(arr):
        return np.array([np.mean(np.std(arr, axis=0), axis=0), np.std(np.std(arr, axis=0), axis=0)])
    
    def ttest_threshold_mean(truth, permuted, num_obs=50):
        truth_mean, truth_std = truth[0], truth[1]
        permuted_mean = permuted[:,0,:]
        permuted_std = permuted[:,1,:]
        true_graph = np.zeros((len(truth_mean), len(truth_mean)))
        for i in range(len(truth_mean)):
            for j in range(len(truth_mean)):
                true_graph[i,j] = ttest_ind_from_stats(permuted_mean[i][j], permuted_std[i][j], num_obs,
                                                truth_mean[j], truth_std[j], num_obs).pvalue
        mean = np.quantile(true_graph, 0.1)
        print("Mean is:", mean)
        graph_est = (true_graph <= mean)*1.0
        return graph_est.transpose()
    
    def ttest_threshold_005(truth, permuted, num_obs=50):
        truth_mean, truth_std = truth[0], truth[1]
        permuted_mean = permuted[:,0,:]
        permuted_std = permuted[:,1,:]
        true_graph = np.zeros((len(truth_mean), len(truth_mean)))
        for i in range(len(truth_mean)):
            for j in range(len(truth_mean)):
                true_graph[i,j] = ttest_ind_from_stats(permuted_mean[i][j], permuted_std[i][j], num_obs,
                                                truth_mean[j], truth_std[j], num_obs).pvalue
        alpha = 0.05
        graph_est = (true_graph <= alpha)*1.0
        return graph_est.transpose()
                         
        
        
    
    def make_GC_graph2(self, x, singular_iteration, make_final_graph,plotSwitch = False, q = 0.5):
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
            predictors, responses, _ = construct_training_dataset(x, self.context)
            X = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device).cuda()
            truth = []
            for i in range(50):
                truth.append(self.model.forward(X))
                torch.cuda.empty_cache()
                gc.collect()
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
                    torch.cuda.empty_cache()
                    gc.collect()
                prediction = [x.cpu().numpy() for x in prediction]
                plot(prediction, "Permuted variable: {}".format(idx))
                prediction = singular_iteration(prediction)
                permuted.append(prediction)
          
            self.graph_est = make_final_graph(truth, np.array(permuted))
            self.graph_est_done = True
            return self.graph_est
        
    
    def make_causal_estimate(self):
        return np.ones((self.numVars, self.numVars))
        #self.make_GC_graph(self.X)
    
    def predict(self, x_test):
        pred_series = self._predict(x_test)
        graph = self.make_GC_graph(x_test)
        return graph, pred_series


if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.lotka_volterra)
    #series, causality_graph = lorenz_generator.create_series([[0.64994384, 0.01750787, 0.72402577, 0.14358566, 0.502893]], F = 8)
    series, causality_graph = lorenz_generator.integrate(p=12, T=3000, args=(1.2,.2,0.05,1.1))
    #_, series2, causality_graph = lorenz_generator.simulate(p=8, T=500, args= (10,))#82, 13.286))
    file = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/returns/random-rels_20_1_3_returns30007000.csv"
    gt = "/home2/s215863/Desktop/Granger Causality/FinanceCPT/relationships/random-rels_20_1_3.csv"
    #series, causality_graph = DataGenerator.finance(file, gt)
    torch.cuda.empty_cache()
    gc.collect()
    n = int(0.8*len(series))
    lstmTester = BNTester(series[:n], cuda = True)
    lstmTester.NUM_EPOCHS = 1000
    lstmTester.trainInherit()
    torch.cuda.empty_cache()
    gc.collect()
    metrics = Metrics(lstmTester, causality_graph, series)
    metrics.vis_pred(start = n)
    metrics.vis_causal_graphs()
    torch.cuda.empty_cache()
    gc.collect()
    
    funcs = [(BNTester.E_n_Var_t, BNTester.mean_threshold),
             (BNTester.E_t_Var_n, BNTester.mean_threshold),
             (BNTester.ESD_n_Var_t, BNTester.ttest_threshold_mean),
             (BNTester.ESD_t_Var_n, BNTester.ttest_threshold_mean),
             (BNTester.ESD_n_Var_t, BNTester.ttest_threshold_005),
             (BNTester.ESD_t_Var_n, BNTester.ttest_threshold_005),
             ]
    for a,b in funcs:
        metrics.pred_graph = lstmTester.make_GC_graph2(series, a, b)
        print(a.__name__, b.__name__)
        metrics.vis_causal_graphs()
        metrics.prec_rec_acc_f1()
        torch.cuda.empty_cache()
        gc.collect()
    
    # lstmTester2 = GVARTesterTRGC(series[:n], cuda = True)
    # lstmTester2.NUM_EPOCHS = 500
    # lstmTester2.trainInherit()
    # metrics2 = Metrics(lstmTester2, causality_graph, series)
    # metrics2.vis_pred(start=n)
    # metrics2.vis_causal_graphs()
    # metrics2.prec_rec_acc_f1()
    
    # 
    
# 
    
    print("Done")
        
        
        
        
        
