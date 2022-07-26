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
        input("Continue: ?")
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
    
    def make_GC_graph(self, x):
        def plot(truth, title):
            mean_over_time = np.mean(truth, axis=0)
            std_over_time = np.std(truth, axis=0)
            lower = mean_over_time - 3*std_over_time
            higher = mean_over_time + 3*std_over_time
            fig, axarr = plt.subplots(2,4, figsize=(10, 5))
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
                prediction = np.mean(np.std(prediction, axis=0), axis=0)
                permuted.append((prediction-truth)/truth)
            return np.array(permuted)
    def make_causal_estimate(self):
        self.make_GC_graph(self.X)
    
    def predict(self, x_test):
        pred_series = self._predict(x_test)
        graph = self.make_GC_graph(x_test)
        return graph, pred_series


    
    
        
        
        
        
        
