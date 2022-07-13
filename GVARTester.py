# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:20:25 2022

@author: anant
"""

from ModelInterface import ModelInterface
import torch
import numpy as np
from GVAR.sennmodels.senn import SENNGC
from GVAR.training import training_procedure_stable
from GVAR.utils import construct_training_dataset
import random
import torch.optim as optim
from torch.nn import MSELoss
from torch.autograd import Variable

class GVARTester(ModelInterface):
    
    def __init__(self, X, cuda = False):
        
        super(GVARTester, self).__init__(cuda)
        self.X = X
        self.device = torch.device("cuda") if cuda else torch.device('cpu')
        self.num_vars = self.X.shape[1]
        self.order = 10
        self.layer_size = 5
        self.num_layers = 100
        self.senn = SENNGC(self.num_vars, self.order, self.layer_size, self.num_layers, self.device).to(self.device)
        self.graph_est = None
        self.coeffs = None
        
        self.base_loss = MSELoss()
        self.alpha = 0.5
        self.parameters = self.senn.parameters()
        
    def train(self, end_epoch: int = 40, batch_size: int = 4, lmbd: float = 0.1,
                       gamma: float = 0.1, seed=42,  initial_learning_rate=0.001, beta_1=0.9,
                       beta_2=0.999, use_cuda=True, verbose=True, test_data=None):
        self.graph_est, _ =  training_procedure_stable(self.X, self.order, self.layer_size, end_epoch, batch_size, lmbd, gamma, verbose = 1)
    
    
    def make_GC_graph(self):
        print("Graph Estimate: ", self.graph_est, self.graph_est.shape)
        return self.graph_est
    
    def _predict(self, x_test, batch_size = 1):
        predictors, responses, time_idx = construct_training_dataset(data=x_test, order=self.order)
        
        print("Predictors: ", predictors, predictors.shape)
        print("Respeonses: ", responses, responses.shape)
        print("Time index: ", time_idx, time_idx.shape)
        
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        print("inputs: ", inputs.shape)
        print(self.device)

        # Get the forecasts and generalised coefficients
        preds, coeffs = self.senn(inputs=inputs)
        preds = preds.cpu().detach().numpy() if self.cuda else preds.detach().numpy()
        return np.concatenate((x_test[:self.order, :], preds), axis=0)
    
    def preprocess_data(self):
        predictors, responses, time_idx = construct_training_dataset(data=self.X, order=self.order)
        return predictors, responses
    
    
    def batch_generator(self, X, Y):# generate tuples of (X_t, X_t+1), (Y_t, Y_t+1)
        X_t = X
        X_t1 = X[1:, :, :]

        yield (X_t, X_t1), Y
    
    def modelforward(self, X):
        inputs = Variable(torch.tensor(X[0], dtype=torch.float)).float().to(self.device)
        next_inputs = Variable(torch.tensor(X[1], dtype=torch.float)).float().to(self.device)
        preds, coeffs = self.senn(inputs=inputs)
        return ((preds, coeffs), next_inputs)
    
    def lossfn(self, pred, Y, lmbd=0.1, gamma=0.1):
        inputs_next = pred[1]
        pred, coeffs = pred[0]
        targets = Variable(torch.tensor(Y, dtype=torch.float)).float().to(self.device)
        base_loss = self.base_loss(pred, targets)
        penalty = (1 - self.alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                  self.alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))
                  
        preds_next, coeffs_next = self.senn(inputs=inputs_next)
        penalty_smooth = torch.norm(coeffs_next - coeffs[1:, :, :, :], p=2)
        
        loss = base_loss + lmbd * penalty + gamma * penalty_smooth
        return loss
    
    def make_causal_estimate(self):
        predictors, responses, time_idx = construct_training_dataset(data=self.X, order=self.order)
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        preds, coeffs = self.senn(inputs=inputs)
        causal_struct_estimate = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0)[0].detach().cpu().numpy()
        causal_struct_estimate = (causal_struct_estimate >= 0.5) * 1.0
        print("Causal struct estimate: ", causal_struct_estimate)
        self.graph_est =  causal_struct_estimate
        return causal_struct_estimate