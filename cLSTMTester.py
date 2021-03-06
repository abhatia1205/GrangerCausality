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


class cLSTMTester(ModelInterface):
    
    def __init__(self, X, cuda = False):
        
        super(cLSTMTester, self).__init__(cuda)
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.origX = np.copy(X)
        self.X = torch.tensor(X[np.newaxis], dtype=torch.float32, device=self.device)
        self.X = self.X.cuda() if cuda else self.X
        self.numVars = self.X.shape[2]
        self.clstm = cLSTM(X.shape[-1], hidden = 100)
        self.clstm = self.clstm.cuda(self.device) if cuda else self.clstm
        self.parameters = self.clstm.parameters()
        
        self.context = 10
        self.base_loss = nn.MSELoss(reduction='mean')
        self.lam = 0
        self.lam_ridge = 0
    
    def train(self, max_iter = 4000):
        train_model_ista( self.clstm, self.X, context=10, lam=10.0, lam_ridge=1e-2, lr=1e-3, max_iter=max_iter,
    check_every=50, verbose =1)
    
    def make_GC_graph(self):
        return self.clstm.GC().cpu().data.numpy()
    
    def _predict(self, x_test, context = 10):
        origX = np.copy(x_test)
        origX = torch.tensor(origX[np.newaxis], dtype=torch.float32, device=self.device)
        X, Y = zip(*[arrange_input(x, context) for x in origX])
        print("Shape is: ", X)
        X = torch.cat(X, dim=0)
        print("Shape is: ", X.shape)
        Y = torch.cat(Y, dim=0)
        
        if self.cuda:
            pred = np.array([network(X)[0].cpu().detach().numpy() for network in self.clstm.networks]).squeeze()
        else:
            pred = np.array([network(X)[0].detach().numpy() for network in self.clstm.networks]).squeeze()
        a = pred[:, :, -1] #End of each array is a prediction
        b = pred[:, 0, :-1] #Start of first array is additional predictions that havent been accounted for
        c = np.zeros((pred.shape[0], 1)) #First prediction is padded (sketchy)
        pred = np.concatenate((c, b, a), axis= 1).transpose()
        return pred
    
    def preprocess_data(self):
        X, Y = zip(*[arrange_input(x, self.context) for x in self.X])
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        return X, Y
    
    def closure(self, x, y, **kwargs):
        self.clstm.zero_grad()
        pred = self.modelforward(x)
        loss = self.lossfn(pred, y)
        loss.backward()
        
        for param in self.clstm.parameters():
            param.data -= self.lr * param.grad
        # Take prox step.
        if self.lam > 0:
            for net in self.clstm.networks:
                prox_update(net, self.lam, self.lr)
        return loss
    
    def batch_generator(self, X, Y):
        yield X, Y
    
    def modelforward(self, X):
        return [network(X)[0] for network in self.clstm.networks]
    
    def lossfn(self, pred, Y):
        loss = sum([self.base_loss(pred[i][:, :, 0], Y[:, :, i]) for i in range(self.numVars)])
        ridge = sum([ridge_regularize(net, self.lam_ridge) for net in self.clstm.networks])
        smooth = loss + ridge
        return smooth
    
    def make_causal_estimate(self):
        return self.clstm.GC().cpu().data.numpy()
    
        
        
        
        
        
