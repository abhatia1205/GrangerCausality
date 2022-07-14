# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:14:24 2022

@author: anant
"""

import random
import numpy as np
import torch
import torch.optim as optim

class ModelInterface():
    
    def __init__(self, cuda):
        self.cuda = cuda
        self.lr = 0.05
        self.NUM_EPOCHS = 20
        self.BATCH_SIZE = 32
        self.history = []
    def preprocess_data(self, X):
        pass
    
    def batch_generator(self, X, Y):
        pass
    
    def modelforward(self, x):
        pass
    
    def lossfn(self, pred, y):
        pass
    
    def make_causal_estimate(self):
        pass
    

    def closure(self,  x, y, **kwargs):
        opt = kwargs["opt"]
        opt.zero_grad()
        pred = self.modelforward(x)
        loss = self.lossfn(pred, y)
        loss.backward()
        opt.step()
        return loss

    def trainInherit(self):
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        
        optimiser = optim.Adam(params=self.parameters, lr= self.lr)
        history =[]
        
        X_train, Y_train = self.preprocess_data()
        self.pretrain_procedure()
        for epoch in range(self.NUM_EPOCHS):
            b_gen = self.batch_generator(X_train, Y_train)
            total_loss = 0
            for x_batch, y_batch in b_gen:
                total_loss += self.closure(x_batch, y_batch, opt = optimiser)
            print("Epoch " + str(epoch)+ " : incurred loss " + str(total_loss))
            history.append(total_loss)
        self.posttrain_procedure()
        self.history = history
        causal_estimate = self.make_causal_estimate()
        return causal_estimate, total_loss, history
    
    def pretrain_procedure(self):
        pass
    
    def posttrain_procedure(self):
        pass
    
    def make_GC_graph(self):
        pass
    
    def _predict(self, x_test):
        pass
    
    def predict(self, x_test):
        pred_series = self._predict(x_test)
        graph = self.make_GC_graph()
        return graph, pred_series
