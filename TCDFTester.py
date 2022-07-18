# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:22:59 2022

@author: anant
"""

from ModelInterface import ModelInterface
import torch
import numpy as np
import random
import torch.optim as optim
from torch.autograd import Variable

from TCDF import TCDF
from TCDF.model import ADDSTCN
import copy
import torch.nn.functional as F
import torch.nn as nn

class ConcatTCDF(nn.Module):
    def __init__(self, num_vars, layers, kernel_size, cuda, dilation_c):
        super().__init__()
        self.networks = nn.ModuleList([ADDSTCN(target, num_vars, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
                        for target in range(num_vars)])
        self.networks = self.networks.cuda() if cuda else self.networks
        
    
    def forward(self, X):
        self.networks.train()
        pred = [network(X)
                for network in self.networks]
        pred = torch.cat(pred, dim=2)
        return pred
    
    def calculate_losses(self, idx, X, Y, loss):
        pred = self.networks[idx](X)
        return loss(pred, Y[:, :, idx])
        
        

class TCDFTester(ModelInterface):
    
    def __init__(self, X, cuda = False):
        
        super(TCDFTester, self).__init__(cuda)
        self.X = X
        self.num_vars = self.X.shape[1]
        self.allcauses = None
        self.allrealllosses = None
        self.allscores = None
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.cuda = cuda
        
        self.kernel_size = 5
        self.layers = 3
        self.dilation_c = self.kernel_size
        self.models = [ADDSTCN(target, self.num_vars, self.layers, kernel_size=self.kernel_size, cuda=self.cuda, dilation_c=self.dilation_c)
                       for target in range(self.num_vars)]
        self.model = ConcatTCDF(self.num_vars, self.layers, self.kernel_size, self.cuda, self.kernel_size)
        self.model = self.model.cuda() if cuda else self.model
        self.parameters = self.model.parameters()
    
    def _predict(self, x_test):
        x_test = x_test.astype('float32').transpose()
        data_x = torch.from_numpy(x_test)
        x_test = Variable(data_x).cuda(device = self.device ) if self.cuda else Variable(data_x)
        x_test = x_test.unsqueeze(0).contiguous() #batch size 1
        #pred = np.array([ np.squeeze(model.forward(x_test).cpu().detach().numpy()) for model in self.models])
        pred = np.squeeze(self.model.forward(x_test).cpu().detach().numpy())
        print("Pred: ", pred, pred.shape)
        return pred
    
    def preprocess_data(self):
        X_train = np.vstack((np.zeros(self.num_vars), self.X[:-1, :]))
        data_x = torch.from_numpy(X_train.astype('float32').transpose())
        data_y = torch.from_numpy(self.X.astype('float32'))
        X_train, Y_train = Variable(data_x).cuda(), Variable(data_y).cuda()
        self.X_train, self.Y_train = X_train, Y_train
        print("X_train: ",X_train)
        print("Y_train: ",Y_train)
        return X_train, Y_train
    
    def pretrain_procedure(self):
        self.model.eval()
        X_train, Y_train = self.preprocess_data()
        self.firstloss = [0 for i in range(self.num_vars)]
        b_gen = self.batch_generator(X_train, Y_train)
        for x_batch, y_batch in b_gen:
            pred = self.model.forward(x_batch)
            for idx in range(self.num_vars):
                self.firstloss[idx] += F.mse_loss(pred[:,:,idx], y_batch[:, :, idx])
    
    def posttrain_procedure(self):
        self.model.eval()
        X_train, Y_train = self.preprocess_data()
        self.realloss = [0 for i in range(self.num_vars)]
        b_gen = self.batch_generator(X_train, Y_train)
        for x_batch, y_batch in b_gen:
            pred = self.model.forward(x_batch)
            for idx in range(self.num_vars):
                self.realloss[idx] += F.mse_loss(pred[:,:,idx], y_batch[:, :, idx])
    
    def batch_generator(self, X, Y):
        yield X.unsqueeze(0).contiguous(), Y.unsqueeze(0).contiguous()
    
    def modelforward(self, X):
        return self.model.forward(X)
    
    def lossfn(self, pred, Y):
        return sum([F.mse_loss(pred[:,:,i], Y[:,:,i]) for i in range(self.num_vars)])
    
    def make_causal_estimate(self):
        def estimate_model(idx):
            model = self.model.networks[idx]
            firstloss = self.firstloss[idx]
            realloss = self.realloss[idx]
            scores = model.fs_attention.data
            s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
            indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
            
            #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
            if len(s)<=5:
                potentials = []
                for i in indices:
                    if scores[i]>1.:
                        potentials.append(i)
            else:
                potentials = []
                gaps = []
                for i in range(len(s)-1):
                    if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
                        break
                    gap = s[i]-s[i+1]
                    gaps.append(gap)
                sortgaps = sorted(gaps, reverse=True)
                
                for i in range(0, len(gaps)):
                    largestgap = sortgaps[i]
                    index = gaps.index(largestgap)
                    ind = -1
                    if index<((len(s)-1)/2): #gap should be in first half
                        if index>0:
                            ind=index #gap should have index > 0, except if second score <1
                            break
                if ind<0:
                    ind = 0
                        
                potentials = indices[:ind+1].tolist()
            print("Potential causes: ", potentials)
            validated = copy.deepcopy(potentials)
            
            #Apply PIVM (permutes the values) to check if potential cause is true cause
            for pot in potentials:
                random.seed(1111)
                X_test2 = X_train.clone().cpu().numpy()
                random.shuffle(X_test2[:,pot,:][0])
                shuffled = torch.from_numpy(X_test2)
                if self.cuda:
                    shuffled=shuffled.cuda()
                model.eval()
                output = self.model.forward(shuffled)
                testloss = F.mse_loss(output[:,:,idx], Y_train[:,:,idx])
                testloss = testloss.cpu().data.item()
                self.testloss=testloss
                diff = firstloss- realloss
                testdiff = firstloss -testloss
                significance = 0.8
                if testdiff>(diff*significance): 
                    validated.remove(pot)
            
            return validated
        
        x,y= self.preprocess_data()
        X_train, Y_train = next(self.batch_generator(x,y))
        self.allcauses = dict()
        for idx in range(self.num_vars):
            self.allcauses[idx]= estimate_model(idx)
        
        N = self.num_vars
        ret = np.zeros((N,N))
        for key, l in self.allcauses.items():
            ret[key, l] = 1
        self.pred_graph = ret
        return ret
    
    def make_GC_graph(self):
        return self.pred_graph
        

class TCDFTesterOrig(ModelInterface):
    
    def __init__(self, X, cuda = False):
        
        super(TCDFTesterOrig, self).__init__(cuda)
        self.X = X
        self.num_vars = self.X.shape[1]
        self.allcauses = None
        self.allrealllosses = None
        self.allscores = None
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.cuda = cuda
        
        self.kernel_size = 5
        self.layers = 3
        self.dilation_c = self.kernel_size
        self.models = [ADDSTCN(target, self.num_vars, self.layers, kernel_size=self.kernel_size, cuda=self.cuda, dilation_c=self.dilation_c)
                       for target in range(self.num_vars)]
        self.model = ConcatTCDF(self.num_vars, self.layers, self.kernel_size, self.cuda, self.kernel_size).cuda()
        self.parameters = self.model.parameters()
    
    def findcauses2(self, x_train, target, epochs =2000, 
               log_interval = 500, lr = 0.01, optimizername = "Adam", seed = 1111, significance = 0.8):
        """Discovers potential causes of one target time series, validates these potential causes with PIVM and discovers the corresponding time delays"""
    
        print("\n", "Analysis started for target: ", target)
        torch.manual_seed(seed)
        
        Y_train = np.expand_dims(x_train[:, target], 0).astype('float32') #get the time series for selected target
        X_train = np.copy(x_train)
        #X_train = np.vstack((np.zeros(self.num_vars), self.X[:-1, :]))
        X_train[:, target] = np.insert(X_train[:-1, target], 0, 0) #remove current info for time signal (lag x by 1)
        X_train = X_train.astype('float32').transpose()
        data_x = torch.from_numpy(X_train)
        data_y = torch.from_numpy(Y_train)
        X_train, Y_train = Variable(data_x), Variable(data_y)
        
        X_train = X_train.unsqueeze(0).contiguous() #batch size 1
        Y_train = Y_train.unsqueeze(2).contiguous()
              
        model = self.models[target]
        if self.cuda:
            model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
    
        optimizer =optim.Adam(model.parameters(), lr=lr)    
        
        scores, firstloss = TCDF.train(1, X_train, Y_train, model, optimizer,log_interval,epochs)
        firstloss = firstloss.cpu().data.item()
        for ep in range(2, epochs+1):
            scores, realloss = TCDF.train(ep, X_train, Y_train, model, optimizer,log_interval,epochs)
        realloss = realloss.cpu().data.item()
        
        s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
        indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
        
        #attention interpretation to find tau: the threshold that distinguishes potential causes from non-causal time series
        if len(s)<=5:
            potentials = []
            for i in indices:
                if scores[i]>1.:
                    potentials.append(i)
        else:
            potentials = []
            gaps = []
            for i in range(len(s)-1):
                if s[i]<1.: #tau should be greater or equal to 1, so only consider scores >= 1
                    break
                gap = s[i]-s[i+1]
                gaps.append(gap)
            sortgaps = sorted(gaps, reverse=True)
            
            for i in range(0, len(gaps)):
                largestgap = sortgaps[i]
                index = gaps.index(largestgap)
                ind = -1
                if index<((len(s)-1)/2): #gap should be in first half
                    if index>0:
                        ind=index #gap should have index > 0, except if second score <1
                        break
            if ind<0:
                ind = 0
                    
            potentials = indices[:ind+1].tolist()
        print("Potential causes: ", potentials)
        validated = copy.deepcopy(potentials)
        
        #Apply PIVM (permutes the values) to check if potential cause is true cause
        for idx in potentials:
            random.seed(seed)
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:,idx,:][0])
            shuffled = torch.from_numpy(X_test2)
            if self.cuda:
                shuffled=shuffled.cuda()
            model.eval()
            output = model(shuffled)
            testloss = F.mse_loss(output, Y_train)
            testloss = testloss.cpu().data.item()
            
            diff = firstloss-realloss
            testdiff = firstloss-testloss
            
            
    
            if testdiff>(diff*significance): 
                validated.remove(idx) 
        
        return validated, realloss, scores.view(-1).cpu().detach().numpy().tolist()
    
    def train(self):
        allcauses = dict()
        allreallosses=dict()
        allscores=dict()
    
        for idx in range(self.num_vars):
            causes, realloss, scores = self.findcauses2(np.copy(self.X), idx)
    
            allscores[idx]=scores
            allcauses[idx]=causes
            allreallosses[idx]=realloss
    
        self.allcauses, self.allrealllosses, self.allscores =  allcauses, allreallosses, allscores
    
    def make_GC_graph(self):
        N = self.num_vars
        ret = np.zeros((N,N))
        for key, l in self.allcauses.items():
            ret[key, l] = 1
        return ret
    
    def _predict(self, x_test):
        x_test = x_test.astype('float32').transpose()
        data_x = torch.from_numpy(x_test)
        x_test = Variable(data_x).cuda(device = self.device ) if self.cuda else Variable(data_x)
        x_test = x_test.unsqueeze(0).contiguous() #batch size 1
        #pred = np.array([ np.squeeze(model.forward(x_test).cpu().detach().numpy()) for model in self.models])
        pred = np.squeeze(self.model.forward(x_test).cpu().detach().numpy())
        print("Pred: ", pred, pred.shape)
        return pred
    
    def trainInherit(self):
        self.train()
        
    
    
        