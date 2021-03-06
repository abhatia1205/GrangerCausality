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
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt


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
        self.graph_est, _ =  training_procedure_stable(self.X, self.order, self.layer_size, end_epoch, batch_size, lmbd, gamma, display=True, verbose = 1)
    
    
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
    
class GVARTesterStable(ModelInterface):
    
    def __init__(self, X, cuda = False):
        
        super(GVARTesterStable, self).__init__(cuda)
        self.X = X
        self.device = torch.device("cuda") if cuda else torch.device('cpu')
        self.num_vars = self.X.shape[1]
        self.order = 10
        self.layer_size = 5
        self.num_layers = 100
        self.senn = SENNGC(self.num_vars, self.order, self.layer_size, self.num_layers, self.device).to(self.device)
        self.senn2 = SENNGC(self.num_vars, self.order, self.layer_size, self.num_layers, self.device).to(self.device)
        self.graph_est = None
        self.senn3 = SENNGC(self.num_vars, self.order, self.layer_size, self.num_layers, self.device).to(self.device)
        self.graph_est = None
        self.coeffs = None
        
        self.base_loss = MSELoss()
        self.alpha = 0.5
        
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
        l = self.X.shape[0]
        predictors, responses, time_idx = construct_training_dataset(data=self.X[:l//2, :], order=self.order)
        yield predictors, responses
        predictors, responses, time_idx = construct_training_dataset(data=self.X[l//2:, :], order=self.order)
        yield predictors, responses
        predictors, responses, time_idx = construct_training_dataset(data=self.X, order=self.order)
        yield predictors, responses
    
    
    def batch_generator(self, X, Y):# generate tuples of (X_t, X_t+1), (Y_t, Y_t+1)
        X_t = X
        X_t1 = X[1:, :, :]

        yield (X_t, X_t1), Y
    
    def modelforward(self, X, senn):
        inputs = Variable(torch.tensor(X[0], dtype=torch.float)).float().to(self.device)
        next_inputs = Variable(torch.tensor(X[1], dtype=torch.float)).float().to(self.device)
        preds, coeffs = senn(inputs=inputs)
        return ((preds, coeffs), next_inputs)
    
    def lossfn(self, pred, Y, senn, lmbd=0.1, gamma=0.1):
        inputs_next = pred[1]
        pred, coeffs = pred[0]
        targets = Variable(torch.tensor(Y, dtype=torch.float)).float().to(self.device)
        base_loss = self.base_loss(pred, targets)
        penalty = (1 - self.alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                  self.alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))
                  
        preds_next, coeffs_next = senn(inputs=inputs_next)
        penalty_smooth = torch.norm(coeffs_next - coeffs[1:, :, :, :], p=2)
        
        loss = base_loss + lmbd * penalty + gamma * penalty_smooth
        return loss
    
    def find_alpha(self):
        Q = 20
        dgen = self.preprocess_data()
        
        predictors, responses = next(dgen)
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        preds, coeffs = self.senn(inputs=inputs)
        predictors, responses = next(dgen)
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        preds2, coeffs2 = self.senn2(inputs=inputs)
        
        a_hat_1 = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0)[0].detach().cpu().numpy()
        a_hat_2 = torch.max(torch.median(torch.abs(coeffs2), dim=0)[0], dim=0)[0].detach().cpu().numpy()
        
        alphas = np.linspace(1 / (a_hat_1.shape[0] * a_hat_1.shape[1]),
                         1 - a_hat_1.shape[0] / (a_hat_1.shape[0] * a_hat_1.shape[1]), Q)
        qs_1 = np.quantile(a=a_hat_1, q=alphas)
        qs_2 = np.quantile(a=a_hat_2, q=alphas)
        agreements = np.zeros((len(alphas), ))
        for i in range(len(alphas)):
            a_1_i = (a_hat_1 >= qs_1[i]) * 1.0
            a_2_i = (a_hat_2 >= qs_2[i]) * 1.0
            # NOTE: we ignore diagonal elements when evaluating stability
            agreements[i] = (balanced_accuracy_score(y_true=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                     y_pred=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten()) +
                             balanced_accuracy_score(y_pred=a_2_i[np.logical_not(np.eye(a_2_i.shape[0]))].flatten(),
                                                     y_true=a_1_i[np.logical_not(np.eye(a_1_i.shape[0]))].flatten())) / 2
        alpha_opt = alphas[np.argmax(agreements)]
        plt.plot(alphas, agreements)
        plt.xlabel("?")
        plt.ylabel("Stability Measure")
        plt.show()
    
        print("Max. stab. at ? = " + str(alpha_opt))

        return alpha_opt
        
        
    def make_causal_estimate(self):
        predictors, responses, time_idx = construct_training_dataset(data=self.X, order=self.order)
        inputs = Variable(torch.tensor(predictors, dtype=torch.float)).float().to(self.device)
        preds, coeffs = self.senn3(inputs=inputs)
        a_hat = torch.max(torch.median(torch.abs(coeffs), dim=0)[0], dim=0)[0].detach().cpu().numpy()
        q = np.quantile(a=a_hat, q=self.alpha_opt)
        causal_struct_estimate = (a_hat >= q) * 1.0
        print("Causal struct estimate: ", causal_struct_estimate)
        self.graph_est =  causal_struct_estimate
        return causal_struct_estimate
    
    def closure(self,  x, y, senn, **kwargs):
        opt = kwargs["opt"]
        opt.zero_grad()
        pred = self.modelforward(x, senn )
        loss = self.lossfn(pred, y, senn)
        loss.backward()
        opt.step()
        return loss
    
    def trainInherit(self):
        
        def train(senn):
            optimiser = optim.Adam(params=senn.parameters(), lr= self.lr)
            X_train, Y_train = next(dgen)
            self.pretrain_procedure()
            for epoch in range(self.NUM_EPOCHS):
                b_gen = self.batch_generator(X_train, Y_train)
                total_loss = 0
                for x_batch, y_batch in b_gen:
                    total_loss += self.closure(x_batch, y_batch, senn, opt = optimiser)
                print("Epoch " + str(epoch)+ " : incurred loss " + str(total_loss))
                history.append(total_loss)
            self.posttrain_procedure()
            return total_loss
        
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        history =[]
        
        dgen = self.preprocess_data()
        train(self.senn)
        history = []
        train(self.senn2)
        self.alpha_opt = self.find_alpha()
        total_loss = train(self.senn3)
        causal_estimate = self.make_causal_estimate()
        self.history = history
        

        return causal_estimate, total_loss, history