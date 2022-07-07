# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:52:02 2022

@author: anant
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

class DataGenerator():
    
    def __init__(self, fun, data=None, **kwargs):
        self.func = fun
        self.kwargs = kwargs
        self.data = data

    def lorenz96_func(x, t):
        p = len(x)
        def gt():
            GC = np.zeros((p, p), dtype=int)
            for i in range(p):
                GC[i, i] = 1
                GC[i, (i + 1) % p] = 1
                GC[i, (i - 1) % p] = 1
                GC[i, (i - 2) % p] = 1
            return GC
        
        def lorenz(x,t, *args):
            d = np.zeros(p)
            # Loops over indices (with operations and Python underflow indexing handling edge cases)
            for i in range(p):
                d[i] = (x[(i + 1) % p] - x[i - 2]) * x[i - 1] - x[i] + args[0]
            return d
        return lorenz, gt

    def generalized_lambda(self, x):
        N = len(x)
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = self.func(x)
        return d

    def lotka_volterra(x, t):
        N = len(x)
        def gt():
            return np.ones((N,N))
        def funcint(x, t, *args):
            coefs, K, r = args
            d = np.zeros(N)
            for i in range(N):
                d[i] = (1 - coefs[i].dot(x)/K[i])*r[i]*x[i]
            return d
        return funcint, gt
    
    def create_series(self, initial, func = None, numSteps = 1000,dt = .01, **kwargs):
        if(func == None):
            func = self.func
        order = len(initial)
        
        initial = np.array(initial)
        ground_truth = func(initial[-1*order:], ground_truth = True, **kwargs)
        for i in range(numSteps):
            xNew = initial[-1] + dt*np.array(func(initial[-1*order:], **kwargs))
            initial = np.vstack([initial, xNew])
        # = initial.transpose()
        plt.plot(initial)
        plt.show()
        
        self.data = initial
        self.numVars = len(self.data[0])
        self.seriesLength = len(self.data)
        print("Created series: ", initial)
        
        return initial, ground_truth
    
    def add_gaussian_noise(self, interval, channels = None):
        if channels == None:
            channels = range(self.numVars)
        if self.data == None:
            print("Data hasn't been instantiated yet")
            return
        retData = np.copy(self.data)
        size = len(retData)
        sigma = [random.uniform(interval[0], interval[1]) for i in range(len(channels))]
        for i, channel in enumerate(channels):
            retData[:,channel] += np.random.normal(0, sigma[i], size = size)
        return retData
    
    def simulate(self, p, T, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0, args = ()):
        if seed is not None:
            np.random.seed(seed)
    
        # Use scipy to solve ODE.
        x0 = np.random.normal(scale=0.01, size=p)
        t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
        funcint, ground_truth = self.func(x0,t)
        X = odeint(funcint, x0, t, args= args)
        X += np.random.normal(scale=sd, size=(T + burn_in, p))
    
        return X[burn_in:], ground_truth()
    
    def csv_to_graph(gtfile, N):
        """Collects the total delay of indirect causal relationships."""
        gtdata = pd.read_csv(gtfile, header=None)
        ret = np.zeros((N,N))
        effects = gtdata[1]
        causes = gtdata[0]

        for i in range(len(effects)):
            key=effects[i]
            value=causes[i]
            ret[key][value] = 1
        return ret
    
    def finance(file, gtfile):
        x = pd.read_csv(file).values
        gt = DataGenerator.csv_to_graph(gtfile, x.shape[1])
        return x, gt

if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.lorenz96_func)
    series = lorenz_generator.create_series([[0.64994384, 0.01750787, 0.72402577, 0.14358566, 0.502893]], F = 8)
    lorenz_generator.ApEn_data(data = np.random.normal(20, 2, size = 20))