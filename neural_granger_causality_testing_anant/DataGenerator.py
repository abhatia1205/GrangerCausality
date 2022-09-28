# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:52:02 2022

@author: anant
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import pandas as pd
from GVAR.datasets.lotkaVolterra.multiple_lotka_volterra import MultiLotkaVolterra
import math
import sdeint

class DataGenerator():
    
    def __init__(self, fun, data=None, sigma = 0, **kwargs):
        self.func = fun
        self.kwargs = kwargs
        self.data = data
        self.sigma = sigma
    
    def noise(self,arr):
        if(self.sigma == 0):
            return arr
        noise = arr+ np.random.normal(0, self.sigma, len(arr))
        return noise

    def lorenz96(x, t):
        p = len(x)
        globF = 10
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
            F = args[0] if len(args) > 0 else globF
            for i in range(p):
                d[i] = (x[(i + 1) % p] - x[i - 2]) * x[i - 1] - x[i] + F
            return d
        return lorenz, gt

    def chua(x,t):
        p = len(x)
        assert p == 3
        def gt():
            return np.array([[1,1,0],
                             [1,1,1],
                             [1,1,1]])
        def fun(x,t, *args):
            d = np.zeros(p)
            # Loops over indices (with operations and Python underflow indexing handling edge cases)
            a,b,c, k = 1.3, .11, 7, 0
            alpha, beta = args if len(args) > 0 else 10.82, 14.286
            h = -b*math.sin(math.pi*x[0]/(2*a) +k )
            d[0] = alpha*(x[1] - h)
            d[1] = x[0] - x[1] + x[2]
            d[2] = -beta*x[1]
            return d
        return fun, gt

    def lotka_volterra(x, t):
        N = len(x)
        halfN = N//2
        numcomp = 2
        if((N/2) % numcomp != 0):
            raise ValueError("Half of num variables must be multiple of num competitors")
        def gt():
            init = np.zeros((N,N))
            for i in range(N):
                init[i,i]=1
                compstart = ((i//numcomp)*numcomp + halfN)%N
                init[i,compstart:compstart+numcomp ] = 1
            return init
        def funcint(x, t, *args):
            alpha, beta, delta, gamma = args
            dx = np.zeros(halfN)
            dy = np.zeros(halfN)
            xi = x[:halfN]
            yi = x[halfN:]
            for i in range(halfN):
                compstart = ((i//numcomp)*numcomp + halfN)%N
                xcomps = yi[compstart:compstart+numcomp]
                dx[i] = alpha*xi[i] - beta*xi[i]*sum(xcomps) - alpha*(xi[i]/200)**2
                ycomps = xi[compstart:compstart+numcomp]
                dy[i] = delta*yi[i]*sum(ycomps) - gamma*yi[i]
            return np.concatenate((dx, dy))
        return funcint, gt
        
    def add_gaussian_noise_post(data, perc, channels = None):
        if channels == None:
            channels = range(len(data[0]))
        retData = np.copy(data)
        size = len(retData)
        sigma = np.std(retData, axis=0)
        for i, channel in enumerate(channels):
            retData[:,channel] += np.random.normal(0, sigma[i]*perc, size = size)
        return retData
    
    def normalize(data):
        m = np.mean(data, axis = 0)
        s = np.std(data, axis=0)
        return (data-m)/s
    
    def integrate(self, p, T, delta_t=0.01, sd=0.1, burn_in=1000,
                       seed=0, args = ()):
        if seed is not None:
            np.random.seed(seed)
        
        if(self.func.__name__ == "lotka_volterra"):
            alpha, beta, delta, gamma = args
            gen = MultiLotkaVolterra(p=p//2, d=2, alpha=alpha, beta=beta, delta=delta, gamma=gamma, sigma=self.sigma)
            series, graph, _ = gen.simulate(int((T+burn_in)/delta_t), dt = delta_t)
            X = series[0] 
            return X[burn_in:], graph
        # Use scipy to solve ODE.
        x0 = np.random.uniform(low = -0.1, high = 0.1, size=p) + args[0]
        t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
        funcint, ground_truth = self.func(x0,t)
        def noisearr(x,t,*args):
            return np.diag(self.sigma*np.ones(p))
        X = sdeint.itoint(funcint, noisearr, x0, t)
        X = DataGenerator.normalize(X)
        return X[burn_in:], ground_truth()
    
    def simulate(self, p, T, delta_t=0.01, sd=0.1, burn_in=1000,
                       seed=0, args = ()):
        if seed is not None:
            np.random.seed(seed)
        
        if(self.func.__name__ == "lotka_volterra"):
            alpha, beta, delta, gamma = args
            gen = MultiLotkaVolterra(p=p//2, d=2, alpha=alpha, beta=beta, delta=delta, gamma=gamma, sigma=self.sigma)
            series, graph, _ = gen.simulate(int((T+burn_in)/delta_t), dt = delta_t)
            X = series[0] 
            return X[burn_in:], graph
        # Use scipy to solve ODE.
        x0 = np.random.uniform(low = -0.1, high = 0.1, size=p) + args[0]
        t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
        tf = (T + burn_in) * delta_t
        funcint, ground_truth = self.func(x0,t)
        X = odeint(funcint, x0, t, args= args)
        #X2 = np.transpose(solve_ivp(funcint3, (0, tf), x0, t_eval=t, args=args)['y'])
        #X += np.random.normal(scale=sd, size=(T + burn_in, p))
        X = DataGenerator.normalize(X)
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
    
    def finance(file, gtfile = None):
        x = pd.read_csv(file, index_col = False)
        if("Date" in list(x.columns)):
            x = x.drop(columns = ["Date"])
        x = x.values
        if(gtfile == None):
            return x, gtfile
        gt = DataGenerator.csv_to_graph(gtfile, x.shape[1])
        return x, gt



if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.lorenz96_func)
    series = lorenz_generator.create_series([[0.64994384, 0.01750787, 0.72402577, 0.14358566, 0.502893]], F = 8)
    lorenz_generator.ApEn_data(data = np.random.normal(20, 2, size = 20))