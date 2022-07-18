#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:53:14 2022

@author: s215863
"""
from GVAR.datasets.lotkaVolterra.multiple_lotka_volterra import MultiLotkaVolterra
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

if(__name__ == "__main__"):
    lorenz_generator = DataGenerator(DataGenerator.chua)
    #series, causality_graph = lorenz_generator.create_series([[0.64994384, 0.01750787, 0.72402577, 0.14358566, 0.502893]], F = 8)
    series, graph = lorenz_generator.simulate(p=3, burn_in=0, T=2000, args=(10.82, 13.286))
    plt.matshow(graph)
    plt.show()
    print(series)
    plt.plot(series)
    plt.show()
    
    
