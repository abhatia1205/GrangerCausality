# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:52:14 2022

@author: anant
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
import os
from torch.nn import MSELoss

class Metrics():
    
    def __init__(self, model, prediction_graph, timeseries, store_directory="trash",**kwargs):
        self.kwargs = kwargs
        self.graph = prediction_graph
        self.pred_graph, self.pred_timeseries = model.predict(timeseries)
        self.timeseries = timeseries
        self.error = timeseries - self.pred_timeseries
        self.model_name = type(model).__name__
        self.directory = store_directory
        if(not os.path.isdir(store_directory)):
            os.makedirs(store_directory)
    def vis_causal_graphs(self):
        # Check learned Granger causality
        plt.matshow(self.pred_graph)
        plt.show()
        GC_est = self.pred_graph
        GC = self.graph
        
        print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
        print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
        print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))
        
        # Make figures
        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
        axarr[0].imshow(GC, cmap='Blues')
        axarr[0].set_title('GC actual')
        axarr[0].set_ylabel('Affected series')
        axarr[0].set_xlabel('Causal series')
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])
        
        axarr[1].imshow(GC_est, cmap='Blues', vmin=0, vmax=1, extent=(0, len(GC_est), len(GC_est), 0))
        axarr[0].set_title(self.model_name)
        axarr[1].set_ylabel('Affected series')
        axarr[1].set_xlabel('Causal series')
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])
        
        # Mark disagreements
        for i in range(len(GC_est)):
            for j in range(len(GC_est)):
                if GC[i, j] != GC_est[i, j]:
                    rect = plt.Rectangle((j, i-0.05), 1, 1, facecolor='none', edgecolor='red', linewidth=1)
                    axarr[1].add_patch(rect)
        
        plt.savefig(os.path.join(self.directory,"causal_graph.jpg"))
        plt.show()
        
    def vis_pred(self, start = 0, timepoints = 100):
        timepoints =  min(timepoints, len(self.error)-start)
        pred = self.pred_timeseries[start:start+timepoints]
        error = self.error[start:start+timepoints]
        fig, axarr = plt.subplots(3, 1, figsize=(10, 5))
        axarr[0].plot(pred+error)
        axarr[0].set_title('Actual timeseries')
        axarr[1].plot(pred)
        axarr[1].set_title('Predicted timeseries: ' + self.model_name)
        axarr[2].plot(error)
        axarr[2].set_title('Error timeseries')
        plt.savefig(os.path.join(self.directory, "pred_graph.jpg"))
        plt.show()
        np.save(os.path.join(self.directory, "pred.npy"), self.pred_timeseries)
        
    def ApEn_series(U, m, r, **kwargs) -> float:
        """Approximate_entropy."""
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [
                len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                for x_i in x
            ]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
        N = len(U)
        plt.plot(U)
        plt.title(kwargs["title"])
        return _phi(m) - _phi(m + 1)
    
    def ApEn_data(self, data = None, **kwargs):
        if(not (isinstance(data, np.ndarray) or isinstance(data, list))):
            data = self.error
        if(not isinstance(data, np.ndarray)):
            data = np.array(data)
        ret = []
        m = 2
        r = 3
        if(len(data.shape) == 1):
            a = Metrics.ApEn_series(data, m, r, title = "Error for variable 0")
            print("ApEn is: ", a)
            return a
        for i in range(len(data[0])):
            ret.append(Metrics.ApEn_series(data[:, i], m, r, title = "Error for variable {}".format(i)))
        print("ApEn is: ", ret)
        return ret
    
    def prec_rec_acc_f1(self, **kwargs):
        if(len(self.pred_graph.shape) != 2 or len(self.graph.shape) != 2 ):
            raise ValueError("Shape is wrong")
        y_true = self.graph.flatten()
        y_pred = self.pred_graph.flatten()
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy= accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        sns.heatmap(confusion_matrix(y_true, y_pred))
        plt.show()

        print("Model: ", self.model_name)
        print("Preicison: {} \nRecall: {} \nAccuracy: {}\nF1 score: {}\n".format(precision, recall, accuracy, f1))
        f = open(os.path.join(self.directory, "numerics.txt"), 'w')
        f.write("Model: "+ self.model_name+"\n")
        f.write("Preicison: {} \nRecall: {} \nAccuracy: {}\nF1 score: {}\n".format(precision, recall, accuracy, f1))
        f.close()
        return (precision, recall, accuracy,f1)
    
    def error_stationarity(self):
        error_mat = self.error.transpose()
        ret = []
        for series in error_mat:
            adf_p = adfuller(series)[1]
            kpss_p = kpss(series)[1]
            lbox_p = max(abs(acorr_ljungbox(series)["lb_pvalue"]))
            ret.append([adf_p, kpss_p, lbox_p])
        N = len(error_mat)
        ind = np.arange(N) 
        width = 0.25
        ret = np.array(ret)
        
        def plot(data, testname):
            ind = np.arange(N)
            bar1 = plt.bar(ind, data, width, color = 'r')
            plt.xlabel("Time series")
            plt.ylabel('P values')
            plt.title(testname+  "P value visualization")
              
            plt.xticks(ind+width,[str(i+1) for i in range(N)])
            plt.show()
          
        adf = ret[:, 0]
        plot(adf, "ADF")
        kps = ret[:, 1]
        plot(kps, "KPSS")
        lbox = ret[:, 2]
        plot(lbox, "Ljung Box")
        f = open(os.path.join(self.directory, "numerics.txt"), 'a')
        mean_arr = np.mean(ret, axis=0)
        f.write("ADF Pval: {} \nKPSS Pval: {} \nLBox Pval: {}\n".format(mean_arr[0], mean_arr[1], mean_arr[2]))
        f.close()
        return ret, mean_arr
    
    def error_variance(self):
        arr = np.std(self.error, axis=0)
        orig_std = np.std(self.timeseries, axis=0)
        print("Error std: ", arr)
        print("Percentage std to orig: ", arr/orig_std)
        f = open(os.path.join(self.directory, "numerics.txt"), 'a')
        f.write("Error std: {}\n".format(arr))
        f.write("Percentage std to orig: {}\n".format(arr/orig_std))
        
        loss = np.sum(self.error**2, axis=0)
        print("MSE Loss from orig: {}".format(loss))
        print("PErcentage MSE from orig: {}".format(loss/(np.sum(self.timeseries, axis=0))))
        f.write("MSE Loss from orig: {}\n".format(loss))
        f.write("PErcentage MSE from orig: {}\n".format(loss/(np.sum(self.timeseries, axis=0))))
        f.close()
        return arr
    
    def test(self, data=None, **kwargs):
        self.vis_causal_graphs()
        self.vis_pred(**kwargs)
        self.prec_rec_acc_f1()
        self.error_variance()
        self.error_stationarity()

    
    
    
    
            
            
            
            
        
        
    
