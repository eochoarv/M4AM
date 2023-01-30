#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:16:55 2023

@author: eochoa
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from cvxopt import matrix, solvers
import math
import matplotlib.pyplot as plt
from utils import *
from scipy.spatial.distance import cdist 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from random import sample


class model_classifier():
    def __init__(self, model):
        #uper(GP_discover, self).__init__()
        self.model = model
    def train_data(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def test_data(self, X, y):
        self.X_test = X
        self.y_test = y
        
    def cv(self):
        pass
    
    def fit(self):
        return self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
class model_logReg(model_classifier):
    def __init__(self, model=LogisticRegression, 
                       random_state=0, 
                       multi_class='multinomial', 
                       penalty='l1', 
                       solver='saga',
                       C=1):
        super(model_logReg, self).__init__(model)
        self.model = model(random_state=random_state, 
                           multi_class=multi_class, 
                           penalty=penalty, 
                           solver=solver,
                           C=C)
        
    def cv(self, folds, C, plot_=False, verbose=False):
        metrics = defaultdict(list)
        for fold, (train, val) in enumerate(folds):
            # training data
            x_in = self.X_train[train, :]
            y_in = self.y_train[train]
            n, p = x_in.shape
            
            # validation data
            x_v = self.X_train[val, :]
            y_v = self.y_train[val]
            
            for i, c in enumerate(C):
                # track current 
                metrics['fold'].append(fold)
                metrics['c'].append(c)
        
                # train model
                
                self.model.C = c
                self.model.fit(x_in, y_in)            
                
                # record training fit 
                #y_hat = res_rf.predict(x_in)
                
                metrics['acc_train'].append(self.model.score(x_in, y_in))
                
                # make predictions on validation data
                #y_hat = res_rf.predict(x_v)
                
                # evaluate predictions
                metrics['acc_val'].append(self.model.score(x_v, y_v))
                
                metrics['n'].append(len(y_v))
                if verbose:
                    print(fold)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df['c'] = np.where(
            pd.notna(metrics_df['c']),
            metrics_df['c'],
            'None'
        )
        
        metrics_df['c'] = np.where(
            metrics_df['c'] == '5.0',
            '0' + metrics_df['c'],
            metrics_df['c']
        )
        cv_acc = metrics_df.groupby('c').mean()
        cv_acc = cv_acc.reset_index()
        cv_acc['c'] = np.double(cv_acc['c'])
        cv_acc = cv_acc.set_index('c')
        cv_acc.sort_index(inplace=True)
        
        best_c = cv_acc.index[np.argmax(cv_acc['acc_val'])]
        
        if(plot_):
            fig, ax = plt.subplots(nrows=1)
            
            _ = ax.plot(cv_acc['acc_train'],
                        label='acc_train')
            _ = ax.plot(cv_acc['acc_val'],
                        label='acc_val')
            
            _ = ax.set_xlabel('c')
            _ = ax.set_ylabel('Cross-Validated MSE')
            _ = ax.legend()
        
        print('best C: {c:.2f}'.format(c=best_c))
        self.model.C = best_c
        return best_c, np.max(cv_acc['acc_val'])
    
    
    
    
class model_randF(model_classifier):
    def __init__(self, model=RandomForestClassifier, 
                       n_estimators=50,
                       max_depth=10, 
                       random_state=0,
                       bootstrap=True,  
                       oob_score=True,
                       max_samples=0.5,
                       n_jobs=1):
        super(model_randF, self).__init__(model)
        self.model = model(n_estimators=n_estimators,
                           max_depth=max_depth, 
                           random_state=random_state,
                           bootstrap=bootstrap,  
                           oob_score=oob_score,
                           max_samples=max_samples,
                           n_jobs=n_jobs)
        
    def cv(self, folds, max_d, n_estimators, plot_=False, verbose=False):
        
        metrics = defaultdict(list)

        for fold, (train, val) in enumerate(folds):
            # training data
            x_in = self.X_train[train, :]
            y_in = self.y_train[train]
            n, p = x_in.shape
            
            # validation data
            x_v = self.X_train[val, :]
            y_v = self.y_train[val]
            
            for j, n_e in enumerate(n_estimators):
                for i, md in enumerate(max_d):
                    # track current 
                    metrics['fold'].append(fold)
                    metrics['n_estimators'].append(n_e)
                    metrics['max_depth'].append(md)
        
                    # train model
                    
                    self.model.n_estimators = n_e
                    self.model.max_depth = md
                    self.model.fit(x_in, y_in)   
        
                    # record training fit 
                    #y_hat = res_rf.predict(x_in)
                    metrics['acc_train'].append(self.model.score(x_in, y_in))
                    #metrics['acc_train'].append(np.mean(y_in == y_hat))
        
                    # make predictions on validation data
                    #y_hat = res_rf.predict(x_v)
        
                    # evaluate predictions
                    metrics['acc_val'].append(self.model.score(x_v, y_v))
                    #metrics['acc_val'].append(np.mean(y_v == y_hat))
        
                    metrics['n'].append(len(y_v))
                    if verbose:
                        print((fold, md, n_e))
                        
                        
        metrics_rf = pd.DataFrame(metrics)
        cv_acc = metrics_rf.groupby(['max_depth','n_estimators']).mean()
        cv_acc = cv_acc.reset_index()
        
        best_max_depth = cv_acc.iloc[np.argmax(cv_acc['acc_val']), 0]    
        best_n_estimators = cv_acc.iloc[np.argmax(cv_acc['acc_val']), 1]
        
        if(plot_):
            fig, ax = plt.subplots(ncols=2)
            fig.set_figheight(10)
            fig.set_figwidth(20)
                
            sns.heatmap(cv_acc.pivot(index='max_depth', columns='n_estimators', values='acc_val'))
            
            n_es = (cv_acc['n_estimators'].
             drop_duplicates().
             to_numpy())
            
            for n_e in n_es:
                alpha = n_e/n_es.max()
                #_ = ax.plot(cv_mse_rf.query('n_estimators == @n_e')['max_depth'],
                _ = ax[0].plot(cv_acc.query('n_estimators == @n_e')['max_depth'],
                               cv_acc.query('n_estimators == @n_e')['acc_train'],
                               label='acc_train',
                               color='b',
                               alpha=alpha)
                
                _ = ax[0].plot(cv_acc.query('n_estimators == @n_e')['max_depth'],
                               cv_acc.query('n_estimators == @n_e')['acc_val'],
                               label='acc_val',
                               color='b',
                               alpha=alpha)
                
                _ = ax[0].set_xlabel('tree depth')
                _ = ax[0].set_ylabel('Cross-Validated ACC')
                #_ = ax.legend()
            
        print('best max_depth: {md:.2f}'.format(md=best_max_depth))
        print('best n_estimators: {n_e:.2f}'.format(n_e=best_n_estimators))
            
        self.model.n_estimators = best_n_estimators
        self.model.max_depth = best_max_depth
                
        return best_max_depth, best_n_estimators, np.max(cv_acc['acc_val'])
    
    
class model_knn(model_classifier):
    def __init__(self, model=KNeighborsClassifier, 
                       n_neighbors=5,
                       n_jobs=1):
        super(model_knn, self).__init__(model)
        self.model = model(n_neighbors=n_neighbors,
                           n_jobs=n_jobs)
        
    def cv(self, folds, n_neighbors_list, plot_=False, verbose=False):
        
        metrics = defaultdict(list)

        for fold, (train, val) in enumerate(folds):
            # training data
            x_in = self.X_train[train, :]
            y_in = self.y_train[train]
            n, p = x_in.shape
            
            # validation data
            x_v = self.X_train[val, :]
            y_v = self.y_train[val]
            
            for j, nn in enumerate(n_neighbors_list):
        
                # track current 
                metrics['fold'].append(fold)
                metrics['n_neighbors'].append(nn)
        
                # train model
                
                self.model.n_neighbors = nn
                self.model.fit(x_in, y_in)   
        
                # record training fit 
                #y_hat = res_rf.predict(x_in)
                metrics['acc_train'].append(self.model.score(x_in, y_in))
                #metrics['acc_train'].append(np.mean(y_in == y_hat))
        
                # make predictions on validation data
                #y_hat = res_rf.predict(x_v)
        
                # evaluate predictions
                metrics['acc_val'].append(self.model.score(x_v, y_v))
                #metrics['acc_val'].append(np.mean(y_v == y_hat))
        
                metrics['n'].append(len(y_v))
                if verbose:
                    print((fold, nn))
                        
                        
        metrics_df = pd.DataFrame(metrics)

        cv_acc = metrics_df.groupby('n_neighbors').mean()
        cv_acc = cv_acc.reset_index()
        cv_acc['n_neighbors'] = np.double(cv_acc['n_neighbors'])
        cv_acc = cv_acc.set_index('n_neighbors')
        cv_acc.sort_index(inplace=True)
        
        best_nn = cv_acc.index[np.argmax(cv_acc['acc_val'])]
        
        if(plot_):
            fig, ax = plt.subplots(nrows=1)
            
            _ = ax.plot(cv_acc['acc_train'],
                        label='acc_train')
            _ = ax.plot(cv_acc['acc_val'],
                        label='acc_val')
            
            _ = ax.set_xlabel('n_neighbors')
            _ = ax.set_ylabel('Cross-Validated ACC')
            _ = ax.legend()
            
        print('best n_neighbors: {nn:.2f}'.format(nn=best_nn))
            
        self.model.n_neighbors = best_nn
                
        return best_nn, np.max(cv_acc['acc_val'])