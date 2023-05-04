#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:48:49 2023

@author: eochoa
"""

import numpy as np
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(1.0)
#expit = lambda x: 1/(1+np.exp(-x))
sigm = lambda x: 1/(1+np.exp(-x))

x = np.random.multivariate_normal(np.zeros(10), np.eye(10))

# 1) Sample prior (M)

class GP_classifier():
    def __init__(self, k, x, y):
        self.k = k
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.K = self.k(self.x)
        
    #def train_data(x,y):
    #    self.x = x
    #    self.y = y
        
    def init_priors(self):
        f = np.random.multivariate_normal(np.zeros(self.n), self.K)
        self.f = f
        
    def update_posterior(self, tol=1e-3):
        #x = self.x
        y = self.y
        f = self.f
        K = self.K
        t = (y+1)/2
        f_prev = f.copy() + 1
        
        while np.linalg.norm(f - f_prev) > tol:
            pi = sigm(f)
            w = t - pi
            W = np.diag(pi*(1-pi))
            f_prev = f.copy()
            f = np.linalg.inv(np.linalg.inv(K) + W) @ (W @ f + w)
            
        self.f = f
        
        return f
    
    def posterior_pred(self, x_star):
        x = self.x
        K = self.K
        f = self.f
        pi = sigm(f)
        W = np.diag(pi*(1-pi))
        
        k_star = kernel(x, x_star).flatten()
        k_star_star = kernel(x_star, x_star).flatten() 
        
        
        mu_star = k_star @ np.linalg.inv(K) @ f
        v_star = k_star_star - k_star @ np.linalg.inv(K + np.linalg.inv(W)) @ k_star
        
        pi_star = sigm(1 / np.sqrt(1+np.pi*v_star/8) * mu_star)
        self.pi_star = pi_star
        
        return pi_star
    
    def posterior_params(self, x_star):
        x = self.x
        K = self.K
        f = self.f
        pi = sigm(f)
        W = np.diag(pi*(1-pi))
        
        k_star = kernel(x, x_star).flatten()
        k_star_star = kernel(x_star, x_star).flatten() 
        
        
        mu_star = k_star @ np.linalg.inv(K) @ f
        v_star = k_star_star - k_star @ np.linalg.inv(K + np.linalg.inv(W)) @ k_star
        
        
        return mu_star, v_star
    
    def posterior_pred_rand(self, x_star):
        x = self.x
        K = self.K
        f = self.f
        pi = sigm(f)
        W = np.diag(pi*(1-pi))
        
        k_star = kernel(x, x_star).flatten()
        k_star_star = kernel(x_star, x_star).flatten() 
        
        
        mu_star = k_star @ np.linalg.inv(K) @ f
        v_star = k_star_star - k_star @ np.linalg.inv(K + np.linalg.inv(W)) @ k_star
        pi_star = sigm(np.random.normal(mu_star, v_star))
        self.pi_star = pi_star
        
        return pi_star
        