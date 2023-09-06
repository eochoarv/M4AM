#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:04:22 2023

@author: eochoa
"""
import numpy as np

def d_gauss(mu1, mu2):
    return ((mu1 - mu2) ** 2) / 2

def I(mu1, mu2, alpha):
    return alpha * d_gauss(mu1, alpha * mu1 + (1-alpha) * mu2) + (1 - alpha) * d_gauss(mu2, alpha * mu1 + (1-alpha) * mu2)

def g(mu1, mua, x):
    return (1 + x) * I(mu1, mua, 1 / (1 + x))
    
def x(mu1, mua, y):
    x1 = 0
    x2 = 1000
    if mu1 == mua:
        return 1
    
    x_mid = (x1 + x2) / 2
    #i = 0
    while abs(g(mu1, mua, x_mid) - y) > 1e-10:
        if g(mu1, mua, x_mid) > y:
            x2 = (x1 + x2) / 2
        else:
            x1 = (x1 + x2) / 2
        
        x_mid = (x1 + x2) / 2
        #i = i + 1
            
    return x_mid

def F(mu, y):
    m = mu.shape[0]
    y_max = (d_gauss(np.sort(mu)[-1],np.sort(mu)[-2]))
    if y > y_max:
        return 1e10
    i_star = np.argmax(mu)
    x_ = [x(mu[i_star], mu[a], y) for a in range(m)]
    return np.sum([(d_gauss(mu[i_star], (mu[i_star] + x_[a] * mu[a]) / (1 + x_[a])) / 
                    d_gauss(mu[a], (mu[i_star] + x_[a] * mu[a]) / (1 + x_[a]))) 
                   for a in np.setdiff1d(range(m), i_star)])
    
def solver_F(mu):
    
    y1 = 0
    y2 = d_gauss(np.sort(mu)[-1],np.sort(mu)[-2]) - 1e-10
    #ymax = (d_gauss(np.sort(mu)[-1],np.sort(mu)[-2]))
    y_mid = (y1 + y2) / 2
    i = 0
    while abs(F(mu, y_mid) - 1) > 1e-3 and i < 100:
        #print(F(mu, y_mid))
        if F(mu, y_mid) > 1:
            y2 = (y1 + y2) / 2
        else:
            y1 = (y1 + y2) / 2
        
        y_mid = (y1 + y2) / 2
        #print(F(mu, y_mid))
        i = i + 1
            
    return y_mid

def solver_alpha(mu):
    m = mu.shape[0]
    i_star = np.argmax(mu)
    y_star = solver_F(mu)
    alpha_star = (np.array([x(mu[i_star], mu[a], y_star) for a in range(m)]) / 
                  (np.sum([x(mu[i_star], mu[a], y_star) for a in range(m)])))
    return alpha_star
