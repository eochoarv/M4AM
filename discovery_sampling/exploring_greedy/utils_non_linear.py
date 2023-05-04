#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:22:08 2023

@author: eochoa
"""

import pdb
import numpy as np
#from matrix_completion import svt_solve, pmf_solve, calc_unobserved_rmse
from numpy.random import randn
from copy import deepcopy

from numpy.random import randn
from copy import deepcopy
import bayes_logistic as bl
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import polynomial_kernel
from scipy.stats import multivariate_normal
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import torch

expit = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x / (1-x))

class discover:
    def __init__(self):
        self.initi_prob()
        self.actions = [] # labeled set
        self.rewards = [] # previous rewards
        self.tau = np.nan
    def initi_prob(self):
        pass
    def refresh(self):
        self.actions = [] # labeled set
        self.rewards = [] # previous rewards
        self.tau = np.nan
    def get_reward(self, action):
        pass
    def take_action(self, action):
        self.actions.append(action)
        reward, side_info = self.get_reward(action)
        self.rewards.append(reward)
        return self.rewards[-1], side_info
    def regret(self):
        pass
    
class f_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1, theta=None, rwd_func=None, rwd_distr=None):
        super(f_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.noise = noise
        self.sig_coef = sig_coef
        self.rwd_func = rwd_func
        self.rwd_distr = rwd_distr
        #self.init_linear()
        if theta is None:
            self.theta = 5*np.random.rand(d)
        else:
            self.theta = theta
        #self.tau = np.nan
    def get_reward(self, action):
        if self.rwd_func is None:
            f = self.x[action,:] @ self.theta
            r = np.random.normal(f, self.noise, 1)[0]
        else:
            f = self.rwd_func(self.x[action,:], self.theta)
            r = np.random.normal(f, self.noise, 1)[0]
        return r , None

    
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        f_star = self.rwd_func(self.x, self.theta)
        f = self.rwd_func(self.x[self.actions,:], self.theta)
        reg = (np.max(f_star) - f).cumsum()
        
        return reg
    
    def regret_search(self):
        f_star = self.x @ self.theta
        f_star = np.sort(f_star)[::-1]
        #p_star = np.sort(expit(f_star))[::-1]
        f = self.x[self.actions, :] @ self.theta
        reg = (f_star[:len(self.actions)] - f).cumsum()
        
        return reg
    
    
class linear_solver:
    def __init__(self, prob):
        self.prob = prob
    def step(self):
        pass
    def run(self, T):
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T, replace=True)
        for t in range(T):
            self.prob.take_action(acts[t])
        return self.prob.regret(), self.prob.rewards
    
    
class Greedy_linear(linear_solver):
    def __init__(self, prob, epsilon=0):
        super(Greedy_linear, self).__init__(prob)
        self.epsilon = epsilon
        self.init_priors()
    def init_priors(self, s0 = 1.0):
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        f = self.prob.x[action, :]
        s_inv = np.linalg.inv(self.sigma_t)
        ffT = np.outer(f, f)
        # pdb.set_trace()
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        theta_t = self.mu_t
        pred = self.prob.x @ theta_t.transpose()
        self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
        return self.prob.regret(), self.prob.rewards

class Greedy_f(linear_solver):
    def __init__(self, prob, reg_func=None, k=0):
        super(Greedy_f, self).__init__(prob)
        self.reg_func = reg_func
        self.init_model()
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
    def init_model(self):
        self.mu_t = np.zeros(self.prob.m)
    def update_model(self, action, r):
        
        x_ = self.prob.x[self.prob.actions + [action], :]
        y_ = np.array(self.prob.rewards + [r])
        
        self.reg_func.fit(x_, y_)
                 
        f = self.reg_func.predict(self.prob.x)
        self.mu_t = f
        return f
    def step(self):
        mu_t = self.mu_t
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(mu_t)
        return at
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = t
            else:
                action = self.step()
                
            #action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_model(action, r)
        return self.prob.regret(), self.prob.rewards


class TS_linear(linear_solver):
    def __init__(self, prob):
        super(TS_linear, self).__init__(prob)
        self.init_priors()
    def init_priors(self, s0 = 1.0):
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        f = self.prob.x[action, :]
        s_inv = np.linalg.inv(self.sigma_t)
        ffT = np.outer(f, f)
        # pdb.set_trace()
        mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        theta_t = np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t.transpose()
        self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
        return self.prob.regret(), self.prob.rewards

class Greedy_Frechet(linear_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, ss=0, k=0, dim=0, supp=None):
        super(Greedy_Frechet, self).__init__(prob)
        self.s0 = s0
        self.supp = supp
        if dim == 0:
            self.dim = self.prob.d
        else:
            self.dim = dim
            
        self.init_priors()
        # number of pulls for each arms
        self.S_k = np.zeros(self.prob.m)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
            
        if ss == 0:
            self.ss = self.prob.m
        else:
            self.ss = ss

    def init_priors(self):
        self.scores_t = np.zeros(self.prob.m)
        #self.mu_t = expit(self.scores_t)
        self.mu_t = 0.5 * np.ones(self.prob.m)
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        n = self.S_k[action]
            
        if n == 0:
            self.mu_t[action] = r
        else:
            self.mu_t[action] = (n * self.mu_t[action] + r) / (n + 1)
                
            
        #self.scores_t[action] = logit(np.clip(self.mu_t[action], 1e-7, 1-1e-7))
        
        #mu_a = np.clip(self.mu_t[action],1e-7,1-1e-7)
        
        #self.scores_t[action] = (n / (n + 1) * self.scores_t[action] + 
        #                         1 / (n + 1) * 1 / (mu_a*(1-mu_a)))
    
        self.S_k[action] = n + 1
            
        #select top selected d arms
        top_arms = np.argsort(self.S_k)[::-1][:self.dim]
            
        
        #dist = [bernoulli(p) for p in self.mu_t[top_arms]]
        u = np.linspace(1e-7, 1-1e-7, 1000)
        
        quantiles = [
            (pd.DataFrame({'action':self.prob.actions, 'reward':np.array(self.prob.rewards).flatten()}).
             groupby('action').
             quantile(u).
             reset_index().query('action == @i').iloc[:,2].to_numpy()) for i in top_arms
            ]
        
        
        x = self.prob.x
        dr = distr_reg_2()
        #dr = distr_reg_3()
        #dr.init_kernel()
        mus = []
        for i in range(self.prob.m):
            #g, _ = dr.predict(x_new=x[i], x=x[top_arms], y=dist,M=1000)
            g, _ = dr.predict(x_new=x[i], x=x[top_arms], y=quantiles,M=1000)
            if self.supp is None:
                iso_reg = IsotonicRegression().fit(u, g)
            else:
                iso_reg = IsotonicRegression(y_min=self.supp[0], y_max=self.supp[1]).fit(u, g)
            #mu_i = np.sum(np.diff(iso_reg.predict(u),1) * (1-u)[1:])
            #mu_i = iso_reg.predict(u)[500]
            df = pd.DataFrame({'q':iso_reg.predict(u), 'p':u}).groupby('q').count().reset_index()
            mu_i = (df['p'] * df['q']).sum() / u.shape[0]
            

            mus.append(mu_i)
        
        self.mu_t[np.setdiff1d(range(self.prob.m), top_arms)] = np.array(mus)[np.setdiff1d(range(self.prob.m), top_arms)]
            
        return self.mu_t
    
    def step(self):
        theta_t = self.mu_t#np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        #pred = self.prob.x @ theta_t.transpose()
        
        at = np.argmax(theta_t)
        return at
    
    def run(self, T, M=0):
        self.prob.refresh()
        for t in range(T):
            
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = np.random.randint(self.ss)
                r, _ = self.prob.take_action(action)
                self.S_k[action] = self.S_k[action] + 1
            else:
                action = self.step()
                r, _ = self.prob.take_action(action)
                self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards


class distr_reg_2:
    def __init__(self):
        self.a = ''
        
    def s_in(self, x, i, X, X_bar, sigma_X_inv):
        s = 1 + (X[i] - X_bar) @ sigma_X_inv @ (x - X_bar)     
        
        return s
        
    def predict(self, x_new, x, y, M=100):
        
        N = x.shape[0]
        x_bar = x.mean(0)
        sigma_x = np.cov(x.T)
        sigma_x_inv = np.linalg.inv(sigma_x)
        #s_in_x = s_in(x, i, X, X_bar, sigma_X_inv)
        #u = np.linspace(1e-7, 1-1e-7, M)
        
        
        g = 0
        s = []
        for i in range(N):
            s.append(self.s_in(x_new, i, x, x_bar, sigma_x_inv))
            g = g + self.s_in(x_new, i, x, x_bar, sigma_x_inv)*y[i]
                
        g = g / N
        
        return g, s
    
    

