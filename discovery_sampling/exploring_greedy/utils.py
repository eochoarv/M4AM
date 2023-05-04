#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:41:50 2023

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
    

class logistic_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1, theta=None):
        super(logistic_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.noise = noise
        self.sig_coef = sig_coef
        self.init_linear()
        if theta is None:
            self.theta = 5*np.random.rand(d)
        else:
            self.theta = theta
        #self.tau = np.nan
    def init_linear(self):
        self.x = randn(self.m * self.d).reshape(self.m, self.d) * self.sigma
        self.sig_coef = randn(self.d) * self.sig_coef
        self.y_expt = np.apply_along_axis(expit, 0, self.x @ self.sig_coef.transpose())
        self.y = np.array([np.random.choice([0, 1], size = 1, p = [1-p, p])[0] for p in self.y_expt])
    def get_reward(self, action):
        f = self.x[action,:] @ self.theta
        p = expit(f)
        r = np.random.binomial(1,p)
        return r , None
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        f_star = self.x @ self.theta
        p_star = expit(f_star)
        
        f = self.x[self.actions, :] @ self.theta
        p = expit(f)
        
        reg = (np.max(p_star) - p).cumsum()
        
        return reg
    
    def regret_search(self):
        f_star = self.x @ self.theta
        p_star = np.sort(expit(f_star))[::-1]
        
        f = self.x[self.actions, :] @ self.theta
        p = expit(f)
        
        reg = (p_star[:len(self.actions)] - p).cumsum()
        
        return reg
    
class linear_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1, theta=None):
        super(linear_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.noise = noise
        self.sig_coef = sig_coef
        self.init_linear()
        if theta is None:
            self.theta = 5*np.random.rand(d)
        else:
            self.theta = theta
        #self.tau = np.nan
    def init_linear(self):
        self.x = randn(self.m * self.d).reshape(self.m, self.d) * self.sigma
        self.sig_coef = randn(self.d) * self.sig_coef
        self.y_expt = np.apply_along_axis(expit, 0, self.x @ self.sig_coef.transpose())
        self.y = np.array([np.random.choice([0, 1], size = 1, p = [1-p, p])[0] for p in self.y_expt])
    def get_reward(self, action):
        f = self.x[action,:] @ self.theta
        r = np.random.normal(f, self.noise, 1)
        return r , None

    
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        f_star = self.x @ self.theta
        f = self.x[self.actions, :] @ self.theta
        reg = (np.max(f_star) - f).cumsum()
        
        return reg
    
    def regret_search(self):
        f_star = self.x @ self.theta
        f_star = np.sort(f_star)[::-1]
        #p_star = np.sort(expit(f_star))[::-1]
        f = self.x[self.actions, :] @ self.theta
        reg = (f_star[:len(self.actions)] - f).cumsum()
        
        return reg
    
class mixture_discover(discover):
    def __init__(self, m, d, sigma = 1.0, sig_coef = 1.0, noise = 0.1, theta=None):
        super(mixture_discover, self).__init__()
        self.m = m
        self.d = d
        self.sigma = sigma
        self.noise = noise
        self.sig_coef = sig_coef
        self.init_linear()
        if theta is None:
            self.theta = 5*np.random.rand(d)
        else:
            self.theta = theta
        #self.tau = np.nan
    def init_linear(self):
        self.x = randn(self.m * self.d).reshape(self.m, self.d) * self.sigma
        self.sig_coef = randn(self.d) * self.sig_coef
        self.y_expt = np.apply_along_axis(expit, 0, self.x @ self.sig_coef.transpose())
        self.y = np.array([np.random.choice([0, 1], size = 1, p = [1-p, p])[0] for p in self.y_expt])

    def mean_(self, x):
        rv1 = multivariate_normal([.5,.5], .1)
        rv2 = multivariate_normal([-.5,-.5], .1)
        rv3 = multivariate_normal([.5,-.5], .1)
        rv4 = multivariate_normal([-.5,.5], .1)
        
        mu_ = (
            rv1.pdf(x)+
            rv2.pdf(x)+
            rv3.pdf(x)+
            rv4.pdf(x)
            )
        return mu_
    
    def get_reward(self, action):
        f = self.mean_(self.x[action,:])
        r = np.random.normal(f, self.noise, 1)
        return r , None
        
    def get_mask(self):
        mask_mat = np.array([False]*(self.m))
        for action in self.actions:
            mask_mat[action] = True
        self.mask_mat = mask_mat
        return mask_mat
    def regret(self):
        f_star = self.mean_(self.x)
        f = self.mean_(self.x[self.actions, :])
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

class logistic_solver:
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

class Greedy_bernoulli_bayes(logistic_solver):
    def __init__(self, prob, alpha_0 = None, beta_0 = None, k=0):
        super(Greedy_bernoulli_bayes, self).__init__(prob)
        self.init_priors(alpha_0, beta_0)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
    def init_priors(self, alpha_0, beta_0):
        if alpha_0 is None:
            self.alpha_0 = np.ones(self.prob.m)
        else:
            self.alpha_0 = alpha_0
            
        if beta_0 is None:
            self.beta_0 = np.ones(self.prob.m)
        else:
            self.beta_0 = beta_0
            
        self.alpha_t = self.alpha_0
        self.beta_t = self.beta_0
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        self.alpha_t[action] += r
        self.beta_t[action] += 1 - r

        return self.alpha_t, self.beta_t
    
    def step(self):
        
        theta_t = self.alpha_t / (self.alpha_t + self.beta_t)#np.random.beta(alpha, beta)

        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(theta_t)
        return at
    
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = t
            else:
                action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards

class Greedy_bernoulli_freq(logistic_solver):
    def __init__(self, prob, alpha_0 = None, beta_0 = None, k=0):
        super(Greedy_bernoulli_freq, self).__init__(prob)
        self.init_priors(alpha_0, beta_0)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
    def init_priors(self, alpha_0, beta_0):
        if alpha_0 is None:
            self.alpha_0 = np.ones(self.prob.m)
        else:
            self.alpha_0 = alpha_0
            
        if beta_0 is None:
            self.beta_0 = np.ones(self.prob.m)
        else:
            self.beta_0 = beta_0
            
        self.alpha_t = self.alpha_0
        self.beta_t = self.beta_0
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        self.alpha_t[action] += r
        self.beta_t[action] += 1 - r

        return self.alpha_t, self.beta_t
    
    def step(self):
        
        theta_t = self.alpha_t / np.array([max(1,(np.array(self.prob.actions) == i).sum()) for i in range(self.prob.m)])

        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(theta_t)
        return at
    
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = t
            else:
                action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards

class TS_bernoulli(logistic_solver):
    def __init__(self, prob, alpha_0 = None, beta_0 = None, k=0):
        super(TS_bernoulli, self).__init__(prob)
        self.init_priors(alpha_0, beta_0)
        self.k = k
    def init_priors(self, alpha_0, beta_0):
        if alpha_0 is None:
            self.alpha_0 = np.ones(self.prob.m)
        else:
            self.alpha_0 = alpha_0
            
        if beta_0 is None:
            self.beta_0 = np.ones(self.prob.m)
        else:
            self.beta_0 = beta_0
            
        self.alpha_t = self.alpha_0
        self.beta_t = self.beta_0
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        self.alpha_t[action] += r
        self.beta_t[action] += 1 - r

        return self.alpha_t, self.beta_t
    
    def step(self):
        
        theta_t = np.random.beta(self.alpha_t, self.beta_t)

        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(theta_t)
        return at
    
    def run(self, T):
        self.prob.refresh()
        for t in range(T):
            if t < self.k:
                action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
            else:
                action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
class TS_logistic(logistic_solver):
    def __init__(self, prob, s0 = 1.0):
        super(TS_logistic, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
    def init_priors(self):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        mu_, sigma_inv = bl.fit_bayes_logistic(np.array(self.prob.rewards), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        sigma_ = np.linalg.inv(sigma_inv)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        theta_t = np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t.transpose()
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def get_gain(self, action, K):
        r = self.prob.y[action]
        mu_, sigma_ = self.update_posterior(action, r, virtual = True)
        pred = self.prob.x @ mu_.transpose()
        return np.sum(np.sort(pred)[-K:])
    def run(self, T, M=0):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
            if M > 0 and np.sum(self.prob.rewards) >= M and np.isnan(self.prob.tau):
                self.prob.tau = t
                #return self.prob.regret(), self.prob.rewards, self.prob.tau
                break
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
    
class Greedy_logistic(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0):
        super(Greedy_logistic, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
    def init_priors(self):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        mu_, sigma_ = bl.fit_bayes_logistic(np.array(self.prob.rewards), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    
    def step(self):
        theta_t = self.mu_t#np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t.transpose()
        
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    
    def get_gain(self, action, K):
        r = self.prob.y[action]
        mu_, sigma_ = self.update_posterior(action, r, virtual = True)
        pred = self.prob.x @ mu_.transpose()
        return np.sum(np.sort(pred)[-K:])
    def run(self, T, M=0):
        self.prob.refresh()
        for t in range(T):
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = t
            else:
                action = self.step()
                
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
class Greedy_logistic_search(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0):
        super(Greedy_logistic_search, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k
    def init_priors(self):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        mu_, sigma_ = bl.fit_bayes_logistic(np.array(self.prob.rewards), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    
    def step(self):
        theta_t = self.mu_t#np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t
        
        self.prob.get_mask()
        pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    
    def get_gain(self, action, K):
        r = self.prob.y[action]
        mu_, sigma_ = self.update_posterior(action, r, virtual = True)
        pred = self.prob.x @ mu_.transpose()
        return np.sum(np.sort(pred)[-K:])
    def run(self, T, M=0):
        self.prob.refresh()
        for t in range(min(T, self.prob.m)):
            if t < self.k:
                #action = np.random.choice(range(self.prob.m), 1, replace=True)[0]
                action = t
            else:
                action = self.step()
                
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret_search(), self.prob.rewards

class TS_logistic_search(logistic_solver):
    def __init__(self, prob, s0 = 1.0):
        super(TS_logistic_search, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
    def init_priors(self):
        self.mu_0 = np.zeros(self.prob.d)
        self.sigma_0 = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.mu_t = np.zeros(self.prob.d)
        self.sigma_t = self.s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        mu_, sigma_inv = bl.fit_bayes_logistic(np.array(self.prob.rewards), self.prob.x[self.prob.actions, :], self.mu_0, self.sigma_0)
        sigma_ = np.linalg.inv(sigma_inv)
        self.mu_t = mu_
        self.sigma_t = sigma_
        return mu_, sigma_
    def step(self):
        theta_t = np.random.multivariate_normal(self.mu_t, self.sigma_t, 1)
        pred = self.prob.x @ theta_t.transpose()
        self.prob.get_mask()
        pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at
    def get_gain(self, action, K):
        r = self.prob.y[action]
        mu_, sigma_ = self.update_posterior(action, r, virtual = True)
        pred = self.prob.x @ mu_.transpose()
        return np.sum(np.sort(pred)[-K:])
    def run(self, T, M=0):
        self.prob.refresh()
        for t in range(T):
            action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
            if M > 0 and np.sum(self.prob.rewards) >= M and np.isnan(self.prob.tau):
                self.prob.tau = t
                #return self.prob.regret(), self.prob.rewards, self.prob.tau
                break
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards


class Greedy_test_1(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0, kernel_dim=0):
        super(Greedy_test_1, self).__init__(prob)
        self.s0 = s0
        
        if kernel_dim == 0:
            self.kernel_dim = self.prob.d
        else:
            self.kernel_dim = kernel_dim
            
        self.init_priors()
        # number of pulls for each arms
        self.S_k = np.zeros(self.prob.m)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k

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
                
            
        self.scores_t[action] = logit(np.clip(self.mu_t[action], 1e-7, 1-1e-7))
        
        mu_a = np.clip(self.mu_t[action],1e-7,1-1e-7)
        #self.scores_t[action] = (n / (n + 1) * self.scores_t[action] + 
        #                         1 / (n + 1) * 1 / (mu_a*(1-mu_a)))
    
        self.S_k[action] = n + 1
            
        #select top selected d arms
        top_arms = np.argsort(self.S_k)[::-1][:self.kernel_dim]
            
        s = self.scores_t[top_arms]
        A = self.prob.x[top_arms].T
        A_inv = np.linalg.inv(A)
        x_ = A_inv @ self.prob.x.T
        #return x_, top_arms
        self.scores_t = (s @ x_).flatten()
        g_ = expit(np.clip(self.scores_t, -1e7, 1e7))
        grad = g_ * (1-g_) / (n+1)
        
        self.mu_t[np.setdiff1d(range(self.prob.m), top_arms)] = (n / (n + 1) * self.mu_t[np.setdiff1d(range(self.prob.m), top_arms)]+
                                                                 grad[np.setdiff1d(range(self.prob.m), top_arms)])
            
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
                action = t
            else:
                action = self.step()
                
            #action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
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
        epsilon = self.epsilon 
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

    
class Greedy_test_2(linear_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0):
        super(Greedy_test_2, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
        # number of pulls for each arms
        self.S_k = np.zeros(self.prob.m)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k

    def init_priors(self):
        self.scores_t = np.zeros(self.prob.m)
        self.mu_t = self.scores_t
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        n = self.S_k[action]
            
        if n == 0:
            self.mu_t[action] = r
        else:
            self.mu_t[action] = (n * self.mu_t[action] + r) / (n + 1)
                
            
        self.scores_t[action] = self.mu_t[action]
        
        
        self.mu_t[action]
    
        self.S_k[action] = n + 1
            
        #select top selected d arms
        top_arms = np.argsort(self.S_k)[::-1][:self.prob.d]
            
        s = self.scores_t[top_arms]
        A = self.prob.x[top_arms].T
        A_inv = np.linalg.inv(A)
        x_ = A_inv @ self.prob.x.T
        #return x_, top_arms
        self.scores_t = (s @ x_).flatten()
        self.mu_t = self.scores_t
            
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
                action = t
            else:
                action = self.step()
                
            #action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
    
class Greedy_test_linear_kernel(linear_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0, kernel='rgf', sig_kernel=1, degree=3, coef0=0, kernel_dim=0):
        super(Greedy_test_linear_kernel, self).__init__(prob)
        self.s0 = s0
        self.init_priors()
        # number of pulls for each arms
        if kernel_dim == 0:
            self.kernel_dim = self.prob.d
        else:
            self.kernel_dim = kernel_dim
        self.S_k = np.zeros(self.prob.m)
        if kernel == 'rgf':
            self.kernel = 1.0 * RBF(sig_kernel)
            self.K_x = self.kernel(self.prob.x)
        else:
            self.kernel = polynomial_kernel
            self.K_x = self.kernel(self.prob.x, degree=degree, coef0=coef0)
        if k == 0:
            self.k = self.prob.m
        else:
            self.k = k

    def init_priors(self):
        self.scores_t = np.zeros(self.prob.m)
        self.mu_t = self.scores_t
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        n = self.S_k[action]
            
        if n == 0:
            self.mu_t[action] = r
        else:
            self.mu_t[action] = (n * self.mu_t[action] + r) / (n + 1)
                
            
        self.scores_t[action] = self.mu_t[action]
    
        self.S_k[action] = n + 1
            
        #select top selected d arms
        top_arms = np.argsort(self.S_k)[::-1][:self.kernel_dim]
        
        #x = self.prob.x
        #idx_min = np.argmin(np.abs(x @ x.T) / (np.linalg.norm(x, axis=1) * np.linalg.norm(x, axis=1)[:,None]))
        #top_arms = [idx_min % self.prob.m, idx_min // self.prob.m]
            
        s = self.scores_t[top_arms]
        
        # check to select orthogonal
        #A = self.prob.x[top_arms].T
        #A_inv = np.linalg.inv(A)
        #x_ = A_inv @ self.prob.x.T
        #x_ = self.K_x[top_arms]
        
        K_x_star = self.K_x[:,top_arms]
        x_ = np.linalg.inv(K_x_star.T @ K_x_star) @ K_x_star.T @ self.K_x
        
        #self.scores_t[np.setdiff1d(range(self.prob.m), top_arms)] = (s @ x_).flatten()[np.setdiff1d(range(self.prob.m), top_arms)]
        self.scores_t = (s @ x_).flatten()
        
        self.mu_t = self.scores_t
            
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
                action = t
            else:
                action = self.step()
                
            #action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
from scipy.stats import norm, bernoulli, binom
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
    
    
class Greedy_Frechet(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, k=0, dim=0):
        super(Greedy_Frechet, self).__init__(prob)
        self.s0 = s0
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

    def init_priors(self):
        self.scores_t = np.zeros(self.prob.m)
        self.mu_t = self.scores_t
        
    def update_posterior(self, action, r):
        # should we use all the previous samples?
        # pdb.set_trace()
        
        n = self.S_k[action]
            
        if n == 0:
            self.mu_t[action] = r
        else:
            self.mu_t[action] = (n * self.mu_t[action] + r) / (n + 1)
                
            
        self.scores_t[action] = self.mu_t[action]
        
        
        self.mu_t[action]
    
        self.S_k[action] = n + 1
            
        #select top selected d arms
        top_arms = np.argsort(self.S_k)[::-1][:self.dim]
            
        dist = [norm(mu) for mu in self.mu_t[top_arms]]
        u = np.linspace(1e-7, 1-1e-7, 1000)
        x = self.prob.x
        dr = distr_reg()
        mus = []
        for i in range(self.prob.m):
            g, _ = dr.predict(x_new=x[i], x=x[top_arms], y=dist,M=1000)
            
            iso_reg = IsotonicRegression().fit(u, g)
            mu_i = iso_reg.predict(u)[500]
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
                action = t
            else:
                action = self.step()
                
            #action = self.step()
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                
        # pdb.set_trace()
        return self.prob.regret(), self.prob.rewards
    
    
class Greedy_logistic_Frechet(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, ss=0, k=0, dim=0):
        super(Greedy_logistic_Frechet, self).__init__(prob)
        self.s0 = s0
        
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
            
        
        dist = [bernoulli(p) for p in self.mu_t[top_arms]]
        u = np.linspace(1e-7, 1-1e-7, 1000)
        x = self.prob.x
        dr = distr_reg()
        mus = []
        for i in range(self.prob.m):
            g, _ = dr.predict(x_new=x[i], x=x[top_arms], y=dist,M=1000)
            
            iso_reg = IsotonicRegression().fit(u, g)
            mu_i = np.sum(np.diff(iso_reg.predict(u),1) * (1-u)[1:])
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


class Greedy_logistic_Frechet_2(logistic_solver):
    def __init__(self, prob, s0 = 1.0, r = 0, ss=0, k=0, dim=0, supp=None):
        super(Greedy_logistic_Frechet_2, self).__init__(prob)
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
        dr = distr_reg_3()
        dr.init_kernel()
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
    
class distr_reg:
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
        u = np.linspace(1e-7, 1-1e-7, M)
        
        
        g = 0
        s = []
        for i in range(N):
            s.append(self.s_in(x_new, i, x, x_bar, sigma_x_inv))
            g = g + self.s_in(x_new, i, x, x_bar, sigma_x_inv)*y[i].ppf(u)
                
        g = g / N
        
        return g, s
    
    
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
    
    
class distr_reg_3:
    def __init__(self):
        self.a = ''
        
    def init_kernel(self, ):
        self.kernel = RBF(1)
        
    def s_in(self, x, i, X, X_bar, sigma_X_inv):
        
        #s = 1 + (X[i] - X_bar) @ sigma_X_inv @ (x - X_bar)     
        s = 1 + (self.kernel(x, X) @ np.linalg.inv(self.kernel(X) + np.eye(X.shape[0])))[0, i]
        
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