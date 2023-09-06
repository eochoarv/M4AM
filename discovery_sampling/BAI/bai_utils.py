#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:48:24 2023

@author: eochoa
"""

import pdb
import numpy as np
#from matrix_completion import svt_solve, pmf_solve, calc_unobserved_rmse
from numpy.random import randn
from copy import deepcopy
from weights_utils import *

from numpy.random import randn
from copy import deepcopy
import bayes_logistic as bl
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import polynomial_kernel
from scipy.stats import multivariate_normal
import pandas as pd



expit = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x / (1-x))

class discover:
    def __init__(self, m):
        self.m = m
        self.initi_prob()
        self.actions = [] # labeled set
        self.rewards = [] # previous rewards
        #self.df = pd.DataFrame({'action':[], 'T':[], 'reward':[]})
        self.T = np.zeros(m)
        self.mean_hat = np.zeros(m)
        self.tau = np.nan
        self.best_action = np.nan
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
        #new_row = pd.DataFrame({'action':[action], 'T':[1], 'reward':[reward]})
        #self.df = pd.concat([self.df, new_row])
        self.mean_hat[action] = (self.mean_hat[action] * self.T[action] + reward) / (self.T[action] + 1)
        self.T[action] = self.T[action] + 1
        return self.rewards[-1], side_info
    #def get_T(self, action):
    #    return self.df.groupby('action').sum()['T'].to_numpy()
    def regret(self):
        pass
    

class bernoulli_discover(discover):
    def __init__(self, m, p):
        super(bernoulli_discover, self).__init__(m)
        self.p = p
        #self.tau = np.nan
        #self.p_hat = np.zeros(m)

    def get_reward(self, action):        
        r = np.random.binomial(1, self.p[action])
        return r , None
    
    #def get_mask(self):
    #    mask_mat = np.array([False]*(self.m))
    #    for action in self.actions:
    #        mask_mat[action] = True
    #    self.mask_mat = mask_mat
    #    return mask_mat
    
    def regret(self):
        reg = (np.max(self.p) - self.p[self.actions]).cumsum()
        
        return reg
    
    
class gauss_discover(discover):
    def __init__(self, m, mu, noise=1):
        super(gauss_discover, self).__init__(m)
        #self.m = m
        self.mu = mu
        self.noise = noise
        #self.mu_hat = np.zeros(m)
        #self.tau = np.nan

    def get_reward(self, action):        
        r = np.random.normal(self.mu[action], self.noise, 1)[0]
        return r , None
    
    #def get_mask(self):
    #    mask_mat = np.array([False]*(self.m))
    #    for action in self.actions:
    #        mask_mat[action] = True
    #    self.mask_mat = mask_mat
    #    return mask_mat
    
    def regret(self):
        reg = (np.max(self.mu) - self.mu[self.actions]).cumsum()
        
        return reg

class linear_discover(discover):
    def __init__(self, m, d, x, theta, noise=1):
        super(linear_discover, self).__init__(m)
        #self.m = m
        self.x = x
        self.theta = theta
        self.theta_hat = np.zeros(m)
        self.noise = noise
        self.d = d
        self.best_arm = np.argmax(self.x @ self.theta)
        self.i0 = 0
        self.c0 = 0.1
        self.c1 = 0.1
        self.c2 = 0.1
        self.sigma = 1
        self.u = 0.1
        self.A = np.zeros([self.d, self.d])
        self.A0 = np.zeros(self.d).astype(int)
        
        #self.mu_hat = np.zeros(m)
        #self.tau = np.nan
        
    def take_action(self, action, theta_t=None):
        self.actions.append(action)
        reward, side_info = self.get_reward(action)
        self.rewards.append(reward)
        if theta_t is not None:
            self.theta_hat = theta_t    
            self.mean_hat = self.x @ self.theta_hat.transpose()
        
        self.T[action] = self.T[action] + 1
        
        self.A += np.outer(self.x[action], self.x[action])
        
        #self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.d)
        #self.c1 = np.min(np.linalg.eigvals(self.A))
        #self.c2 = (1+self.u) * (self.sigma**2)
        
        return self.rewards[-1], side_info
    
    def get_reward(self, action):
        f = self.x[action,:] @ self.theta
        r = np.random.normal(f, self.noise, 1)
        return r , None
    
    #def get_mask(self):
    #    mask_mat = np.array([False]*(self.m))
    #    for action in self.actions:
    #        mask_mat[action] = True
    #    self.mask_mat = mask_mat
    #    return mask_mat
    
    
    def regret(self):
        f_star = self.x @ self.theta
        f = self.x[self.actions, :] @ self.theta
        reg = (np.max(f_star) - f).cumsum()
        
        return reg

    
class solver:
    def __init__(self, prob):
        self.prob = prob
    def step(self):
        pass
    def run(self, T_max):
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T_max, replace=True)
        for t in range(T_max):
            self.prob.take_action(acts[t])
        
        best_action = np.argmax(self.prob.mean_hat)
        self.prob.best_action = best_action
        self.tau = T_max
        
        return self.prob.regret(), self.prob.rewards, best_action, T_max
    
class linear_solver:
    def __init__(self, prob):
        self.prob = prob
    def step(self):
        pass
    def run(self, T_max):
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T_max, replace=True)
        for t in range(T_max):
            self.prob.take_action(acts[t])
        return self.prob.regret(), self.prob.rewards
    
class track_n_stop_solver(solver):
    def __init__(self, prob):
        super(track_n_stop_solver, self).__init__(prob)
        self.prob = prob
        self.alpha_star = np.ones(self.prob.m) / self.prob.m
        self.alpha_hist = []
    def step(self, t):
        
        m = self.prob.m
        T = self.prob.T
        mu_hat = self.prob.mean_hat
        i_star = np.argmax(mu_hat)
        if np.min(T) < np.sqrt(t):
            action = np.argmin(T)
        else:
            y_star = solver_F(mu_hat)
            alpha_star = (np.array([x(mu_hat[i_star], mu_hat[a], y_star) for a in range(m)]) / 
                          np.sum([x(mu_hat[i_star], mu_hat[a], y_star) for a in range(m)]))
            self.alpha_star = alpha_star
            action = np.argmax(t*alpha_star - T)
        self.alpha_hist.append(self.alpha_star)
        z = []
        for a in np.setdiff1d(range(m), i_star):
            mu_prime = mu_hat.copy()
            mu_prime[a] = mu_hat.max()
            z.append(sum(T * np.power(mu_prime - mu_hat, 2)) / 2)
        
        stop = min(z) >= self.beta[t]
        
        return action, stop
        
    def run(self, T_max, delta):
        self.prob.refresh()
        m = self.prob.m
        t_array = np.arange(T_max)+1.001
        self.beta = m * np.log(np.log(t_array)) + np.log(1/delta)
        for t in range(m):
            r, _ = self.prob.take_action(t)

        for t in range(m, T_max):
            action, stop = self.step(t)
            r, _ = self.prob.take_action(action)
            
            if stop:
                #print('stop')
                self.prob.tau = t
                break
            
        best_action = np.argmax(self.prob.mean_hat)
        self.prob.best_action = best_action
        #self.tau = T_max
            
        return self.prob.regret(), self.prob.rewards, best_action, self.prob.tau
    
class sample_track_n_stop_solver(solver):
    def __init__(self, prob):
        super(sample_track_n_stop_solver, self).__init__(prob)
        self.prob = prob
        self.alpha_star = np.ones(self.prob.m) / self.prob.m
        self.alpha_hist = []
    def step(self, t):
        
        m = self.prob.m
        T = self.prob.T
        mu_hat = self.prob.mean_hat
        i_star = np.argmax(mu_hat)
        if np.min(T) < np.sqrt(t):
            action = np.argmin(T)
        else:
            y_star = solver_F(mu_hat)
            alpha_star = (np.array([x(mu_hat[i_star], mu_hat[a], y_star) for a in range(m)]) / 
                          np.sum([x(mu_hat[i_star], mu_hat[a], y_star) for a in range(m)]))
            #print(m, alpha_star)
            self.alpha_star = alpha_star
            action = np.random.choice(m, 1, p=alpha_star)[0]
            #action = np.argmax(t*alpha_star - T)
        self.alpha_hist.append(self.alpha_star)
            
        z = []
        for a in np.setdiff1d(range(m), i_star):
            mu_prime = mu_hat.copy()
            mu_prime[a] = mu_hat.max()
            z.append(sum(T * np.power(mu_prime - mu_hat, 2)) / 2)
        
        stop = min(z) >= self.beta[t]
        
        return action, stop
        
    def run(self, T_max, delta):
        self.prob.refresh()
        m = self.prob.m
        t_array = np.arange(T_max)+1.001
        self.beta = m * np.log(np.log(t_array)) + np.log(1/delta)
        for t in range(m):
            r, _ = self.prob.take_action(t)

        for t in range(m, T_max):
            action, stop = self.step(t)
            r, _ = self.prob.take_action(action)
            
            if stop:
                #print('stop')
                self.prob.tau = t
                break
            
        best_action = np.argmax(self.prob.mean_hat)
        self.prob.best_action = best_action
        #self.tau = T_max
            
        return self.prob.regret(), self.prob.rewards, best_action, self.prob.tau

    
class Greedy_linear(linear_solver):
    def __init__(self, prob):
        super(Greedy_linear, self).__init__(prob)
        self.init_priors()
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        
    def _build_A0(self):
        r = 0
        k = 0
        while r < self.prob.d:
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.prob.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
                self.prob.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                self.prob.A0[r] = arm
                r += 1
                #logging.debug('current rank %s' % str(r))
        self.prob.c0 = np.min(np.linalg.eigvals(self.prob.A)) / np.sqrt(self.prob.d)
        self.prob.c1 = np.min(np.linalg.eigvals(self.prob.A))
        
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
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at, theta_t
    
    def _Z(self, t):
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.prob.x @ self.prob.theta_hat
        gaps = rewards[self.best_arm_hat] - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        if np.any(gaps == 0):
            return 0
        Y = self.prob.x[self.best_arm_hat, :] - self.prob.x
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1/gaps)*Y
        _I = np.ones((self.prob.d, 1))
        design = self.prob.T
        A_inv = np.linalg.pinv(self.prob.x.T @ np.diag(design) @ self.prob.x)
        U, D, V = np.linalg.svd(A_inv)
        Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
        newY = (Y @ Ainvhalf)**2
        rho = newY @ _I
        Z = (1/(2*np.max(rho)))
        return Z
    
    def _stopping_rule(self, t, delta):
        #logging.debug('Started stopping rule')
        # theoretical threshold
        temp = np.sqrt( np.linalg.det(((1 / (self.prob.u*self.prob.c1)) * self.prob.A) + np.eye(self.prob.d)));
        beta_t = self.prob.c2 * np.log(temp/ delta)
        # heuristic threshold
        #beta_t = self.c2 * (np.log(1/self.delta)+ 0.5*np.log(t) + 0.5*self.d*np.log(1/self.u + 1))
        Z_t = self._Z(t)
        #if t % 1000 == 0:
        #    logging.critical('[round %s] stopping rule:  %s > %s and (t >= d) = %s?' % (str(t), str(Z_t), str(beta_t), str((t >= self.d))));
        self.Z_t.append(Z_t)
        self.beta_t.append(beta_t)
        return (Z_t > beta_t) and (t >= self.prob.d)
        
    def run(self, T_max, delta, M):
        self.prob.refresh()
        self._build_A0()
        self.prob.c2 = (1+self.prob.u) * (self.prob.sigma**2)
        #for t in range(M):
        #    action = np.random.choice(self.prob.m)
        #    r, _ = self.prob.take_action(action)
        #    self.update_posterior(action, r)
            #self.best_arm_hat = np.argmax(self.prob.mean_hat)
            
        for t in range(T_max):
            f = self.prob.c0 * np.sqrt(t);
            if np.min(np.linalg.eigvals(self.prob.A)) < f:
                action = self.prob.A0[self.prob.i0]
                self.prob.i0 = np.mod(self.prob.i0 + 1, self.prob.d)
                r, _ = self.prob.take_action(action, theta_t)
                self.update_posterior(action, r)
                
            else:
                action, theta_t = self.step()
                r, _ = self.prob.take_action(action, theta_t)
                self.update_posterior(action, r)
                self.best_arm_hat = np.argmax(self.prob.mean_hat)
                stop = self._stopping_rule(t, delta)
            
            if stop:
                #print('stop')
                #self.prob.tau = t
                tau = t
                break
            
        
        return self.prob.regret(), self.prob.rewards, self.best_arm_hat, t
    
    
    
class Greedy_linear(linear_solver):
    def __init__(self, prob):
        super(Greedy_linear, self).__init__(prob)
        self.init_priors()
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        
    def _build_A0(self):
        r = 0
        k = 0
        while r < self.prob.d:
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.prob.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
                self.prob.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                self.prob.A0[r] = arm
                r += 1
                #logging.debug('current rank %s' % str(r))
        self.prob.c0 = np.min(np.linalg.eigvals(self.prob.A)) / np.sqrt(self.prob.d)
        self.prob.c1 = np.min(np.linalg.eigvals(self.prob.A))
        
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
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(pred)
        return at, theta_t
    
    def _Z(self, t):
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.prob.x @ self.prob.theta_hat
        gaps = rewards[self.best_arm_hat] - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        if np.any(gaps == 0):
            return 0
        Y = self.prob.x[self.best_arm_hat, :] - self.prob.x
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1/gaps)*Y
        _I = np.ones((self.prob.d, 1))
        design = self.prob.T
        A_inv = np.linalg.pinv(self.prob.x.T @ np.diag(design) @ self.prob.x)
        U, D, V = np.linalg.svd(A_inv)
        Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
        newY = (Y @ Ainvhalf)**2
        rho = newY @ _I
        Z = (1/(2*np.max(rho)))
        return Z
    
    def _stopping_rule(self, t, delta):
        #logging.debug('Started stopping rule')
        # theoretical threshold
        temp = np.sqrt( np.linalg.det(((1 / (self.prob.u*self.prob.c1)) * self.prob.A) + np.eye(self.prob.d)));
        beta_t = self.prob.c2 * np.log(temp/ delta)
        # heuristic threshold
        #beta_t = self.c2 * (np.log(1/self.delta)+ 0.5*np.log(t) + 0.5*self.d*np.log(1/self.u + 1))
        Z_t = self._Z(t)
        #if t % 1000 == 0:
        #    logging.critical('[round %s] stopping rule:  %s > %s and (t >= d) = %s?' % (str(t), str(Z_t), str(beta_t), str((t >= self.d))));
        self.Z_t.append(Z_t)
        self.beta_t.append(beta_t)
        return (Z_t > beta_t) and (t >= self.prob.d)
        
    def run(self, T_max, delta, M):
        self.prob.refresh()
        self._build_A0()
        self.prob.c2 = (1+self.prob.u) * (self.prob.sigma**2)
        #for t in range(M):
        #    action = np.random.choice(self.prob.m)
        #    r, _ = self.prob.take_action(action)
        #    self.update_posterior(action, r)
            #self.best_arm_hat = np.argmax(self.prob.mean_hat)
            
        for t in range(T_max):
            f = self.prob.c0 * np.sqrt(t);
            if np.min(np.linalg.eigvals(self.prob.A)) < f:
                action = self.prob.A0[self.prob.i0]
                self.prob.i0 = np.mod(self.prob.i0 + 1, self.prob.d)
                r, _ = self.prob.take_action(action, self.prob.theta_hat)
                self.update_posterior(action, r)
                
            else:
                action, theta_t = self.step()
                r, _ = self.prob.take_action(action, theta_t)
                self.update_posterior(action, r)
                self.best_arm_hat = np.argmax(self.prob.mean_hat)
                stop = self._stopping_rule(t, delta)
            
            if stop:
                #print('stop')
                #self.prob.tau = t
                tau = t
                break
            
        
        return self.prob.regret(), self.prob.rewards, self.best_arm_hat, t

