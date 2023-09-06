#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:42:11 2023

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
        self.T_hist = []
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
        self.T_hist.append(self.T)
        return self.rewards[-1], side_info
    #def get_T(self, action):
    #    return self.df.groupby('action').sum()['T'].to_numpy()
    def regret(self):
        pass
    

class linear_discover(discover):
    def __init__(self, m, d, x, theta, noise=1):
        super(linear_discover, self).__init__(m)
        #self.m = m
        self.x = x
        self.theta = theta
        self.noise = noise
        self.d = d
        self.best_arm = np.argmax(self.x @ self.theta)
        self.sigma = 1
        
        #self.mu_hat = np.zeros(m)
        #self.tau = np.nan
        
    def take_action(self, action):
        self.actions.append(action)
        reward, side_info = self.get_reward(action)
        self.rewards.append(reward)
        self.T[action] = self.T[action] + 1
        self.T_hist.append(self.T.copy())
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
    
    
class Greedy_linear(linear_solver):
    def __init__(self, prob):
        super(Greedy_linear, self).__init__(prob)
        self.init_priors()
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        
        self.i0 = 0
        self.c0 = 0.1
        self.c1 = 0.1
        self.c2 = 0.1
        self.u = 0.1
        
    def _build_A0(self):
        r = 0
        #k = 0
        while r < self.prob.d:
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
                
                rwd, _ = self.prob.take_action(arm)
                self.update_posterior(arm, rwd)
                #self.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                self.A0[r] = arm
                r += 1
                #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        self.c1 = np.min(np.linalg.eigvals(self.A))
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
        
        #f = self.prob.x[action, :]
        #s_inv = np.linalg.inv(self.sigma_t)
        #ffT = np.outer(f, f)
        # pdb.set_trace()
        #mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        #sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        #self.mu_t = mu_
        #self.sigma_t = sigma_
        
        self.temp += r * self.prob.x[action]
        
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
    def step(self):
        #theta_t = self.mu_t
        #pred = self.prob.x @ theta_t.transpose()
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(self.mean_hat)
        return at
    
    def _Z(self, t):
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.prob.x @ self.theta_hat
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
    
    def _stopping_rule_1(self, t, delta):
            s_ = delta * (self.prob.x[:,None] @ self.sigma_hat @ self.prob.x[:,:,None]).flatten()
            
            return ((self.mean_hat - s_)[self.best_arm_hat] > np.delete(self.mean_hat + s_, self.best_arm_hat, 0)).all(), (self.mean_hat - s_)[self.best_arm_hat], np.delete(self.mean_hat + s_, self.best_arm_hat, 0)
             
        
    def _stopping_rule(self, t, delta):
        #logging.debug('Started stopping rule')
        # theoretical threshold
        temp = np.sqrt( np.linalg.det(((1 / (self.u*self.c1)) * self.A) + np.eye(self.prob.d)));
        beta_t = self.c2 * np.log(temp/ delta)
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
        self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        tau = np.nan
            
        #for t in range(M):
        #    action = np.random.choice(self.prob.m)
        #    r, _ = self.prob.take_action(action)
        #    self.update_posterior(action, r)
            #self.best_arm_hat = np.argmax(self.prob.mean_hat)
            
        for t in range(T_max):
            f = self.c0 * np.sqrt(t);
            #print(np.min(np.linalg.eigvals(self.A)), f)
            if True:#np.min(np.linalg.eigvals(self.A)) < f:
                action = self.A0[self.i0]
                self.i0 = np.mod(self.i0 + 1, self.prob.d)
                r, _ = self.prob.take_action(action)
                self.update_posterior(action, r)
                
            else:
                action = self.step()
                r, _ = self.prob.take_action(action)
                self.update_posterior(action, r)
            
            stop,_,_ = self._stopping_rule_1(t, delta)
            
            if stop:
                #print('stop')
                #self.prob.tau = t
                tau = t
                break
            
        
        return self.prob.regret(), self.prob.rewards, self.best_arm_hat, tau
    
    
class lazy_TS(linear_solver):
    def __init__(self, prob, laziness_factor=1, averaging=True):
        super(lazy_TS, self).__init__(prob)
        self.init_priors()
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.g = self.prob.d
        self.laziness_factor = laziness_factor
        self.design = np.zeros(self.prob.m)
        self.design_hist = []
        self.averaging = averaging
        self.cumulative_design = np.zeros(self.prob.m)
        self.support_history = []
        
        self.count_fe = 0
        self.i0 = 0
        self.c0 = 0.1
        self.c1 = 0.1
        self.c2 = 0.1
        self.u = 0.1
        self.psi_inv = 0
        
    def _build_A0(self):
        r = 0
        #k = 0
        while r < self.prob.d:
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
                
                rwd, _ = self.prob.take_action(arm)
                self.update_posterior(arm, rwd)
                #self.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                self.A0[r] = arm
                r += 1
                self.design_hist.append(self.design)
                #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        self.c1 = np.min(np.linalg.eigvals(self.A))
        
    def __optimal_allocation(self, init):

        #design = init
        design = np.ones(self.prob.m)
        design /= design.sum()
        # maximum number of iterations
        max_iter = 1000
        # construct Y
        rewards = self.prob.x @ self.theta_hat
        gaps = np.max(rewards) - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        Y = self.prob.x[self.best_arm_hat, :] - self.prob.x
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1 / gaps) * Y
        _I = np.ones((self.prob.d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.prob.x.T @ np.diag(design) @ self.prob.x)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newY = (Y @ Ainvhalf)**2
            rho = newY @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = Y[idx, :, None]
            g = ((self.prob.x @ A_inv @ y) * (self.prob.x @ A_inv @ y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2 / (count + 2)
            design_update = -gamma * design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

            design += design_update

            #if count % 100 == 0:
            #   print(('design status %s, %s, %s' %(count, relative, np.max(rho))))
            #    logging.debug('design status %s, %s, %s, %s' %
            #                  (self.seed, count, relative, np.max(rho)))

            if relative < 0.01:
            #    logging.debug('design status %s, %s, %s, %s' %
            #                  (self.seed, count, relative, np.max(rho)))
                break

        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        design /= np.sum(design)
        return design, np.max(rho)
    
    def _Lazy_update(self, t):
        if t == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *t), t+1])
            self.design, rho = self.__optimal_allocation(self.design)
            self.psi_inv = 2 *rho;

        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
        
        #f = self.prob.x[action, :]
        #s_inv = np.linalg.inv(self.sigma_t)
        #ffT = np.outer(f, f)
        # pdb.set_trace()
        #mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        #sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        #self.mu_t = mu_
        #self.sigma_t = sigma_
        
        self.temp += r * self.prob.x[action]
        
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
    def step(self):
        #theta_t = self.mu_t
        #pred = self.prob.x @ theta_t.transpose()
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(self.mean_hat)
        return at
    
    def _sampling_rule1(self, t):
        arm = np.random.choice(self.prob.m, 1, p=self.design)[0]
        return arm
        
    def _sampling_rule(self, t):
        f = self.c0 * np.sqrt(t);
        if np.min(np.linalg.eigvals(self.A)) < f:
            arm = self.A0[self.i0]
            self.i0 = np.mod(self.i0 + 1, self.prob.d)
            self.count_fe += 1;
        else:
            if self.averaging:
                # Averaging
                self.cumulative_design += self.design
                self.support = (self.cumulative_design > 0)
                self.map_support = np.array(range(self.prob.m))[self.support]
                index = np.argmin((self.prob.T -
                                   self.cumulative_design)[self.map_support])
            else:
                # Agressive update
                self.support = (self.design > 0)
                self.map_support = np.array(range(self.prob.m))[self.support]
                index = np.argmin(
                    (self.prob.T - t * self.design)[self.map_support])
            arm = self.map_support[index]
            self.support_history.append(np.sum(self.support))
        return arm
    
    def _Z(self, t):
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.prob.x @ self.theta_hat
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
    
    def _stopping_rule_1(self, t, delta):
            s_ = delta * (self.prob.x[:,None] @ self.sigma_hat @ self.prob.x[:,:,None]).flatten()
            
            return ((self.mean_hat - s_)[self.best_arm_hat] > np.delete(self.mean_hat + s_, self.best_arm_hat, 0)).all(), (self.mean_hat - s_)[self.best_arm_hat], np.delete(self.mean_hat + s_, self.best_arm_hat, 0)
             
        
    def _stopping_rule(self, t, delta):
        #logging.debug('Started stopping rule')
        # theoretical threshold
        temp = np.sqrt( np.linalg.det(((1 / (self.u*self.c1)) * self.A) + np.eye(self.prob.d)));
        beta_t = self.c2 * np.log(temp/ delta)
        # heuristic threshold
        #beta_t = self.c2 * (np.log(1/self.delta)+ 0.5*np.log(t) + 0.5*self.d*np.log(1/self.u + 1))
        Z_t = self._Z(t)
        #if t % 1000 == 0:
        #    logging.critical('[round %s] stopping rule:  %s > %s and (t >= d) = %s?' % (str(t), str(Z_t), str(beta_t), str((t >= self.d))));
        self.Z_t.append(Z_t)
        self.beta_t.append(beta_t)
        return (Z_t > beta_t) and (t >= self.prob.d)
        
    def run(self, T_max, delta, sampling_rule=1):
        self.delta = delta
        self.prob.refresh()
        
        #temp = np.zeros(self.prob.d)
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        tau = np.nan
            
        for t in range(T_max):
            if sampling_rule == 1:
                action = self._sampling_rule(t)
            else:
                action = self._sampling_rule1(t)
            #r = self.prob.x[pull] @ self.prob.theta + np.random.normal(1, 1)[0][0]
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
            
            #temp += r * self.prob.x[action]
            #self.allocation[pull] += 1
            #self.A += np.outer(self.X[pull], self.X[pull])
            #self.mu_hat = np.linalg.pinv(self.A) @ temp
            #self.best_arm_hat = np.argmax(self.X @ self.mu_hat)
            
            if self._stopping_rule(t, delta):
                tau = t
                break
            
            self._Lazy_update(t)
            self.design_hist.append(self.design)
        
        return self.prob.regret(), self.prob.rewards, self.best_arm_hat, tau
    
    
class G_allocation(linear_solver):
    def __init__(self, prob, laziness_factor=1, averaging=True):
        super(G_allocation, self).__init__(prob)
        self.init_priors()
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.g = self.prob.d
        self.laziness_factor = laziness_factor
        self.design = np.zeros(self.prob.m)
        self.design_hist = []
        self.averaging = averaging
        self.cumulative_design = np.zeros(self.prob.m)
        self.support_history = []
        
        self.count_fe = 0
        self.i0 = 0
        self.c0 = 0.1
        self.c1 = 0.1
        self.c2 = 0.1
        self.u = 0.1
        self.psi_inv = 0
        
    def _build_A0(self):
        r = 0
        #k = 0
        while r < self.prob.d:
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
                
                rwd, _ = self.prob.take_action(arm)
                self.update_posterior(arm, rwd)
                #self.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                self.A0[r] = arm
                r += 1
                self.design_hist.append(self.design)
                #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        self.c1 = np.min(np.linalg.eigvals(self.A))
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        self.temp += r * self.prob.x[action]
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
    def _sampling_rule(self, t):
        
        outer_x = self.prob.x[:,:,None] @ self.prob.x[:,None]
        x_prime_list = np.zeros(self.prob.m, dtype=int)
        max_list = np.zeros(self.prob.m)
        for i in range(self.prob.m):
            g_ = (self.prob.x[:,None] @ np.linalg.inv(self.A + outer_x[i]) @ self.prob.x[:,:,None]).flatten()
            x_prime_list[i] = int(np.argmax(g_))
            #print(x_prime_list[i])
            #print(g_)
            max_list[i] = g_[x_prime_list[i]]
        
        arm = np.argmin(max_list)
        #arm = np.random.choice(self.prob.m, 1, p=self.design)[0]
        return arm
    
    
    def _stopping_rule(self, t, delta):
        x = self.prob.x
        A_inv = self.sigma_hat
        C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        theta_hat = self.theta_hat
        m = self.prob.m
        
        gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        
        norms = (np.sqrt((x - x[:,None]).reshape(-1, 2)[:,None] @ A_inv @ (x - x[:,None]).reshape(-1, 2)[:,:,None]).flatten()).reshape(m,m)
        
        return (norms * C < gaps).sum(1).max() == m - 1

        
    def run(self, T_max, delta, sampling_rule=1):
        self.delta = delta
        self.prob.refresh()
        
        #temp = np.zeros(self.prob.d)
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta):
                tau = t
                break
            
            self.design_hist.append(self.design)
        
        return self.prob.regret(), self.prob.rewards, self.best_arm_hat, tau
