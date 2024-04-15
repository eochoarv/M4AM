#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 00:17:12 2023

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
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
import pandas as pd
import itertools
import random
from collections import defaultdict


expit = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x / (1-x))

class AdaHedge:
    def __init__(self, K):
        self.L = np.zeros(K)
        self.Delta = 0.01

    def act(self):
        eta = np.log(len(self.L)) / self.Delta
        u = np.exp(-eta * (self.L - np.min(self.L)))
        return u / np.sum(u)
    
    def incur(self, l):
        u = self.act()
        eta = np.log(len(self.L)) / self.Delta
        self.L += l
        m = np.min(l) - (1 / eta) * np.log(np.dot(u, np.exp(-eta * (l - np.min(l)))))
    
        #assert np.isfinite(m) and np.dot(u, l) >= m - 1e-7, f"hedge loss {np.dot(u, l)}, Δ={h.Δ}, η={η}, mix loss {m}, u={u}, L={h.L}"
        self.Delta += np.dot(u, l) - m
        
def randargmin(b,**kw):
    idxs = np.where(np.abs(b-np.nanmin(b))<1e-20)[0]    
    return np.random.choice(idxs)

def optimal_allocation(x, theta, rho, epsilon, init=None, max_iter = 1000):

        #design = init
        m, d = x.shape
        design = np.ones(m)
        design /= design.sum()
        
        # maximum number of iterations
        
        # construct Y
        rewards = x @ theta
        gaps = np.abs(rewards - rho) + epsilon
        #bad_arms = np.where(gaps <= 0)[0]
        amb_arms = np.where(gaps == 0)[0]
        good_arms = np.where(gaps > 0)[0]
        gaps = np.delete(gaps, amb_arms, 0)
        #gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        X = np.delete(x, amb_arms, 0)
        #X = x[good_arms, :]
        X = (1 / gaps) * X
        _I = np.ones((d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(x.T @ np.diag(design) @ x)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newX = (X @ Ainvhalf)**2
            phi = newX @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(phi)
            X_ = X[idx, :, None]
            g = ((x @ A_inv @ X_) * (x @ A_inv @ X_)).flatten()
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
        return design, np.max(phi)

def optimal_allocation_G(x, theta, rho, epsilon, init=None, max_iter = 1000):

        #design = init
        m, d = x.shape
        design = np.ones(m)
        design /= design.sum()
        
        _I = np.ones((d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(x.T @ np.diag(design) @ x)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newX = (x @ Ainvhalf)**2
            phi = newX @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(phi)
            X_ = x[idx, :, None]
            g = ((x @ A_inv @ X_) * (x @ A_inv @ X_)).flatten()
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
        return design, np.max(phi)

def optimal_allocation_XY(x, y, theta, rho, epsilon, init=None, max_iter = 1000):

        #design = init
        m, d = x.shape
        design = np.ones(m)
        design /= design.sum()
        
        # maximum number of iterations
        
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(x.T @ np.diag(design) @ x)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newY = (y @ Ainvhalf)**2
            phi = newY @ np.ones((newY.shape[1], 1))

            idx = np.argmax(phi)
            Y_ = y[idx, :, None]
            g = ((x @ A_inv @ Y_) * (x @ A_inv @ Y_)).flatten()
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
        
        return design, np.max(phi)


def gen_data(N=100, p=2, scale=1):

    theta = (2*np.random.rand(p)-1)
    theta = theta / np.linalg.norm(theta)
    theta = scale * theta
    r = np.random.rand(N).reshape(N, 1)-1
    x = np.random.rand(N,p).reshape(N, p)
    x = x - x.mean(0)
    x = r*normalize(x, axis=1, norm='l2')
    
    #x = np.random.rand(p*N).reshape(N, p)
    
    x_ = np.concatenate([np.ones((N,1)), x],axis=1)
    f = x @ theta
    #f = np.power(f,2)
    p = expit(f)
    y = np.random.binomial(1,p)
    return x, y, theta

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
        #self.rho = rho
        #self.good_arms = np.where(self.x @ self.theta > rho)
        self.sigma = 1
        
        #self.mu_hat = np.zeros(m)
        #self.tau = np.nan
      
        Y = np.zeros((m*m, self.d))
        for i in range(m):
            Y[m*i:m*(i+1),:] = self.x - self.x[i,:] 
          
        self.y_hat = Y 
        
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
    def __init__(self, prob, rho, epsilon):
        self.epsilon = epsilon
        self.prob = prob
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
    def _build_A0(self):
        r = 0
        seed = 0
        #random.seed(seed)
        while r < self.prob.d:
            np.random.seed(seed)
            arm = np.random.randint(self.prob.m)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.prob.x[arm], self.prob.x[arm])) > r:
            
                rwd, _ = self.prob.take_action(arm)
                self.update_posterior(arm, rwd)
                self.A0[r] = arm
                r += 1
            seed = seed + 1
                #self.design_hist.append(self.design)
                #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        #self.c1 = np.min(np.linalg.eigvals(self.A))
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)
        
    def update_posterior(self, action, r):
        
        self.temp += r * self.prob.x[action]     
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.A_inv = self.sigma_hat
        
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
    def step(self):
        pass
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        d = self.prob.d
        c_ = 10#2*np.log(1/delta)*np.sqrt(np.log(1/delta)*(np.log(1/delta) + d))
        
        #C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        #C = np.sqrt(d * np.log(t / (d * delta * 1)) + c_delta/1)
        C = np.sqrt(2 *np.log(1/delta) + d * np.log(t / 1) + c_/1)
        theta_hat = self.theta_hat
        m = self.prob.m
        gaps = np.abs(x @ theta_hat - self.rho) + epsilon   
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())

        return C < np.min(gaps/norms)
    
    def performance_e(self):
        epsilon = self.epsilon
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return performance
    
    def performance(self):
        epsilon = self.epsilon
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] > self.rho).sum())
        
        return performance
            
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        np.random.seed(seed)
        self.prob.refresh()
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        #self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        acts = np.random.choice(range(self.prob.m), T_max, replace=True)
        np.random.seed(seed)
        tau = np.nan
        for t in range(self.prob.d, T_max):
            r, _ = self.prob.take_action(acts[t])
            self.update_posterior(acts[t], r)
        #return self.prob.regret(), self.prob.rewards
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
            
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau


class Greedy_linear(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(Greedy_linear, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
    
    def _sampling_rule(self, t):
        at = np.nanargmax(self.mean_hat)
        return at
        
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau


class lazy_TS(linear_solver):
    def __init__(self, prob, rho, epsilon, laziness_factor=1, averaging=True):
        
        super(lazy_TS, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []

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
        #self.c1 = 0.1
        #self.c2 = 0.1
        #self.u = 0.1
        self.psi_inv = 0
    
    def _Lazy_update(self, t, epsilon):
        if t == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *t), t+1])
            #self.design, rho = self.__optimal_allocation(self.design, epsilon)
            x = self.prob.x
            theta = self.theta_hat
            self.design, rho = optimal_allocation(x, theta, self.rho, epsilon)                                       
            self.psi_inv = 2 *rho;        
        
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
             
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        
        #temp = np.zeros(self.prob.d)
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        #self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)

            
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
            
            self._Lazy_update(t, epsilon)
            self.design_hist.append(self.design)
            
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau


class G_allocation(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(G_allocation, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
    
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

        
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau

class Oracle_linear(linear_solver):
    def __init__(self, prob, rho, epsilon, opt_desing):
        super(Oracle_linear, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.opt_desing = opt_desing
        self.i0 = 0
        self.count_fe = 0
    
    def _sampling_rule(self, t):
        f = self.c0 * np.sqrt(t);
        if np.min(np.linalg.eigvals(self.A)) < f:
            arm = self.A0[self.i0]
            self.i0 = np.mod(self.i0 + 1, self.prob.d)
            self.count_fe += 1;
        else:
            self.support = (self.opt_desing > 0)
            self.map_support = np.array(range(self.prob.m))[self.support]
            index = np.argmin((self.prob.T - t * self.opt_desing)[self.map_support])
                
            arm = self.map_support[index]
        return arm
    
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau

class G_allocation_phase(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(G_allocation_phase, self).__init__(prob, rho, epsilon)
        self.init_priors()
       #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
    
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

        
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        x = self.prob.x
        design, _ = optimal_allocation_G(x, self.theta_hat, self.rho, self.epsilon)
        u = 1.35
        np.random.seed(seed)
        tau = np.nan
        phase_index = 1
        t = self.prob.d
        continue_ = True
        
        while continue_:
            #self.delta_t = self.delta/(2*self.phase_index**2*self.K_Z**2)
            num_samples = int(np.ceil(u ** phase_index))
            allocation = np.random.choice(self.prob.m, num_samples, p=design).tolist()
            #allocation = np.array([allocation.count(i) for i in range(self.prob.m)])
            #pulls = np.vstack([np.tile(x[i], (num, 1)) for i, num in enumerate(allocation) if num > 0])
            for action in allocation:
                #action = self._sampling_rule(t)
                r, _ = self.prob.take_action(action)
                self.update_posterior(action, r)
                if self._stopping_rule(t, delta, epsilon) or t >= T_max:
                    tau = t
                    continue_ = False
                    break
                
                t = t + 1
                
            phase_index = phase_index + 1
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau


class xy_adaptive(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(xy_adaptive, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.active_arms = list(range(self.prob.m))
        self.build_Y()
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
    def build_Y(self):
        
        Y = []
        x_ = np.zeros_like(self.prob.x) * np.nan
        x_[self.active_arms] = self.prob.x[self.active_arms]
        for x, x_prime in itertools.permutations(x_, 2):
            Y.append(x - x_prime)
            
        self.Yhat = np.array(Y)
        self.prob.x_hat = x_
        
    def drop_arms(self):
                
        active_arms = self.active_arms.copy()
        
        for arm_idx in active_arms:
            
            arm = self.prob.x[arm_idx, :, None]
            
            for arm_idx_prime in active_arms:
                
                if arm_idx == arm_idx_prime:
                    continue
                    
                arm_prime = self.prob.x[arm_idx_prime, :, None]
                y = arm_prime - arm

                if np.sqrt(2*y.T@self.A_inv@y*np.log(2*np.pi**2*self.prob.m**2*self.phase_index**2/(6*self.delta))) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
                
    def _sampling_rule(self, t):
        
        score = np.diag(self.Yhat@self.A_inv@self.Yhat.T).reshape(1, -1) \
                - ((self.prob.x@self.A_inv@self.Yhat.T)**2/(1+ np.diag(self.prob.x @ self.A_inv @ self.prob.x.T).reshape(-1, 1)))
            
        arm_idx = randargmin(score[np.arange(self.prob.m), np.nanargmax(score, axis=1)])
        
        rho = np.nanmax(score[arm_idx,:])
        
        return arm_idx, rho

    
    def run(self, T_max, alpha, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        n_past = self.prob.d*(self.prob.d + 1) + 1
        rho = 1
        rho_past = 1
        self.phase_index = 1
        tau = np.nan
        #t = 0
        self.N = 0
        continue_ = True
        t = 0
        while (len(self.active_arms) > 1) and continue_:
            
            #self._build_A0()
            self.A = np.eye(self.prob.d)
            self.A_inv = np.eye(self.prob.d)
            self.temp = np.zeros(self.prob.d)
            n = self.prob.d
            #t = t + self.prob.d
            self.N = self.N + self.prob.d
            
                
            while rho/n >= alpha*rho_past/n_past:
                    
                arm_idx, rho = self._sampling_rule(t)
                r, _ = self.prob.take_action(arm_idx)
                self.update_posterior(arm_idx, r)
                t = t + 1
                n = n + 1
                self.N = self.N + 1   
                if t >= T_max:
                    #tau = t
                    continue_ = False
                    #return t
                    break
                    
                
                
            n_past = n
            rho_past = rho
            
            self.drop_arms()
            self.build_Y()
            self.phase_index += 1
            
        tau = t
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau

class LinGapE(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(LinGapE, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.eye(self.prob.d)#np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
        
    def confidence_bound(self, x, t):
        L = 1
        sigma = 1
        reg = 1
        d = self.prob.d
        m = self.prob.m
        tmp = np.sqrt(x.dot(self.A_inv).dot(x))
        return tmp * (sigma * np.sqrt(d * np.log(m * m * (1 + t * L * L) / reg / self.delta)) + np.sqrt(reg) * 2)

    def _sampling_rule(self, t):
        
        
        #it = np.argmax(self.mean_hat)
        #jt = np.argmax(self.mean_hat - self.mean_hat[it] +
        #              np.array([self.confidence_bound(x - self.prob.x[it], t) for x in self.prob.x]))
        jt = np.argmax(-np.abs(self.mean_hat - self.rho) +
                       np.array([self.confidence_bound(x, t) for x in self.prob.x]))
        #B = self.mean_hat[jt] - self.mean_hat[it] + self.confidence_bound(self.prob.x[it] - self.prob.x[jt], t)
        B = -np.abs(self.mean_hat[jt] - self.rho) + self.confidence_bound(self.prob.x[jt], t)
        
        y = self.prob.x[jt]
        tmp = [y.dot(np.linalg.inv(self.A + np.outer(x, x))).dot(y) for x in self.prob.x]
        # print tmp
        return np.argmin(tmp)
        
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        #for t in range(self.prob.m):
        #    action = t
        #    r, _ = self.prob.take_action(action)
        #    self.update_posterior(action, r)
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau


class rage(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(rage, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.active_arms = list(range(self.prob.m))
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.factor = 5
        
        
    def optimal_allocation(self):
        
        design = np.ones(self.prob.m)
        design /= design.sum()  
        
        max_iter = 5000
        
        x_hat = self.prob.x[self.active_arms, :]
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.prob.x.T@ np.diag(design) @ self.prob.x)    
            U,D,V = np.linalg.svd(A_inv)
            Ainvhalf = U@np.diag(np.sqrt(D))@V.T
            
            newY = (x_hat@Ainvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))
                        
            idx = np.argmax(rho)
            y = x_hat[idx, :, None]
            g = ((self.prob.x @A_inv@ y )*(self.prob.x @A_inv@y)).flatten()
            g_idx = np.argmax(g)
                        
            gamma = 2/(count+2)
            design_update = -gamma*design
            design_update[g_idx] += gamma
                
            relative = np.linalg.norm(design_update)/(np.linalg.norm(design))
                        
            design += design_update
            
            #if count % 100 == 0:
            #    logging.debug('design status %s, %s, %s, %s' % (self.seed, count, relative, np.max(rho)))
                            
            if relative < 0.01:
                 break
                        
        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        
        return design, np.max(rho)
        
    
    def rounding(self, design, num_samples):
        
        num_support = (design > 0).sum()
        support_idx = np.where(design>0)[0]
        support = design[support_idx]
        n_round = np.ceil((num_samples - .5*num_support)*support)

        while n_round.sum()-num_samples != 0:
            if n_round.sum() < num_samples:
                idx = np.argmin(n_round/support)
                n_round[idx] += 1
            else:
                idx = np.argmax((n_round-1)/support)
                n_round[idx] -= 1

        allocation = np.zeros(len(design))
        allocation[support_idx] = n_round
            
        return allocation.astype(int)
    
    def drop_arms(self):
                
        active_arms = self.active_arms.copy()
        removes = set()
        scores = np.abs(self.prob.x[active_arms,:] @ self.theta_hat - self.rho) + self.epsilon
        # gap = 2**(-(self.phase_index+2))
        gap = 2**(-(self.phase_index))

        for i,s in enumerate(scores):
            if gap <= s:
                #arm_idx = int(self.idxs[t][1])
                removes.add(self.active_arms[i])

        for r in removes:
            self.active_arms.remove(r)

    
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.phase_index = 1
        self.delta = delta
        self.delta_t = delta
        self.prob.refresh()
        #n_past = self.prob.d*(self.prob.d + 1) + 1
        #rho = 1
        #rho_past = 1
        tau = np.nan
        self.N = 0
        continue_ = True
        t = 0
        np.random.seed(seed)
        
        while (len(self.active_arms) > 1) and continue_:
            
            self.delta_t = delta / (self.phase_index**2)
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            n_min = 2*self.factor*support
            eps = 1/self.factor #approx factor
            num_samples = max(np.ceil(8*(2**(self.phase_index-1))**2*rho*(1+eps)*np.log(2*self.prob.m**2/self.delta_t)), n_min).astype(int)
            allocation = self.rounding(design, num_samples)
            
            for a, n_samples in enumerate(allocation):
                if n_samples > 0:
                    for j in range(n_samples):
                        r, _ = self.prob.take_action(a)
                        self.update_posterior(a, r)
                        t = t + 1
                        self.N = self.N + 1   
                        if t >= T_max:
                            #tau = t
                            continue_ = False
                            #return t
                            break
                        if self._stopping_rule(t, delta, epsilon):
                            tau = t
                            break
            
            self.drop_arms()
            self.phase_index += 1
            
        tau = t
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau
    
class x_adaptive(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(x_adaptive, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.active_arms = list(range(self.prob.m))
        self.build_X()
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
    def build_X(self):
        
        Y = []
        x_ = np.zeros_like(self.prob.x) * np.nan
        x_[self.active_arms] = self.prob.x[self.active_arms]
        #for x, x_prime in itertools.permutations(x_, 2):
        #    Y.append(x - x_prime)
            
        #self.Yhat = np.array(Y)
        self.Xhat = x_
        
    def drop_arms(self):
                
        active_arms = self.active_arms.copy()
        
        for arm_idx in active_arms:
            
            arm = self.prob.x[arm_idx, :, None]

            if np.sqrt(2 * arm.T @ self.A_inv @ arm *np.log(2*np.pi**2*self.prob.m**2*self.phase_index**2/(6*self.delta))) <= abs(arm.T@self.theta_hat-self.rho):
                self.active_arms.remove(arm_idx)
                break
                
    def _sampling_rule(self, t):
        
        score = np.diag(self.Xhat@self.A_inv@self.Xhat.T).reshape(1, -1) \
                - ((self.prob.x@self.A_inv@self.Xhat.T)**2/(1+ np.diag(self.prob.x @ self.A_inv @ self.prob.x.T).reshape(-1, 1)))
            
        arm_idx = randargmin(score[np.arange(self.prob.m), np.nanargmax(score, axis=1)])
        
        rho = np.nanmax(score[arm_idx,:])
        
        return arm_idx, rho

    
    def run(self, T_max, alpha, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        n_past = self.prob.d*(self.prob.d + 1) + 1
        rho = 1
        rho_past = 1
        self.phase_index = 1
        tau = np.nan
        #t = 0
        self.N = 0
        continue_ = True
        t = 0
        while (len(self.active_arms) > 1) and continue_:
            
            #self._build_A0()
            self.A = np.eye(self.prob.d)
            self.A_inv = np.eye(self.prob.d)
            self.temp = np.zeros(self.prob.d)
            n = self.prob.d
            #t = t + self.prob.d
            self.N = self.N + self.prob.d
            
                
            while rho/n >= alpha*rho_past/n_past:
                    
                arm_idx, rho = self._sampling_rule(t)
                r, _ = self.prob.take_action(arm_idx)
                self.update_posterior(arm_idx, r)
                t = t + 1
                n = n + 1
                self.N = self.N + 1   
                if t >= T_max:
                    #tau = t
                    continue_ = False
                    #return t
                    break
                    
                
                
            n_past = n
            rho_past = rho
            
            self.drop_arms()
            self.build_X()
            self.phase_index += 1
            
        tau = t
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau



class xy_adaptive1(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(xy_adaptive1, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.d)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.active_arms = list(range(self.prob.m))
        self.build_Y()
        
        
        
    def _build_A0(self):
        self.N += self.prob.d
        self.A = np.eye(self.prob.d)
        self.A_inv = np.linalg.inv(self.A)
        self.b = np.zeros((self.prob.d, 1))
        
    def build_Y(self):
        
        Y = []
        for x, x_prime in itertools.permutations(self.prob.x[self.active_arms], 2):
            Y.append(x - x_prime)
            
        self.Yhat = np.array(Y)
    
    def drop_arms(self):
                
        active_arms = self.active_arms.copy()
        
        for arm_idx in active_arms:
            
            arm = self.prob.x[arm_idx, :, None]

            if np.sqrt(2 * arm.T @ self.A_inv @ arm *np.log(2*np.pi**2*self.prob.m**2*self.phase_index**2/(6*self.delta))) <= abs(arm.T@self.theta_hat-self.rho):
                self.active_arms.remove(arm_idx)
                break
            
    def drop_arms1(self):
                
        active_arms = self.active_arms.copy()
        
        for arm_idx in active_arms:
            
            arm = self.prob.x[arm_idx, :, None]
            
            for arm_idx_prime in active_arms:
                
                if arm_idx == arm_idx_prime:
                    continue
                    
                arm_prime = self.prob.x[arm_idx_prime, :, None]
                y = arm_prime - arm

                if np.sqrt(2*y.T@self.A_inv@y*np.log(2*np.pi**2*self.prob.m**2*self.phase_index**2/(6*self.delta))) <= y.T@self.theta_hat:
                    self.active_arms.remove(arm_idx)
                    break
    
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
    
    def select_greedy_arm(self):
        
        score = np.diag(self.Yhat@self.A_inv@self.Yhat.T).reshape(1, -1) \
                - ((self.prob.x@self.A_inv@self.Yhat.T)**2/(1+ np.diag(self.prob.x @ self.A_inv @ self.prob.x.T).reshape(-1, 1)))
            
        arm_idx = randargmin(score[np.arange(self.prob.m), np.argmax(score, axis=1)])
        
        rho = np.max(score[arm_idx,:])
        
        return arm_idx, rho
        
    def run(self, T_max, alpha, delta):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        n_past = self.prob.d*(self.prob.d + 1) + 1
        rho = 1
        rho_past = 1
        self.phase_index = 1
        tau = np.nan
        t = 0
        self.N = 0
        while len(self.active_arms) > 1:
            
            self._build_A0()
            n = self.prob.d
            
                
            while rho/n >= alpha*rho_past/n_past:
                    
                arm_idx, rho = self.select_greedy_arm()
                arm = self.prob.x[arm_idx, :, None]
                r, _ = self.prob.take_action(arm_idx)
                self.update_posterior(arm_idx, r)
                    
                n += 1
                self.N += 1   
                t += 1
                
            n_past = n
            rho_past = rho
            
            self.drop_arms()
            self.build_Y()
            self.phase_index += 1
            
        tau = t
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau

    def run_1(self, T_max, delta):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        self._build_A0()
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau
    
    
    
class LinGame(linear_solver):
    def __init__(self, prob, rho, epsilon, averaging=True):
        
        super(LinGame, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.best_arm_hat = 0
        self.beta_t = []
        self.Z_t = []
        self.averaging = averaging
        self.hs = defaultdict(AdaHedge)
        #self.N = np.zeros(self.prob.m)

        self.g = self.prob.d
        
        self.design = np.zeros(self.prob.m)
        self.design_hist = []
        self.cumulative_design = np.zeros(self.prob.m)
        self.count_fe = 0
        
        self.i0 = 0
        self.c0 = 0.1
        #self.c1 = 0.1
        #self.c2 = 0.1
        #self.u = 0.1
        self.psi_inv = 0


    def alt_min_k(self, design, k):
        sum_design = np.sum(design)
        
        #if sum_design < 1e-10:
        #    return 0, µ, k
        
        design = design / sum_design  # Avoid dividing by small quantities
        arm = self.prob.x[k,:]
        #arm_star = armstar(pep, µ)
        
        #assert arm != arm_star
        K = self.prob.m#len(pep.arms)
        #direction = arm - arm_star
        d = self.prob.d#len(direction)
        
        # Construction of the matrix V = (sum_k w_k a_k a_k^T)^{-1}
        sum_arms_matrix = np.zeros((d, d))
        for j in range(K):
            sum_arms_matrix += design[j] * np.outer(self.prob.x[j], self.prob.x[j])
        
        Vinv = np.linalg.inv(sum_arms_matrix)
        
        # Closest point
        eta = (np.dot(arm, self.theta_hat) - self.rho) / np.dot(np.dot(arm, Vinv), arm)
        theta_prime = self.theta_hat - eta * np.dot(Vinv, arm)
        
        # Divergence to that point
        val = 0.5 * sum_design * (np.dot(arm, self.theta_hat) - self.rho)**2 / np.dot(np.dot(arm, Vinv), arm)
        
        return val, theta_prime
    
    
    def alt_min(self, design):
        alt = [self.alt_min_k(design, i) for i in range(self.prob.m)]
        min_result = alt[np.argmin([a[0] for a in alt])][1]
        return min_result

    def glrt(self, design):
        # Check that the size of μ is 1
        #assert len(μ) == 1
    
        star = None#istar(pep, μ)
        theta_prime = self.alt_min(design)
        val = None
        k = None
    
        # Return a tuple containing the computed values
        return val, (k, theta_prime), (star, None)

    def optimistic_gradient(self, t, design, theta_prime, Vinv):
        m = self.prob.m
        grads = np.zeros(m)
        for k in range(m):
            arm = self.prob.x[k,:]
            ref_value = np.dot((self.theta_hat - theta_prime), arm)
            confidence_width = np.log(t)
            deviation = np.sqrt(2 * confidence_width * np.dot(arm, np.dot(Vinv, arm)))
            if ref_value > 0:
                grads[k] = 0.5 * (ref_value + deviation)**2
            else:
                grads[k] = 0.5 * (ref_value - deviation)**2
            grads[k] = min(grads[k], confidence_width)
        return grads
    
    def _sampling_rule(self, t):
        f = self.c0 * np.sqrt(t);
        Vinv = self.A_inv
        theta_hat = self.theta_hat
        #nb_I = nanswers(pep, theta_hat) ### number of answ?
        m = self.prob.m
        #star = istar(pep, theta_hat) ### optimal??
        #star = self.best_arm_hat
        star = str(set(self.good_arms_hat))
        
        if star not in self.hs:  # if we never saw that star, initialize an AdaHedge learner
            self.hs[star] = AdaHedge(m)

        # query the learner
        design = self.hs[star].act()
        self.design = design

        # best response λ-player
        _, (_, theta_prime), (_, _) = self.glrt(design)

        opt_grad = self.optimistic_gradient(t, self.prob.T, theta_prime, Vinv)

        self.hs[star].incur(-opt_grad)

        #k = track(sr.t, N, w)
        
        #if np.min(np.linalg.eigvals(self.A)) < f:
        #    arm = self.A0[self.i0]
        #    self.i0 = np.mod(self.i0 + 1, self.prob.d)
        #    self.count_fe += 1;
        #else:
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
            #self.support_history.append(np.sum(self.support))
            
        return arm
    
    def _sampling_rule2(self, t):
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
             
    def run(self, T_max, delta, seed):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        
        #temp = np.zeros(self.prob.d)
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        #self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        np.random.seed(seed)
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            action = self._sampling_rule(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)

            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
            
            self.design_hist.append(self.design)
            
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau
