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



expit = lambda x: 1/(1+np.exp(-x))
logit = lambda x: np.log(x / (1-x))

def randargmin(b,**kw):
    idxs = np.where(np.abs(b-b.min())<1e-20)[0]    
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
    def __init__(self, prob, rho, epsilon):
        super(Greedy_linear, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
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
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        self.temp += r * self.prob.x[action]
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
        return self.theta_hat
    
    def _sampling_rule(self, t):
        at = np.nanargmax(self.mean_hat)
        return at
    
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        theta_hat = self.theta_hat
        m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        
        return (norms * C < np.abs(x @ theta_hat - self.rho) + epsilon).sum() == m
             
        
    def run(self, T_max, delta):
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
        
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau
    
    
class lazy_TS(linear_solver):
    def __init__(self, prob, rho, epsilon, laziness_factor=1, averaging=True):
        
        super(lazy_TS, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.m)
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
            
                if np.linalg.norm(self.prob.x[arm]) >= .8:
                    rwd, _ = self.prob.take_action(arm)
                    self.update_posterior(arm, rwd)
                    #self.A += np.outer(self.prob.x[arm], self.prob.x[arm])
                    self.A0[r] = arm
                    r += 1
                    self.design_hist.append(self.design)
                    #logging.debug('current rank %s' % str(r))
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        self.c1 = np.min(np.linalg.eigvals(self.A))
        
    def __optimal_allocation(self, init, epsilon):

        #design = init
        design = np.ones(self.prob.m)
        design /= design.sum()
        # maximum number of iterations
        max_iter = 1000
        # construct Y
        rewards = self.prob.x @ self.theta_hat
        gaps = np.abs(rewards - self.rho) + epsilon
        #bad_arms = np.where(gaps <= 0)[0]
        amb_arms = np.where(gaps == 0)[0]
        #good_arms = np.where(gaps > 0)[0]
        #gaps = np.delete(gaps, bad_arms, 0)
        gaps = np.delete(gaps, amb_arms, 0)
        #gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        #X = self.prob.x[good_arms, :]
        X = np.delete(self.prob.x, amb_arms, 0)
        X = (1 / gaps) * X
        _I = np.ones((self.prob.d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.prob.x.T @ np.diag(design) @ self.prob.x)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newX = (X @ Ainvhalf)**2
            phi = newX @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(phi)
            x = X[idx, :, None]
            g = ((self.prob.x @ A_inv @ x) * (self.prob.x @ A_inv @ x)).flatten()
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
    
    def _Lazy_update(self, t, epsilon):
        if t == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *t), t+1])
            self.design, rho = self.__optimal_allocation(self.design, epsilon)
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
        
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
    def step(self):
        #theta_t = self.mu_t
        #pred = self.prob.x @ theta_t.transpose()
        #self.prob.get_mask()
        #pred[self.prob.mask_mat] = np.nan
        at = np.nanargmax(self.mean_hat)
        return at
    
    def _sampling_rule_2(self, t):
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
    
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        d = self.prob.d
        c_delta = 0#2*np.log(1/delta)*np.sqrt(np.log(1/delta)*(np.log(1/delta) + d))
        #C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        #C = np.log( (6 / np.pi ** 2) * t * self.prob.x / delta)
        #C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        C = np.sqrt(d * np.log(t / (d * delta * 1)) + c_delta/1)# + 20
        theta_hat = self.theta_hat
        #m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        gaps = np.abs(x @ theta_hat - self.rho) + epsilon
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        
        return C < np.min(gaps/norms)#(norms * C < gaps).sum() == m
             
    def run(self, T_max, delta, sampling_rule=1):
        epsilon = self.epsilon
        self.delta = delta
        self.prob.refresh()
        
        #temp = np.zeros(self.prob.d)
        self.design = np.ones(self.prob.m)
        self.design /= self.design.sum()
        
        self.c2 = (1+self.u) * (self.prob.sigma**2)
        self._build_A0()
        tau = np.nan
            
        for t in range(self.prob.d, T_max):
            if sampling_rule == 1:
                action = self._sampling_rule(t)
            else:
                action = self._sampling_rule2(t)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)

            
            if self._stopping_rule(t, delta, epsilon):
                tau = t
                break
            
            self._Lazy_update(t, epsilon)
            self.design_hist.append(self.design)
            
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau
    
    
class G_allocation(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(G_allocation, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.m)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        
        
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
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        self.temp += r * self.prob.x[action]
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
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
    
    
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        c_delta = 0
        d = self.prob.d
        #C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        #C = 2 * (2**0.5) * np.log( (6 / np.pi ** 2) * t ** 2 * self.prob.m / delta)
        #C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        C = np.sqrt(d * np.log(t / (d * delta * 1)) + c_delta/1)
        theta_hat = self.theta_hat
        m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        gaps = np.abs(x @ theta_hat - self.rho) + epsilon
        
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        
        return C < np.min(gaps/norms)
        
        #return (norms * C < gap).sum(1).max() == m - 1

        
    def run(self, T_max, delta):
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
        
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau



class Oracle_linear(linear_solver):
    def __init__(self, prob, rho, epsilon, opt_desing):
        super(Oracle_linear, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.m)
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
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.prob.d)
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        self.temp += r * self.prob.x[action]
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
        return self.theta_hat
    
    def _sampling_rule(self, t):
        f = self.c0 * np.sqrt(t);
        if np.min(np.linalg.eigvals(self.A)) < f:
            arm = self.A0[self.i0]
            self.i0 = np.mod(self.i0 + 1, self.prob.d)
            self.count_fe += 1;
        else:
                # Agressive update
            self.support = (self.opt_desing > 0)
            self.map_support = np.array(range(self.prob.m))[self.support]
            index = np.argmin((self.prob.T - t * self.opt_desing)[self.map_support])
                
            arm = self.map_support[index]
            #self.support_history.append(np.sum(self.support))
        return arm
    
    #def _sampling_rule(self, t):
        #at = np.nanargmax(self.mean_hat)
    #    arm = np.random.choice(self.prob.m, 1, p=self.opt_desing)[0]
    #    return arm
    
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        c_delta = 0
        d = self.prob.d 
        #C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        #C = 2 * (2**0.5) * np.log( (6 / np.pi ** 2) * t ** 2 * self.prob.m / delta)
        #C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        C = np.sqrt(d * np.log(t / (d * delta * 1)) + c_delta/1)
        theta_hat = self.theta_hat
        m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        gaps = np.abs(x @ theta_hat - self.rho) + epsilon
        
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        
        return C < np.min(gaps/norms)
             
        
    def run(self, T_max, delta):
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
        
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau

class xy_adaptive(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(xy_adaptive, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.d)
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
        
    def init_priors(self, s0 = 1.0):
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        self.temp += r * self.prob.x[action]
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.A_inv  = np.linalg.pinv(self.A)
        self.theta_hat = self.A_inv @ self.temp
        self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = self.A_inv
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
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
    
    def select_greedy_arm(self):
        
        score = np.diag(self.Yhat@self.A_inv@self.Yhat.T).reshape(1, -1) \
                - ((self.prob.x@self.A_inv@self.Yhat.T)**2/(1+ np.diag(self.prob.x @ self.A_inv @ self.prob.x.T).reshape(-1, 1)))
            
        arm_idx = randargmin(score[np.arange(self.prob.m), np.argmax(score, axis=1)])
        
        rho = np.max(score[arm_idx,:])
        
        return arm_idx, rho
    
    
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        #C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        #C = 2 * (2**0.5) * np.log( (6 / np.pi ** 2) * t ** 2 * self.prob.m / delta)
        C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        theta_hat = self.theta_hat
        m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        gaps = np.abs(x @ theta_hat - self.rho) + epsilon
        
        norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        
        return C < np.min(gaps/norms)
        
        #return (norms * C < gap).sum(1).max() == m - 1
        
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
        
        true_means = self.prob.x @ self.prob.theta
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau
