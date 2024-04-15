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
from sklearn.linear_model import LogisticRegression
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
    

class logistic_discover(discover):
    def __init__(self, m, d, x, theta, intercept=0, noise=1):
        super(logistic_discover, self).__init__(m)
        #self.m = m
        self.x = x
        self.theta = theta
        self.intercept = intercept
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
        f = self.x[action,:] @ self.theta + self.intercept
        
        r = np.random.binomial(1, expit(f), 1)
        return r , None
    
    #def get_mask(self):
    #    mask_mat = np.array([False]*(self.m))
    #    for action in self.actions:
    #        mask_mat[action] = True
    #    self.mask_mat = mask_mat
    #    return mask_mat
    
    
    #def regret(self):
    #    f_star = self.x @ self.theta
    #    f = self.x[self.actions, :] @ self.theta
    #    reg = (np.max(f_star) - f).cumsum()
        
    #    return reg

    
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
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.theta_hat = np.zeros(self.prob.d)
        self.ratio_hist = []
        self.B_hist = []
        self.Ht = np.zeros([self.prob.d, self.prob.d])
        self.intercept_hat = 0
        self.epsilon = epsilon
        self.rho = rho
        self.init_priors()
        
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
        
        self.clf = LogisticRegression(random_state=0, fit_intercept=True, penalty='none')
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.Ht_inv = s0 * np.eye(self.prob.d)
        self.intercept_hat = 0
        self.mean_hat = np.zeros(self.prob.m)
        
    def update_posterior(self, action, r):
        
        #f = self.prob.x[action, :]
        #s_inv = np.linalg.inv(self.sigma_t)
        #ffT = np.outer(f, f)
        # pdb.set_trace()
        #mu_ = np.dot(np.linalg.inv(s_inv + ffT / self.prob.noise**2), np.dot(s_inv, self.mu_t) + r * f / self.prob.noise**2)
        #sigma_ = np.linalg.inv(s_inv + ffT/self.prob.noise**2)
        #self.mu_t = mu_
        #self.sigma_t = sigma_
        
        #self.temp += r * self.prob.x[action]
        
        if len(np.unique(self.prob.rewards)) >= 2:
            self.clf.fit(self.prob.x[self.prob.actions, :], np.array(self.prob.rewards).flatten())
            self.theta_hat = self.clf.coef_[0]
            self.intercept_hat = self.clf.intercept_
            
        #m_x = expit(self.prob.x[action] @ self.theta_hat)
        
        #self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        self.mean_hat = expit(self.prob.x @ self.theta_hat + self.intercept_hat)
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.Ht += np.outer(self.prob.x[action], self.prob.x[action]) * self.mean_hat[action] * (1-self.mean_hat[action])
        self.sigma_hat = np.linalg.pinv(self.A)
        self.A_inv = self.sigma_hat
        self.Ht_inv = np.linalg.pinv(self.Ht)
        
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    def step(self):
        pass
    def _stopping_rule(self, t, delta, epsilon):
        x = self.prob.x
        A_inv = self.sigma_hat
        H_inv = self.Ht_inv
        #C = np.log(self.prob.m ** 2 / delta) / np.log(t)
        #C = np.log( (6 / np.pi ** 2) * t * self.prob.x / delta)
        C = np.sqrt(self.prob.d * np.log((1 + t) / delta))# + 20
        theta_hat = self.theta_hat
        intercept_hat = self.intercept_hat
        #m = self.prob.m
        
        #gaps = ((x - x[:,None]).reshape(-1, 2) @ theta_hat).reshape(m,m)
        #gap = np.abs(x @ theta_hat - self.rho).min()
        gaps = np.abs(expit(x @ theta_hat +intercept_hat) - self.rho) + epsilon
        #norms = (np.sqrt(x[:,None] @ A_inv @ x[:,:,None]).flatten())
        norms = (np.sqrt(x[:,None] @ H_inv @ x[:,:,None]).flatten())
        self.ratio_hist.append(np.min(gaps/norms))
        self.B_hist.append(np.max(gaps+norms*C))
        return C < np.min(gaps/norms)
    
    def run(self, T_max):
        epsilon = self.epsilon
        self.prob.refresh()
        acts = np.random.choice(range(self.prob.m), T_max, replace=True)
        tau = np.nan
        for t in range(T_max):
            r, _ = self.prob.take_action(acts[t])
            self.update_posterior(acts[t], r)
            
            
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau
    
    
class Greedy_linear(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(Greedy_linear, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.d)
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
        self.theta_hat = np.zeros(self.prob.d)
        self.intercept_hat = 0
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
        rewards = expit(self.prob.x @ self.theta_hat + self.intercept_hat)
        gaps = np.abs(rewards - self.rho)
        #bad_arms = np.where(gaps <= 0)[0]
        amb_arms = np.where(gaps == 0)[0]
        #good_arms = np.where(gaps > 0)[0]
        gaps = np.delete(gaps, amb_arms, 0)
        #gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        #X = self.prob.x[good_arms, :]
        X = np.delete(self.prob.x, amb_arms, 0)
        X = (1 / gaps) * X
        _I = np.ones((self.prob.d, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.prob.x.T @ np.diag(design) @ self.prob.x)
            H_inv = np.linalg.pinv(self.prob.x.T@ np.diag(design * self.mean_hat * (1-self.mean_hat)) @ self.prob.x)
            U, D, V = np.linalg.svd(H_inv)
            #Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
            Hinvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            #newX = (X @ Ainvhalf)**2
            newX = (X @ Hinvhalf)**2
            phi = newX @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(phi)
            x = X[idx, :, None]
            #g = ((self.prob.x @ A_inv @ x) * (self.prob.x @ A_inv @ x)).flatten()
            g = ((self.prob.x @ H_inv @ x) * (self.prob.x @ H_inv @ x)).flatten()
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
    
    def _Lazy_update(self, t):
        if t == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *t), t+1])
            self.design, rho = self.__optimal_allocation(self.design)
            self.psi_inv = 2 *rho;

        
    def init_priors(self, s0 = 1.0):
        
        self.clf = LogisticRegression(random_state=0, fit_intercept=True, penalty='none')
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        self.intercept_hat = 0
        
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
            
            self._Lazy_update(t)
            self.design_hist.append(self.design)
            
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return self.good_arms_hat, performance, tau
    
    
class G_allocation(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(G_allocation, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.d)
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
        
        self.clf = LogisticRegression(random_state=0)
        #self.intercept = 0
        
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        #self.temp += r * self.prob.x[action]
        #self.A += np.outer(self.prob.x[action], self.prob.x[action])
        #self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        
        if len(np.unique(self.prob.rewards)) >= 2:
            self.clf.fit(self.prob.x[self.prob.actions, :], np.array(self.prob.y[self.prob.actions]))
            self.theta_hat = self.clf.coef_[0]
            #self.intercept = self.clf.intercept_
    
        self.mean_hat = expit(self.prob.x @ self.theta_hat)
        #self.sigma_hat = np.linalg.pinv(self.A)
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
        self.theta_hat = np.zeros(self.prob.d)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.opt_desing = opt_desing
        
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
        #at = np.nanargmax(self.mean_hat)
        arm = np.random.choice(self.prob.m, 1, p=self.opt_desing)[0]
        return arm
             
        
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



class G_allocation1(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(G_allocation1, self).__init__(prob, rho, epsilon)
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
        
        self.clf = LogisticRegression(random_state=0)
        #self.intercept = 0
        
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
                
        #self.temp += r * self.prob.x[action]
        #self.A += np.outer(self.prob.x[action], self.prob.x[action])
        #self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        
        if len(np.unique(self.prob.rewards)) >= 2:
            self.clf.fit(self.prob.x[self.prob.actions, :], np.array(self.prob.y[self.prob.actions]))
            self.theta_hat = self.clf.coef_
            #self.intercept = self.clf.intercept_
    
        self.mean_hat = expit(self.prob.x @ self.theta_hat)
        #self.sigma_hat = np.linalg.pinv(self.A)
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
    
    

class GLGapE(linear_solver):
    def __init__(self, prob, rho, epsilon, alpha):
        super(GLGapE, self).__init__(prob, rho, epsilon)
        self.init_priors()
        #self.theta_hat = np.zeros(self.prob.m)
        self.A = np.eye(self.prob.d)#np.zeros([self.prob.d, self.prob.d])
        self.A_inv = np.eye(self.prob.d)
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.alpha = alpha
        
    def performance_e(self):
        epsilon = self.epsilon
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return performance
    
    def performance(self):
        epsilon = self.epsilon
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] > self.rho).sum())
        
        return performance
    
    def init_priors(self, s0 = 1.0):
        self.clf = LogisticRegression(random_state=0)
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
    
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
        
    def confidence_bound(self, x, t):
        L = 1
        sigma = 1
        reg = 1
        d = self.prob.d
        m = self.prob.m
        tmp = np.sqrt(x.dot(self.A_inv).dot(x))
        #                 (sigma * np.sqrt(d * np.log(m * m * (1 + t * L * L) / reg / self.delta)) + np.sqrt(reg) * 2)
        C = np.sqrt(2*self.alpha*np.log(t)) #(sigma * np.sqrt(d * np.log(m * m * (1 + t * L * L) / reg / self.delta)) + np.sqrt(reg) * 2)
        return tmp * C

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
    
    
    def select_gap(self, t, delta):
        d = self.prob.d
        x = self.prob.x
        C_t = self.alpha * np.sqrt(2*d*t*np.log((np.pi**2)*d*(t**2)/(6*delta)))
        beta = C_t*(x[:,None,:] @ self.A_inv @ x[:,:,None]).flatten()
        B = np.abs(self.mean_hat - self.rho) + beta
        
        return np.argmax(B), np.max(B)
        
        
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

class glm_rage(linear_solver):
    def __init__(self, prob, rho, epsilon):
        super(glm_rage, self).__init__(prob, rho, epsilon)
        self.init_priors()
        self.theta_hat = np.zeros(self.prob.d)
        self.A = np.zeros([self.prob.d, self.prob.d])
        self.A0 = np.zeros(self.prob.d).astype(int)
        self.Ht = np.zeros([self.prob.d, self.prob.d])
        self.active_arms = list(range(self.prob.m))
        self.temp = np.zeros(self.prob.d)
        self.epsilon = epsilon
        self.rho = rho
        self.good_arms = np.where(self.prob.x @ self.prob.theta + epsilon > rho)
        self.bad_arms = np.where(self.prob.x @ self.prob.theta - epsilon < rho)
        self.factor = 5
        
    def performance_e(self):
        epsilon = self.epsilon
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        return performance
    
    def performance(self):
        epsilon = self.epsilon
        true_means = expit(self.prob.x @ self.prob.theta)
        performance = ((true_means[self.good_arms_hat] <= self.rho).sum(),
                        (true_means[self.bad_arms_hat] > self.rho).sum())
        
        return performance
      
    def update_posterior(self, action=None, r=None):
        
        if len(np.unique(self.prob.rewards)) >= 2:
            self.clf.fit(self.prob.x[self.prob.actions, :], np.array(self.prob.rewards).flatten())
            self.theta_hat = self.clf.coef_[0]
            #self.intercept = self.clf.intercept_
    
        self.mean_hat = expit(self.prob.x @ self.theta_hat)
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
        if action is not None:
            self.A += np.outer(self.prob.x[action], self.prob.x[action])
        self.sigma_hat = np.linalg.pinv(self.A)
        self.A_inv = self.sigma_hat
        
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
        
    def optimal_allocation(self):
        
        design = np.ones(self.prob.m)
        design /= design.sum()  
        
        max_iter = 5000
        
        x_hat = self.prob.x[self.active_arms, :]
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.prob.x.T@ np.diag(design) @ self.prob.x)    
            H_inv = np.linalg.pinv(self.prob.x.T@ np.diag(design * self.mean_hat * (1-self.mean_hat)) @ self.prob.x)
            U,D,V = np.linalg.svd(H_inv)
            Hinvhalf = U@np.diag(np.sqrt(D))@V.T
            
            newY = (x_hat@Hinvhalf)**2
            rho = newY@np.ones((newY.shape[1], 1))
                        
            idx = np.argmax(rho)
            y = x_hat[idx, :, None]
            g = ((self.prob.x @H_inv@ y )*(self.prob.x @H_inv@y)).flatten()
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
        design /= np.sum(design)
        
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
        self._build_A0()
        #n_past = self.prob.d*(self.prob.d + 1) + 1
        #rho = 1
        #rho_past = 1
        tau = np.nan
        self.N = 0
        continue_ = True
        t = 0
        np.random.seed(seed)
        
        while (len(self.active_arms) > 1) and continue_:
            #print(self.phase_index)
            self.delta_t = delta / (self.phase_index**2)
            design, rho = self.optimal_allocation()
            support = np.sum((design > 0).astype(int))
            #n_min = 2*self.factor*support
            n_min = (self.prob.d**2)*self.factor
            eps = 1/self.factor #approx factor
            #num_samples = max(np.ceil(8*(2**(self.phase_index-1))**2*rho*(1+eps)*np.log(2*self.prob.m**2/self.delta_t)), n_min).astype(int)
            num_samples = max(3*(1+eps)*rho*np.log(1/self.delta_t), n_min)
            #print(num_samples)
            #print(self.delta_t)
            #print(n_min)
            #print(rho)
            #print(design)
            
            allocation = np.int64(design * num_samples)#self.rounding(design, num_samples)
            print(num_samples)
            for a, n_samples in enumerate(allocation):
                #print(a)
                #print(n_samples)
                if n_samples > 0:
                    for j in range(n_samples):
                        #print(j)
                        r, _ = self.prob.take_action(a)
                        #self.update_posterior(a, r)
                        t = t + 1
                        self.N = self.N + 1   
                        if t >= T_max:
                            #tau = t
                            continue_ = False
                            #return t
                            break
                        #if self._stopping_rule(t, delta, epsilon):
                        #    tau = t
                        #    break
                        
            self.update_posterior()
            
            self.drop_arms()
            self.phase_index += 1
            
        tau = t
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau