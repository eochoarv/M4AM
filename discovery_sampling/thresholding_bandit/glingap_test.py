#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:31:16 2023

@author: eochoa
"""

class LinGapE(linear_solver):
    def __init__(self, prob, rho, epsilon, alpha):
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
        self.alpha = alpha
        
    def init_priors(self, s0 = 1.0):
        self.clf = LogisticRegression(random_state=0)
        self.theta_hat = np.zeros(self.prob.d)
        self.sigma_hat = s0 * np.eye(self.prob.d)  # to adapt according to the true distribution of theta
        
    def update_posterior(self, action, r):
        
        if len(np.unique(self.prob.rewards)) >= 2:
            self.clf.fit(self.prob.x[self.prob.actions, :], np.array(self.prob.y[self.prob.actions]))
            self.theta = self.clf.coef_
            #self.intercept = self.clf.intercept_
    
        self.mean_hat = expit(self.prob.x @ self.theta_hat)
        #self.sigma_hat = np.linalg.pinv(self.A)
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        
        #self.temp += r * self.prob.x[action]     
        self.A += np.outer(self.prob.x[action], self.prob.x[action])
        #self.theta_hat = np.linalg.pinv(self.A) @ self.temp
        #self.mean_hat = self.prob.x @ self.theta_hat
        self.sigma_hat = np.linalg.pinv(self.A)
        self.A_inv = self.sigma_hat
        
        self.good_arms_hat = np.where(self.mean_hat > self.rho)[0]
        self.bad_arms_hat = np.where(self.mean_hat <= self.rho)[0]
        self.best_arm_hat = np.argmax(self.mean_hat)
        
        return self.theta_hat#, sigma_
    
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
            action, B = self.select_gap(t, delta)
            r, _ = self.prob.take_action(action)
            self.update_posterior(action, r)
                        
            if B < epsilon:
                tau = t
                break
        
        #success = self.good_arms < self.good_arms_hat and self.bad_arms < self.bad_arms_hat
        
        #true_means = self.prob.x @ self.prob.theta
        #performance = ((true_means[self.good_arms_hat] + epsilon <= self.rho).sum(),
        #                (true_means[self.bad_arms_hat] - epsilon > self.rho).sum())
        
        performance = self.performance()
        performance_e = self.performance_e()
        
        return self.good_arms_hat, (performance, performance_e), tau
