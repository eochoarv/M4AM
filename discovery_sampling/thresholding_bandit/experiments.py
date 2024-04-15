#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:29:23 2023

@author: eochoa
"""

from tbp_utils_2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import normalize
import ipdb


d = 2
x = np.eye(d)
rho = 0.5
alpha = 0.1
x = np.concatenate([x, rho*(np.cos(alpha) * x[0:1] + np.sin(alpha) * x[1:2])])
#x = np.concatenate([x, (x[0:1]*(1/2+0.05) + x[1:2]*(1/2-0.05) )])#+ x[2:3]*(1/3))])
theta = x[0]
deltas = [0.1,0.05,0.01,0.001]

def sim_perf_deltas(solver_function, prob, algo_params, deltas, T_max, N, seed=0):
    
   #mean_list = defaultdict(list)
   #gap_list = defaultdict(list)

    mean_hat_list = defaultdict(list)
    tau_list = defaultdict(list)
    g_arms_list = defaultdict(list)
    perf_list = defaultdict(list)
    perf_e_list = defaultdict(list)
    A_list = defaultdict(list)
    #opt_desing = optimal_allocation(x, theta, rho, epsilon)
    #A_lambda = np.multiply((x[:,:,None] @ x[:,None]), opt_desing[0][:,None, None]).sum(0)
    #prob = linear_discover(**prob_dict)
    
    for delta in tqdm(deltas):
        for i in range(N):
            #gap_list[delta].append(np.sort(x @ theta)[-1] - np.sort(x @ theta)[-2])
            #mean_list[delta].append(x @ theta)
            
            t_prob = deepcopy(prob)
            solver = solver_function(t_prob, **algo_params) #(t_prob, rho, epsilon)
            g_arms, (perf, perf_e), tau = solver.run(T_max, delta, i+seed)
            tau_list[delta].append(tau)
            g_arms_list[delta].append(g_arms)
            perf_list[delta].append(perf[0] + perf[1])
            perf_e_list[delta].append(perf_e[0] + perf_e[1])
        #    succ_list1.append(best == prob.best_arm)
            #mean_hat_list[delta].append(x @ solver.theta_hat)
            A_list[delta].append(solver.A)
            
    return perf_list, perf_e_list, tau_list


def sim_perf(solver_function, probs, algo_params, delta, T_max, N, seed=0):
    

    mean_hat_list = defaultdict(list)
    tau_list = defaultdict(list)
    g_arms_list = defaultdict(list)
    perf_list = defaultdict(list)
    perf_e_list = defaultdict(list)
    A_list = defaultdict(list)
    
    #for prob in tqdm(probs):
    for probs_name, prob_list in tqdm(probs.items()):
        for i in range(N):
            prob = prob_list[i]
            #gap_list[delta].append(np.sort(x @ theta)[-1] - np.sort(x @ theta)[-2])
            #mean_list[delta].append(x @ theta)
            
            t_prob = deepcopy(prob)
            solver = solver_function(t_prob, **algo_params) #(t_prob, rho, epsilon)
            g_arms, (perf, perf_e), tau = solver.run(T_max, delta, i+seed)
            tau_list[probs_name].append(tau)
            g_arms_list[probs_name].append(g_arms)
            perf_list[probs_name].append(perf[0] + perf[1])
            perf_e_list[probs_name].append(perf_e[0] + perf_e[1])
        #    succ_list1.append(best == prob.best_arm)
            #mean_hat_list[delta].append(x @ solver.theta_hat)
            #A_list[d].append(solver.A)
            
    return perf_list, perf_e_list, tau_list
