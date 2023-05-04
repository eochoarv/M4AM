#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:34:25 2023

@author: eochoa
"""

import numpy as np
import joblib
import importlib
import matplotlib.pyplot as plt
import os 
import pickle
import bayes_logistic as bl
from collections import defaultdict
import warnings
#warnings.filterwarnings("ignore")
from tqdm import trange, tqdm

os.chdir('~/Users/eochoa/Documents/Research_Abuj/ASD/Code/')

import pandas as pd
from utils import *
from linear_utils import *
from logistic_utils import *
from GP_utils import *
from linear_utils import *
import expt_utils as expt_utils
from sklearn.preprocessing import normalize
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import expit, logit
import sklearn
import seaborn as sns
# importlib.reload(expt_utils)
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances

from scipy.stats import multivariate_normal


n_mix = 4
S_list = []
rv_list = []
size_list = [600,200,150,50]
mean_list = [[5,5],[-5,-5],[-5,5],[5,-5]]
sigma_list = [1.7,1.8,1.5,1.7]

for i in range(n_mix):
    mui = mean_list[i]
    sigmai = sigma_list[i]*np.eye(2)
    Si = np.random.multivariate_normal(mui, sigmai, size_list[i])
    rvi = multivariate_normal(mean=mui, cov=sigmai) 
    pdfi = rvi.pdf(Si)
    #Si = np.concatenate([Si, i*np.ones((size_list[i],1)), pdfi.reshape((size_list[i],1))], axis=1)
    rv_list.append(rvi)
    S_list.append(Si)
    
x = np.concatenate(S_list)

N = 1000
p = 2
theta = 5*np.random.rand(p+1)
expit = lambda x: 1/(1+np.exp(-x))
#x = np.random.multivariate_normal(np.zeros(p),np.eye(p)+np.ones((p,p))*.5,N)
x_ = np.concatenate([np.ones((N,1)), x],axis=1)
#f = x @ theta
#p = expit(f) 
y = x_ @ theta + np.random.normal(size=N)#np.random.binomial(1,p)


n_row, n_col = y.shape[0], 1
_, n_desc =  x.shape
cutoff = n_desc
T = n_row
prob_dict = {
    "m": n_row,
    "noise": 0.01,
    "d": cutoff
  }
prob = linear_discover(**prob_dict)
x = x - x.mean(0)
x = normalize(x, axis=0, norm='l2')
prob.x = x.copy()
prob.y = y.copy()

        
T = 1000
t_prob = deepcopy(prob)
solver = Greedy_linear(t_prob, **algo_params)
reg, _ = solver.run(T)[:2]


from dash import Dash, dcc, html, Input, Output
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

app = Dash(__name__)

app.layout = html.Div([
    html.H4("Visualization of PCA's explained variance"),
    dcc.Graph(id="graph"),
    html.P("Number of components:"),
    dcc.Slider(
        id='slider',
        min=2, max=5, value=3, step=1)
])


@app.callback(
    Output("graph", "figure"), 
    Input("slider", "value"))

def run_and_plot(n):
    
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.scatter(x[solver.prob.actions[:n], 0], 
            x[solver.prob.actions[:n], 1], 
            c='r')

    fig = px.scatter_matrix(
        components,
        color=boston.target,
        dimensions=range(n_components),
        labels=labels,
        title=f'Total Explained Variance: {var:.2f}%')
    fig.update_traces(diagonal_visible=False)
    return fig


app.run_server(debug=True,host = '127.0.0.1')
