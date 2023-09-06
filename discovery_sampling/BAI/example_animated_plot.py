#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 09:37:52 2023

@author: eochoa
"""


from random import randint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


from linear_bai_as_utils import *
from weights_utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import seaborn as sns

from sklearn.preprocessing import normalize

expit = lambda x: 1/(1+np.exp(-x))
def gen_data(N=100, p=2, scale=1):

    #N = 10000
    #p = 10
    theta = scale*(2*np.random.rand(p)-1)
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

x, y, theta = gen_data(100, p=2, scale=20)
m, d = x.shape

prob_dict = {
        "m": m,
        "d": d,
        "x": x,
        "theta": theta,
      }

prob = linear_discover_as(**prob_dict)
solver = lazy_TS_as(prob, laziness_factor=1)
reg, rwd, best, tau = solver.run(96, 0.05, 1)


#mask = solver.prob.mask_mat_hist[i]
#design = solver.design_hist[i]
#plt.scatter(solver.prob.x[~mask,0],
#            solver.prob.x[~mask,1],
#            alpha=design)

#plt.scatter(solver.prob.x[solver.prob.best_arm,0],
#            solver.prob.x[solver.prob.best_arm,1],
#            c='r')



# create empty lists for the x and y data

# create the figure and axes objects
fig, ax = plt.subplots(ncols=2)

# function that draws each frame of the animation
def animate(i, x, best, best_hat, actions, masks, designs):
    
    mask = masks[i]
    design = designs[i]
    n = len(designs)
    
    #print(mask)
    #print(design)
    ax[0].clear()
    ax[0].scatter(x[~mask,0],
                x[~mask,1],
                alpha=design)

    ax[0].scatter(x[best,0],
               x[best,1],
                c='r')
    
    ax[0].scatter(x[best_hat,0],
                  x[best_hat,1],
                  c='orange')
    ax[0].set_title(str(i + 1) + '/' + str(n))
    
    ax[0].set_xlim((x[:,0].min(), x[:,0].max()))
    ax[0].set_ylim((x[:,1].min(), x[:,1].max()))
    
    ax[1].clear()
    ax[1].scatter(x[actions[:i],0],
                  x[actions[:i],1],
                  c=np.arange(i))

    ax[1].scatter(x[best,0],
                  x[best,1],
                  c='r')
    
    ax[1].scatter(x[best_hat,0],
                  x[best_hat,1],
                  c='orange')
    
    ax[1].set_title(str(i + 1) + '/' + str(n))
    
    ax[1].set_xlim((x[:,0].min(), x[:,0].max()))
    ax[1].set_ylim((x[:,1].min(), x[:,1].max()))

n = len(solver.design_hist)
anim = animation.FuncAnimation(
    fig, 
    animate, 
    fargs=(
        x,
        solver.prob.best_arm,
        best,
        solver.prob.actions,
        solver.prob.mask_mat_hist,
        solver.design_hist
        ),
    frames=n, 
    interval=1000,
    repeat=False)

plt.show()