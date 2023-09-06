#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:42:12 2023

@author: eochoa
"""


from random import randint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


#from bai_utils import *
from linear_bai_utils import *
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

x, y, theta = gen_data(1000, p=2, scale=20)
m, d = x.shape

prob_dict = {
        "m": m,
        "d": d,
        "x": x,
        "theta": theta,
      }

prob = linear_discover(**prob_dict)
solver = lazy_TS(prob, laziness_factor=1)
reg, rwd, best, tau = solver.run(1000, 0.05, 1)


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
def animate(i, x, best, designs, Ts):
    
    #mask = masks[i]
    design = designs[i]
    T = Ts[i]
    
    #print(mask)
    #print(design)
    ax[0].clear()
    ax[0].scatter(x[:,0],
                x[:,1],
                alpha=design)

    ax[0].scatter(x[best,0],
                  x[best,1],
                  c='r')
    ax[0].set_title(i)
    
    ax[1].clear()
    ax[1].scatter(x[:,0],
                x[:,1],
                alpha=T / sum(T))

    ax[1].scatter(x[best,0],
                  x[best,1],
                  c='r')
    ax[1].set_title(i)
    
    #ax.set_xlim([0,20])
    #ax.set_ylim([0,10])

n = len(solver.design_hist)
anim = animation.FuncAnimation(
    fig, 
    animate, 
    fargs=(
        x,
        best,
        solver.design_hist,
        solver.prob.T_hist
        ),
    frames=n, 
    interval=1000,
    repeat=True)

plt.show()