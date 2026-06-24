#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:58:46 2024

@author: eochoa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z



def find_peaks(spec, wn, n=10, dist=50):
    #idx_sort = np.argsort(x[j,])[::-1]
    #idx_decre = np.where(np.diff(x[j,:][:-1])<0)[0]
    #idx_incre = np.where(np.diff(x[j,:][:-1])>0)[0]+1
    
    idx_sort = np.argsort(spec)[::-1]
    idx_decre = np.where(np.diff(spec[:-1])<0)[0]
    idx_incre = np.where(np.diff(spec[:-1])>0)[0]+1
    
    idx_flatt_1 = np.where(np.diff(spec[:-1])==0)[0]
    idx_flatt_2 = np.where(np.diff(spec[:-1])==0)[0]+1
    
    idx_peaks = [i for i in range(spec.shape[0]) 
                 if (i in idx_decre and i in idx_incre) 
                 or (i in idx_flatt_1 and i in idx_flatt_2) ]
    
    peaks = []
    peaks_wn = []
    for i in idx_sort: 
        if i not in idx_peaks:
            continue
        elif len(peaks_wn)>0 and np.min(np.abs(np.array(peaks_wn) - wn[i])) < dist:
            continue
        else:
            peaks_wn.append(wn[i])
            peaks.append(i)
        if len(peaks) == n:
            return peaks, peaks_wn
        
    return None
    
    #peaks = np.array([i for i in idx_sort if i in idx_peaks])
    #peaks = np.array([i for i in idx_sort if i in idx_decre and i in idx_incre])


def distance_peaks(spec_1, spec_2, int_1, int_2):
    #dist_mat = pairwise_distances(np.array(peaks_list[0]).flatten()[:,None],
    #                   np.array(peaks_list[2]).flatten()[:,None])
    
    n = len(spec_1)
    dist_mat = pairwise_distances(np.array(spec_1).flatten()[:,None],
                                  np.array(spec_2).flatten()[:,None])
    
    dist_int_mat = pairwise_distances(np.array(int_1).flatten()[:,None],
                                      np.array(int_2).flatten()[:,None])
    
    min_val = dist_mat.min()
    dist_mat_aux = dist_mat.copy()
    idx_minval_list = []
    for i in range(n):
        #print(dist_mat_aux)
        #
        #min_val =
        idx_minval = np.array(np.where(dist_mat_aux == min_val))
        idx_minval_list.append(np.array(np.where(dist_mat == min_val)).T[0])
        dist_mat_aux = dist_mat_aux[np.logical_and(np.arange(n-i)[:,None] != idx_minval[0,0], 
                                                   np.arange(n-i)[None,:] != idx_minval[1,0])].reshape(n-1-i,n-1-i)

        if dist_mat_aux.shape[0] > 0: 
            min_val = dist_mat_aux.min()

    return np.sqrt((np.array([dist_mat[idx_[0], idx_[1]] * dist_int_mat[idx_[0], idx_[1]]
                              for idx_ in idx_minval_list]) ** 2).sum())


def k_means_cluster_peak(k, dist_mat):
    # Initialization: choose k centroids (Forgy, Random Partition, etc.)
    #centroids = [c1, c2, ..., ck]
    s = list(range(dist_mat.shape[0]))
    np.random.shuffle(s)
    centroids = s[-k:]
    
    points = list(range(dist_mat.shape[0]))
    
    # Initialize clusters list
    #clusters = [[] for _ in range(k)]
    
    #clusters = defaultdict(list)
    
    # Loop until convergence
    converged = False
    while not converged:
        # Clear previous clusters
        #clusters = [[] for _ in range(k)]
        clusters =  [[] for i in range(k)] #defaultdict(list)
        
        # Assign each point to the "closest" centroid 
        for point in points:
            #distances_to_each_centroid = [distance(point, centroid) for centroid in centroids]
            #distances_to_each_centroid = [dist_mat[point, centroid] for centroid in centroids]
            distances_to_each_centroid = dist_mat[point, centroids]
            cluster_assignment = np.argmin(distances_to_each_centroid)
            clusters[cluster_assignment].append(point)
        
        # Calculate new centroids
        #   (the standard implementation uses the mean of all points in a
        #     cluster to determine the new centroid)
        #new_centroids = [calculate_centroid(cluster) for cluster in clusters]
        
        #print(clusters)
        new_centroids = [cluster[(dist_mat[cluster,:][:,cluster] ** 2).sum(1).argmin()] for cluster in clusters]
        
        converged = (new_centroids == centroids)
        centroids = new_centroids
        
        if converged:
            return clusters

def interpol_spect(df_spectr, min_wl=0, max_wl=3500, n=1400, id_spetr='file_name'):
    
    spectr_list = df_spectr[id_spetr].drop_duplicates()
    df_spectr_min = df_spectr.groupby(id_spetr).min()
    df_spectr_max = df_spectr.groupby(id_spetr).max()
    all_wl = np.linspace(min_wl, max_wl, num=n, endpoint=True)

    df_spectr_inter = pd.DataFrame()
    

    for fn in spectr_list:
        df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True)
        #df_lenght = df_spectr_fn.shape[0]    
        min_wl = df_spectr_min.loc[fn, 'wavelength']
        max_wl = df_spectr_max.loc[fn, 'wavelength']
        
        polymer_fn = df_spectr_fn['polymer'][0]
    
        x = df_spectr_fn.loc[:, 'wavelength']
        y = df_spectr_fn.loc[:, 'intensity']
        f = interp1d(x, y)
        #x_new = np.linspace(min_wl, max_wl, num=df_lenght, endpoint=True)
        x_new = all_wl[np.logical_and(all_wl >= min_wl, all_wl <= max_wl)]
        x_out = all_wl[np.logical_or(all_wl < min_wl, all_wl > max_wl)]
        
        df_spectr_inter = pd.concat([df_spectr_inter, 
                                  pd.DataFrame({'wavelength':np.concatenate([x_new, x_out]), 
                                                'intensity':np.concatenate([f(x_new), np.zeros(len(x_out))]) , 
                                                'polymer':[polymer_fn] * n , 
                                                id_spetr:[fn] * n }).sort_values('wavelength')])
        
    return df_spectr_inter.reset_index(drop=True)
