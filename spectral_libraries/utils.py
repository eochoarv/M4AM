#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:23:23 2023

@author: eochoa
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore")

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



def norm_spect(df_spectr, norm_type='MINMAX', id_spetr='file_name'):
    
    spectr_list = df_spectr[id_spetr].drop_duplicates()
    
    poly_list = df_spectr['polymer'].drop_duplicates().to_numpy()
    
    df_spectr_min = df_spectr.groupby(id_spetr).min()
    df_spectr_max = df_spectr.groupby(id_spetr).max()
    
    df_spectr_mean = df_spectr.groupby(id_spetr).mean()
    df_spectr_sd = df_spectr.groupby(id_spetr).std()
    
    df_spectr_min_p = df_spectr.groupby(['polymer', 'wavelength']).min().reset_index()
    df_spectr_max_p = df_spectr.groupby(['polymer', 'wavelength']).max().reset_index()
    
    df_spectr_mean_p = df_spectr.groupby(['polymer', 'wavelength']).mean().reset_index()
    df_spectr_sd_p = df_spectr.groupby(['polymer', 'wavelength']).std().reset_index()
    
    df_spectr_norm = pd.DataFrame()
    
    
    if norm_type == 'MINMAX':

        for fn in spectr_list:
            df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True) 
            min_norm = df_spectr_min.loc[fn, 'intensity']
            max_norm = df_spectr_max.loc[fn, 'intensity']
            
            df_spectr_fn.loc[:,'intensity'] = (df_spectr_fn['intensity'] - 
                                               min_norm) / (max_norm - min_norm)
            
            df_spectr_norm = pd.concat([df_spectr_norm, 
                                         df_spectr_fn])
        return df_spectr_norm
    
    if norm_type == 'SNV':
    
        for fn in spectr_list:
            df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True)
            mean_norm = df_spectr_mean.loc[fn, 'intensity']
            sd_norm = df_spectr_sd.loc[fn, 'intensity']
            
            df_spectr_fn.loc[:,'intensity'] = (df_spectr_fn['intensity'] - 
                                               mean_norm) / sd_norm
            
            df_spectr_norm = pd.concat([df_spectr_norm, 
                                         df_spectr_fn])
            
        return df_spectr_norm.reset_index(drop=True)
    
    
    #if norm_type == 'MINMAX2':

     #   for p in poly_list:
     #       df_spectr_p = df_spectr.query('polymer == @p').reset_index(drop=True) 
     #       min_norm = df_spectr_min.loc[fn, 'intensity']
     #       max_norm = df_spectr_max.loc[fn, 'intensity']
            
     #       min_norm = (df_spectr_min_p.
     #                   query('polymer == @p').
     #                   loc[:, ['wavelength', 'intensity']].
     #                   rename(columns={'intensity':'min_intensity'}))
     #       max_norm = (df_spectr_max_p.
     #                   query('polymer == @p').
     #                   loc[:, ['wavelength', 'intensity']].
     #                   rename(columns={'intensity':'max_intensity'}))
            
     #       df_spectr_fn.loc[:,'intensity'] = (df_spectr_fn['intensity'] - 
     #                                          min_norm) / (max_norm - min_norm)
            
     #       df_spectr_norm = pd.concat([df_spectr_norm, 
     #                                    df_spectr_fn])
     #   return df_spectr_norm
    
    if norm_type == 'SNV2':
    
        for p in poly_list:
            df_spectr_p = df_spectr.query('polymer == @p').reset_index(drop=True) 
            mean_norm = (df_spectr_mean_p.
                         query('polymer == @p').
                         loc[:, ['wavelength', 'intensity']].
                         rename(columns={'intensity':'mean_intensity'}))
            sd_norm = (df_spectr_sd_p.
                       query('polymer == @p').
                       loc[:, ['wavelength', 'intensity']].
                       rename(columns={'intensity':'sd_intensity'}))
            df_spectr_p = (df_spectr_p.
                           merge(mean_norm, on='wavelength').
                           merge(sd_norm, on='wavelength'))
            df_spectr_p.loc[:,'intensity'] = (df_spectr_p['intensity'] - 
                                              df_spectr_p['mean_intensity']) / df_spectr_p['sd_intensity']
            df_spectr_p = df_spectr_p.drop(columns=['mean_intensity', 'sd_intensity'])
            #df_spectr_p.loc[:,'intensity'] = (df_spectr_fn['intensity'] - mean_norm) / sd_norm
            
            df_spectr_norm = pd.concat([df_spectr_norm, 
                                         df_spectr_p])
            
        return df_spectr_norm.reset_index(drop=True)


def corr_matching(X_ref, y_ref, X_match, r_min=0.5):
    y_ref_ = y_ref.reset_index(drop=True)
    n_ref = X_ref.shape[0]
    #n_match = X_match.shape[0]
    R = np.corrcoef(X_ref, X_match)[:n_ref, n_ref:]
    return(y_ref_[R.argmax(0)], R.max(0))


def nn_matching(X_ref, y_ref, X_match, r_min=0.5, tau=10):
    y_ref_ = y_ref.reset_index(drop=True)
    n_ref = X_ref.shape[0]
    D = pairwise_distances(X_ref, X_match)
    df_D = pd.DataFrame(D)
    df_D['polymer_ref'] = y_ref
    smx = (df_D.melt(id_vars=['polymer_ref']).
                 groupby(['variable','polymer_ref']).
                 min().
                 reset_index().
                 pivot(index='variable', columns='polymer_ref', values='value'))

    label_list = np.array(smx.columns)
    smx = smx.to_numpy()
    smx_1 = (smx / smx.min(1)[:,None])
    smx_2 = (smx / np.sort(smx, 1)[:,1][:,None])
    smx_1[smx_1 == 1] = smx_2[smx_1 == 1]
    smx = np.exp(-smx_1/tau)
    #(y_ref_[smx.argmin(1)], smx.min(1))
    
    return(label_list[np.argmax(smx, 1)], smx.max(1))

def corr_matching_nan(X_ref, y_ref, X_match, r_min=0.5):
    y_ref_ = y_ref.reset_index(drop=True)
    n_ref = X_ref.shape[0]
    #n_match = X_match.shape[0]
    X_match_zeros = X_match.copy()
    X_match_zeros[np.isnan(X_match_zeros)] = 0
    X_ref_nan = X_ref.copy()
    c = np.sum(np.isnan(X_match), 1)
    
    R = (np.cov(X_ref, X_match_zeros))/ (np.nanstd(X_ref_nan, 1)[:,None] * np.nanstd(X_match, 1)[None,:]) * (c-1)/c
    R = R[:n_ref, n_ref:]
    #R = np.corrcoef(X_ref, X_match)[:n_ref, n_ref:]
    return(y_ref_[R.argmax(0)], R.max(0))


def norm_spect_pivot(x, norm_type='SNV'):
    if norm_type=='SNV':
        return (x - x.mean(1)[:,None]) / x.std(1)[:,None]
    if norm_type=='MINMAX':
        return (x - x.min(1)[:,None]) / (x.max(1)[:,None] - x.min(1)[:,None])
    if norm_type=='Q':
        iqr = np.quantile(x, .8, axis=1)-np.quantile(x, .2, axis=1)
        return (x - np.median(x, 1)[:,None]) / iqr[:,None]

