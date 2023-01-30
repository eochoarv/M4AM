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

warnings.filterwarnings("ignore")

def interpol_spect(df_spectr, min_wl=0, max_wl=3500, n=1400, id_spetr='file_name'):
    
    spectr_list = df_spectr[id_spetr].drop_duplicates()
    df_spectr_min = df_spectr.groupby('file_name').min()
    df_spectr_max = df_spectr.groupby('file_name').max()
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
                                                'polymer':[polymer_fn] * 1400 , 
                                                'file_name':[fn] * 1400 }).sort_values('wavelength')])
        
    return df_spectr_inter.reset_index(drop=True)



def norm_spect(df_spectr, norm_type='MINMAX', id_spetr='file_name'):
    
    spectr_list = df_spectr[id_spetr].drop_duplicates()
    
    df_spectr_min = df_spectr.groupby('file_name').min()
    df_spectr_max = df_spectr.groupby('file_name').max()
    
    df_spectr_mean = df_spectr.groupby('file_name').mean()
    df_spectr_sd = df_spectr.groupby('file_name').std()
    
    df_spectr_norm = pd.DataFrame()
    
    
    if norm_type == 'MINMAX':

        for fn in spectr_list:
            df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True) 
            min_norm = df_spectr_min.loc[fn, 'intensity']
            max_norm = df_spectr_max.loc[fn, 'intensity']
            
            df_spectr_fn.loc[:,'intensity'] = (df_spectr_fn['intensity'] - min_norm) / (max_norm - min_norm)
            
            df_spectr_norm = pd.concat([df_spectr_norm, 
                                         df_spectr_fn])
        return df_spectr_norm
    
    if norm_type == 'SNV':
    
        for fn in spectr_list:
            df_spectr_fn = df_spectr.query(id_spetr + ' == @fn').reset_index(drop=True)
            mean_norm = df_spectr_mean.loc[fn, 'intensity']
            sd_norm = df_spectr_sd.loc[fn, 'intensity']
            
            df_spectr_fn.loc[:,'intensity'] = (df_spectr_fn['intensity'] - mean_norm) / sd_norm
            
            df_spectr_norm = pd.concat([df_spectr_norm, 
                                         df_spectr_fn])
            
        return df_spectr_norm.reset_index(drop=True)
    
    
def corr_matching(X_ref, y_ref, X_match, r_min=0.5):
    y_ref_ = y_ref.reset_index(drop=True)
    n_ref = X_ref.shape[0]
    #n_match = X_match.shape[0]
    R = np.corrcoef(X_ref, X_match)[:n_ref, n_ref:]
    return(y_ref_[R.argmax(0)], R.max(0))
    