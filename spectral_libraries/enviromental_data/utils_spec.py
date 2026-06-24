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



def gen_data_sets(data_1, data_2, data_test, seed=0):
    
    #np.random.seed(seed)
    
    #poly_list_data_2 = np.unique(data_2['polymer'])
    #idx_strat = [np.where(data_2['polymer'] == p)[0] for p in poly_list_data_2]
    #ids_ref = np.concatenate([np.random.choice(x, size=int(x.shape[0]/2), replace=False) for x in idx_strat])
    #idx_ref = np.array([i in ids_ref for i in range(data_2.shape[0])])
    
    #data_ref = pd.concat([data_1, data_2.loc[idx_ref,:]], axis=0).reset_index(drop=True)
    #data_rest = data_2.loc[~idx_ref,:].reset_index(drop=True)
    data_ref = data_1
    data_rest = data_2

    poly_list = np.unique(data_ref['polymer'])
    label_table = pd.DataFrame({'label':np.arange(poly_list.shape[0]),
                                'polymer':poly_list})
    
    data_ref = data_ref.merge(label_table)
    data_rest = data_rest.merge(label_table)

    
    reference_library = (data_ref.
                         iloc[:,[-2,-3,-1]+list(range(data_ref.shape[1]-5))].
                         #loc[idx_ref,:].
                         reset_index(drop=True))

    calibration_set = (data_rest.
                         iloc[:,[-2,-3,-1]+list(range(data_rest.shape[1]-5))].
                         #loc[~idx_ref,:].
                         reset_index(drop=True))

    test_set = (data_test.
                         iloc[:,[-3,-2,-1]+list(range(data_test.shape[1]-5))].
                         reset_index(drop=True))

    return reference_library, calibration_set, test_set, label_table

def process_data(reference_library,
                 calibration_set,
                 test_set):

    # store labels/polymer for reference library
    y_train = reference_library.iloc[:,1].to_numpy()
    label_train = reference_library.iloc[:,2].to_numpy()
    # store spectra for reference library
    X_train = reference_library.iloc[:,3:].to_numpy()
    wl_list = np.float64(reference_library.columns[3:])

    # Perform SNV normalization
    X_train = (X_train - X_train.mean(1)[:, None]) / X_train.std(1)[:, None]

    # store labels/polymer for calibration set
    y_cal = calibration_set.iloc[:,1].to_numpy()
    label_cal = calibration_set.iloc[:,2].to_numpy()
    # store spectra for calibration set
    X_cal = calibration_set.iloc[:,3:].to_numpy()

    # Perform SNV normalization
    X_cal = (X_cal - X_cal.mean(1)[:, None]) / X_cal.std(1)[:, None]


    # store labels/polymer for test set
    y_test = test_set.iloc[:,1].to_numpy()
    label_test = test_set.iloc[:,2].to_numpy()
    # store spectra for calibration set
    X_test = test_set.iloc[:,3:].to_numpy()

    # Perform SNV normalization
    X_test = (X_test - X_test.mean(1)[:, None]) / X_test.std(1)[:, None]
    n_test = X_test.shape[0]

    return X_train, X_cal, X_test, y_train, y_cal, y_test, label_train, label_cal, label_test, n_test, wl_list


def gen_data_sets_ir(data_1, data_2, data_test, seed=0):
    
    #np.random.seed(seed)
    
    #poly_list_data_2 = np.unique(data_2['polymer'])
    #idx_strat = [np.where(data_2['polymer'] == p)[0] for p in poly_list_data_2]
    #ids_ref = np.concatenate([np.random.choice(x, size=int(x.shape[0]/2), replace=False) for x in idx_strat])
    #idx_ref = np.array([i in ids_ref for i in range(data_2.shape[0])])
    
    #data_ref = pd.concat([data_1, data_2.loc[idx_ref,:]], axis=0).reset_index(drop=True)
    #data_rest = pd.concat([data_3, data_2.loc[~idx_ref,:]], axis=0).reset_index(drop=True)
    #data_rest = data_2.loc[~idx_ref,:].reset_index(drop=True)
    
    data_ref = data_1
    data_rest = data_2
    
    poly_list = np.unique(data_ref['polymer'])
    label_table = pd.DataFrame({'label':np.arange(poly_list.shape[0]),
                                'polymer':poly_list})
    data_ref = data_ref.merge(label_table)
    data_rest = data_rest.merge(label_table)

    
    reference_library = (data_ref.
                         iloc[:,[-2,-3,-1]+list(range(data_ref.shape[1]-5))].
                         #loc[idx_ref,:].
                         reset_index(drop=True))

    calibration_set = (data_rest.
                         iloc[:,[-2,-3,-1]+list(range(data_rest.shape[1]-5))].
                         #loc[~idx_ref,:].
                         reset_index(drop=True))

    #gloves_data_ir.iloc[:,[-4,-2,-1]+list(range(gloves_data_ir.shape[1]-6))]
    
    test_set = (data_test.
                         iloc[:,[-4,-2,-1]+list(range(data_test.shape[1]-6))].
                         reset_index(drop=True))

    return reference_library, calibration_set, test_set, label_table

def process_data_ir(reference_library,
                 calibration_set,
                 test_set):

    # store labels/polymer for reference library
    y_train = reference_library.iloc[:,1].to_numpy()
    label_train = reference_library.iloc[:,2].to_numpy()
    # store spectra for reference library
    X_train = reference_library.iloc[:,3:].to_numpy()
    wl_list = np.float64(reference_library.columns[3:])

    # Perform SNV normalization
    X_train = (X_train - X_train.mean(1)[:, None]) / X_train.std(1)[:, None]

    # store labels/polymer for calibration set
    y_cal = calibration_set.iloc[:,1].to_numpy()
    label_cal = calibration_set.iloc[:,2].to_numpy()
    # store spectra for calibration set
    X_cal = calibration_set.iloc[:,3:].to_numpy()

    # Perform SNV normalization
    X_cal = (X_cal - X_cal.mean(1)[:, None]) / X_cal.std(1)[:, None]


    # store labels/polymer for test set
    y_test = test_set.iloc[:,1].to_numpy()
    label_test = test_set.iloc[:,2].to_numpy()
    # store spectra for calibration set
    X_test = test_set.iloc[:,3:].to_numpy()

    # Perform SNV normalization
    X_test = (X_test - X_test.mean(1)[:, None]) / X_test.std(1)[:, None]
    n_test = X_test.shape[0]

    return X_train, X_cal, X_test, y_train, y_cal, y_test, label_train, label_cal, label_test, n_test, wl_list


def matching(ref_library, spectra_data_ir, limits, lib_name, data_name):
    
    lim1, lim2 = limits

    wn_gl = np.array(spectra_data_ir.columns[:-4])

    idx_gl = list(np.logical_or(np.logical_and(wn_gl > lim1[0], wn_gl < lim1[1]),
                           np.logical_and(wn_gl > lim2[0], wn_gl < lim2[1])
                          ) ) + [False]*4


    wn_fl = np.array(ref_library.columns[:-4])

    idx_fl = list(np.logical_or(np.logical_and(wn_fl > lim1[0], wn_fl < lim1[1]),
                                np.logical_and(wn_fl > lim2[0], wn_fl < lim2[1])
                                )) + [False]*4
    
    #idx_fl[np.where(idx_fl)[0][-1]] = False

    x_train_ir = ref_library.iloc[:,idx_fl].to_numpy()
    x_train_ir = norm_spect_pivot(x_train_ir, norm_type='SNV')
    y_train_ir = ref_library.iloc[:,-2]


    x_test_ir = spectra_data_ir.loc[:,idx_gl].to_numpy()
    x_test_ir = norm_spect_pivot(x_test_ir, norm_type='SNV')
    
    pred_ = corr_matching(x_train_ir, y_train_ir, 
                          x_test_ir, r_min=0)
    pred_polymers = pred_[0]
    pred_hqi = pred_[1]
    pred_df = pd.DataFrame(pred_polymers)
    pred_df = pred_df.reset_index(drop=True) 
    polymer_name = 'polymer_'+lib_name
    hqi_name = 'hqi_'+lib_name
    pred_df.columns = [polymer_name]
    pred_df['id'] = spectra_data_ir['id']
    pred_df['background_'+data_name] = spectra_data_ir['background']
    pred_df[hqi_name] = pred_hqi
    
    
    #matches_mean = pred_df.reset_index().query('@hqi_name>.7').query('background!="background"').groupby(polymer_name).count()[hqi_name]

    #spectra_data_ir['polymer_flopp'] = pred_df['polymer_flopp']
    #spectra_data_ir['hqi_flopp'] = pred_df['hqi_flopp']

    return pred_df, pred_

def matching_r(ref_library, spectra_data_ir, limits, lib_name, data_name):
    
    lim1, lim2 = limits

    wn_gl = np.array(spectra_data_ir.columns[:-3])

    idx_gl = list(np.logical_and(wn_gl > lim1, wn_gl < lim2)) + [False]*3


    wn_fl = np.array(ref_library.columns[:-4])

    idx_fl = list(np.logical_and(wn_fl > lim1, wn_fl < lim2)) + [False]*4
    
    #idx_fl[np.where(idx_fl)[0][-1]] = False

    x_train_ir = ref_library.iloc[:,idx_fl].to_numpy()
    x_train_ir = norm_spect_pivot(x_train_ir, norm_type='SNV')
    y_train_ir = ref_library.iloc[:,-2]


    x_test_ir = spectra_data_ir.loc[:,idx_gl].to_numpy()
    x_test_ir = norm_spect_pivot(x_test_ir, norm_type='SNV')
    
    pred_ = corr_matching(x_train_ir, y_train_ir, 
                          x_test_ir, r_min=0)
    pred_polymers = pred_[0]
    pred_hqi = pred_[1]
    pred_df = pd.DataFrame(pred_polymers)
    pred_df = pred_df.reset_index(drop=True) 
    polymer_name = 'polymer_'+lib_name
    hqi_name = 'hqi_'+lib_name
    pred_df.columns = [polymer_name]
    pred_df['id'] = spectra_data_ir['id']
    #pred_df['background_'+data_name] = spectra_data_ir['background']
    pred_df[hqi_name] = pred_hqi
    
    
    #matches_mean = pred_df.reset_index().query('@hqi_name>.7').query('background!="background"').groupby(polymer_name).count()[hqi_name]

    #spectra_data_ir['polymer_flopp'] = pred_df['polymer_flopp']
    #spectra_data_ir['hqi_flopp'] = pred_df['hqi_flopp']

    return pred_df, pred_