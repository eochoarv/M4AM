#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:03:59 2025

@author: eochoa
"""
import numpy as np
import pandas as pd
from utils_cp import *
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from tqdm import tqdm

def gen_data_sets(data_1, data_2, data_test, seed):
    
    np.random.seed(seed)
    #idx_ref = np.random.rand(data_2.shape[0]) > .80
    
    poly_list_data_2 = np.unique(data_2['polymer'])
    idx_strat = [np.where(data_2['polymer'] == p)[0] for p in poly_list_data_2]
    ids_ref = np.concatenate([np.random.choice(x, size=int(x.shape[0]/2), replace=False) for x in idx_strat])
    idx_ref = np.array([i in ids_ref for i in range(data_2.shape[0])])
    
    data_ref = pd.concat([data_1, data_2.loc[idx_ref,:]], axis=0).reset_index(drop=True)
    data_rest = data_2.loc[~idx_ref,:].reset_index(drop=True)
    
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

#rebecca_df,stearates_df,gloves_data_raman
#slopp_df,stearates_df,gloves_data_raman
        
def sim_split(df_1, df_2, df_3, N_sim):
    
    res = defaultdict(list)
    distr_list = []

    for i in tqdm(range(N_sim)):
        ## Process spectral data
        seed = i
        reference_library, calibration_set, test_set, label_table = gen_data_sets(#rebecca_df,
                                                                                  df_1,
                                                                                  df_2,
                                                                                  df_3,
                                                                                  seed)

        (X_train, X_cal, X_test, 
         y_train, y_cal, y_test, 
         label_train, label_cal, label_test, 
         n_test, wl_list) = process_data(reference_library,
                         calibration_set,
                         test_set)

        ## Calculate HQI

        # Nearest Neighbor

        # Calculate HQI for calibration set
        smx_cal = smx_nn(X_train, X_cal, label_train)

        # Calculate HQI for test set
        smx_test = smx_nn(X_train, X_test, label_train)

        hist_data = plt.hist(smx_cal[np.arange(X_cal.shape[0]),label_cal], density=False, color='b', alpha=.3, bins=np.linspace(0, 1, 100), )
        distr_list.append(hist_data[0][:,None])
        ## Perform Conformal Prediction over the test set

        # Set confidence level 1 - alpha
        n = smx_cal.shape[0]
        alpha = 0.1
        for alpha in [alpha]:#np.linspace(0.01,.2):
            prediction_sets, qhat = conform_pred(smx_cal, smx_test, label_cal, n, alpha, cond=0)

            # If we know to true labels of the test set, calculate the empirical confidence 
            if label_test[0] != '' and not np.isnan(label_test[0]):
                empirical_confidence = prediction_sets[np.arange(n_test),label_test].mean()
            else:
                empirical_confidence = None

            #print('Theo Confidence: '+str(1-alpha))
            #print('Threshold(s) used: '+str(np.int64(100*(1-qhat))/100))
            #if empirical_confidence is not None:
            #    print('Empirical confidence: '+str(int(100*empirical_confidence)/100))
            #print('Mean prediction set size: '+str(np.round(prediction_sets.sum(1).mean(), 2)))
            #print('')

        prediction_sets_n = [list(label_table.loc[ps,'polymer']) for ps in prediction_sets]
        prediction_sets_size = np.array([len(ps) for ps in prediction_sets_n])

        zero_label = np.where(prediction_sets_size == 0)[0]
        single_label = np.where(prediction_sets_size == 1)[0]
        mult_label = np.where(prediction_sets_size > 1)[0]

        #print('Sets with zero labels: ' + str(zero_label.shape[0]))
        #print('Sets with one label: ' + str(single_label.shape[0]))
        #print('Sets with >1 labels: ' + str(mult_label.shape[0]))

        #if y_test[0] != '' and not np.isnan(label_test[0]):
        #    print('accuracy of single labels: ' +str(np.mean([prediction_sets_n[i][0]==y_test[i] for i in single_label])))

        output = [prediction_sets_n[i][0] if i in single_label else 
              '--outlier--' if i in zero_label else 
              '--manual check!--' for i in range(label_test.shape[0])]
        output_df = pd.DataFrame({'id':test_set['id'],
                              'prection_set':prediction_sets_n,
                              'output':output})

        (polymer_list_rb,
        polymer_list_slp,
        stearate_list) = (np.unique(rebecca_df['polymer']),
        np.unique(slopp_df['polymer']),
        np.unique(stearates_df['polymer'])
        )
        table_poly_class = pd.DataFrame({'polymer':np.concatenate([polymer_list_rb,
        polymer_list_slp,
        stearate_list]),
        'class':np.concatenate([['polymer']*polymer_list_rb.shape[0],
        ['polymer']*polymer_list_slp.shape[0],
        ['stearate']*stearate_list.shape[0]])}).set_index('polymer')
        prediction_sets_class = [list(np.unique([table_poly_class.loc[p,'class'] for p in x])) for x in output_df['prection_set']]
        #prediction_sets_class
        prediction_sets_class = pd.DataFrame({'class':[x[0] if len(x)==1 else 'empty' if len(x)==0 else 'both' for x in prediction_sets_class],
         'n':1}).groupby('class').sum()


        res['threshold_list'].append(np.round(1-qhat, 4))
        res['mean_prediction_set_size_list'].append(np.round(prediction_sets.sum(1).mean(), 2))

        res['sets_zero_list'].append(zero_label.shape[0])
        res['sets_one_list'].append(single_label.shape[0])
        res['sets_one_plus_list'].append(mult_label.shape[0])

        for class_ in ['both', 'polymer', 'stearate', 'empty']:
            if class_ in prediction_sets_class.index:
                res['n_pred_sets_'+class_].append(prediction_sets_class.loc[class_,'n'])

    res_df = pd.concat([pd.DataFrame(res).mean(),
    pd.DataFrame(res).std()], axis=1)
    res_df.columns = ['mean', 'std']
    res_df
    
    distr_df = pd.DataFrame({'freq':np.concatenate(distr_list,axis=1).mean(1),
             'hqi':np.linspace(0, 1, 100)[1:]})
    
    return res_df, distr_df