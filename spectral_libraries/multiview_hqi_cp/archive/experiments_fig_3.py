#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 09:56:01 2025

@author: eochoa
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from utils import *
#from utils_cpdir import *
from utils_experiments import *
from utils_mvcp import *
from utils import *
import os
from tqdm import tqdm


# Reference Libraries

rebecca_data_ir = pd.read_csv('processed_data_new/rebecca_data_ir_final.csv')
del rebecca_data_ir['Unnamed: 0']
rebecca_data_ir.columns = list(np.float64(rebecca_data_ir.columns[:-4])) + list(rebecca_data_ir.columns[-4:])

rebecca_data_raman = pd.read_csv('processed_data_new/rebecca_data_raman_final_specy.csv')
del rebecca_data_raman['Unnamed: 0']
rebecca_data_raman.columns = list(np.float64(rebecca_data_raman.columns[:-4])) + list(rebecca_data_raman.columns[-4:])


# Process Data
poly_abbr_table = pd.read_excel('data_new/CCOPTIR_20240604_MixedMPsStdsv2_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:9,-5:-3]
poly_abbr_table.columns = ['polymer', 'abbr']

poly_and_comments_list = pd.read_excel('data_new/CCOPTIR_20240604_MixedMPsStdsv2_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:,9:13]

poly_list = list(np.unique(rebecca_data_raman['polymer'])) + ['LDPE']

notes_mix_std_fluor = pd.read_excel('data_new/MVCP_20241206_RP_AllCCOPTIRData_FluorescenceOnly.xlsx',
         sheet_name='MPs + Standards').iloc[:, [0,9,10,11,12]]
notes_mix_std_fluor.columns = ['id','polymer_ir', 'polymer_raman', 'polymer_overall', 'comments']
notes_mix_std_fluor = notes_mix_std_fluor.query('polymer_overall in @poly_list')
notes_mix_std_fluor_ids = np.array(notes_mix_std_fluor['id'])


IR_spectra_mix_std = pd.read_csv('processed_data_new/data_ir_mix_std_rebecca.csv')
del IR_spectra_mix_std['Unnamed: 0']
#non_bg = IR_spectra['background'] == 'no_background'
IR_spectra_mix_std.columns = list(np.float64(IR_spectra_mix_std.columns[:-8])) + list(IR_spectra_mix_std.columns[-8:])

IR_spectra_mix_std = (IR_spectra_mix_std.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(IR_spectra_mix_std.shape[1]-3))+[-1,-3,-2]])

IR_spectra_mix_std_rest = IR_spectra_mix_std.loc[[x not in poly_list 
                                                  for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
#IR_spectra_mix_std = IR_spectra_mix_std.loc[[x in ['LDPE', 'PP', 'PSL'] for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
IR_spectra_mix_std = IR_spectra_mix_std.loc[[x in poly_list 
                                             for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
IR_spectra_mix_std.loc[IR_spectra_mix_std['polymer'] == 'LDPE','polymer'] = 'PE'
IR_spectra_mix_std = IR_spectra_mix_std.sort_values('id').reset_index(drop=True)


Raman_spectra_mix_std = pd.read_csv('processed_data_new/data_raman_mix_std_rebecca_specy.csv')
#notes_mix_std_rest = np.array(Raman_spectra_mix_std['comments'])
notes_mix_std = Raman_spectra_mix_std.iloc[:,-5:]

notes_mix_std = (notes_mix_std.
                 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
                 drop(['polymer_x', 'abbr'], axis=1).
                 rename({'polymer_y':'polymer_overall'},axis=1).
                 merge(poly_abbr_table, left_on='polymer_raman', right_on='abbr').
                 drop(['polymer_raman', 'abbr'], axis=1).
                 rename({'polymer':'polymer_raman'},axis=1).
                 merge(poly_abbr_table, left_on='polymer_ir', right_on='abbr').
                 drop(['polymer_ir', 'abbr'], axis=1).
                 rename({'polymer':'polymer_ir'},axis=1))

notes_mix_std.loc[notes_mix_std['polymer_overall'] == 'LDPE','polymer_overall'] = 'PE'
notes_mix_std.loc[notes_mix_std['polymer_raman'] == 'LDPE','polymer_raman'] = 'PE'
notes_mix_std.loc[notes_mix_std['polymer_ir'] == 'LDPE','polymer_ir'] = 'PE'


#notes_mix_std_rest[[x is np.nan for x in notes_mix_std_rest]] = ''
del Raman_spectra_mix_std['Unnamed: 0']
Raman_spectra_mix_std.columns = list(np.float64(Raman_spectra_mix_std.columns[:-7])) + list(Raman_spectra_mix_std.columns[-7:])
Raman_spectra_mix_std = (Raman_spectra_mix_std.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(Raman_spectra_mix_std.shape[1]-5))+[-1,-2]])

#poly_and_comments_list_rest = poly_and_comments_list.loc[[x not in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

notes_mix_std = notes_mix_std.sort_values('id').reset_index(drop=True)
Raman_spectra_mix_std = Raman_spectra_mix_std.sort_values('id').reset_index(drop=True)

Raman_spectra_mix_std_rest = Raman_spectra_mix_std.loc[[x not in poly_list 
                                                        for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
notes_mix_std_rest = notes_mix_std.loc[[x not in poly_list 
                                        for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

#poly_and_comments_list = poly_and_comments_list.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

notes_mix_std = notes_mix_std.loc[[x in poly_list 
                                   for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
Raman_spectra_mix_std = Raman_spectra_mix_std.loc[[x in poly_list 
                                                   for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

Raman_spectra_mix_std.loc[Raman_spectra_mix_std['polymer'] == 'LDPE','polymer'] = 'PE'

ids_ideal_set = [(x in poly_list) and (y in poly_list) and (z in poly_list) 
                 for x, y, z in zip(notes_mix_std['polymer_raman'], 
                                    notes_mix_std['polymer_ir'], 
                                    notes_mix_std['polymer_overall'])]

ids_ideal_set = np.logical_and([x not in notes_mix_std_fluor_ids 
                                for x in Raman_spectra_mix_std['id']],
                               ids_ideal_set)

Raman_spectra_mix_std['id'] = ["mix_std_"+str(x) for x in Raman_spectra_mix_std['id']]
IR_spectra_mix_std['id'] = ["mix_std_"+str(x) for x in IR_spectra_mix_std['id']]

Raman_spectra_mix_std_ideal = Raman_spectra_mix_std.loc[ids_ideal_set,:].reset_index(drop=True)
IR_spectra_mix_std_ideal = IR_spectra_mix_std.loc[ids_ideal_set,:].reset_index(drop=True)

Raman_spectra_mix_std_lq = Raman_spectra_mix_std.loc[~ids_ideal_set,:].reset_index(drop=True)
IR_spectra_mix_std_lq = IR_spectra_mix_std.loc[~ids_ideal_set,:].reset_index(drop=True)

Raman_spectra_mix_std_lq = Raman_spectra_mix_std_lq.query('id != "mix_std_223"').reset_index(drop=True)
IR_spectra_mix_std_lq = IR_spectra_mix_std_lq.query('id != "mix_std_223"').reset_index(drop=True)



poly_abbr_table = pd.read_excel('data_new/CCOPTIR_20240326_MixedMPsv7.3_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:9,-5:-3]
poly_abbr_table.columns = ['polymer', 'abbr']

poly_and_comments_list = pd.read_excel('data_new/CCOPTIR_20240326_MixedMPsv7.3_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:,9:13]

poly_list = list(np.unique(rebecca_data_raman['polymer'])) + ['LDPE']

notes_mix_fluor = pd.read_excel('data_new/MVCP_20241206_RP_AllCCOPTIRData_FluorescenceOnly.xlsx',
         sheet_name='MPs Only').iloc[:, [0,9,10,11,12]]
notes_mix_fluor.columns = ['id','polymer_ir', 'polymer_raman', 'polymer_overall', 'comments']
notes_mix_fluor = notes_mix_fluor.query('polymer_overall in @poly_list')
notes_mix_fluor_ids = np.array(notes_mix_fluor['id'])


IR_spectra_mix = pd.read_csv('processed_data_new/data_ir_mix_rebecca.csv')
del IR_spectra_mix['Unnamed: 0']
#non_bg = IR_spectra['background'] == 'no_background'
IR_spectra_mix.columns = list(np.float64(IR_spectra_mix.columns[:-8])) + list(IR_spectra_mix.columns[-8:])

IR_spectra_mix = (IR_spectra_mix.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(IR_spectra_mix.shape[1]-3))+[-1,-3,-2]])

IR_spectra_mix_rest = IR_spectra_mix.loc[[x not in poly_list 
                                          for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
#IR_spectra_mix = IR_spectra_mix.loc[[x in ['LDPE', 'PP', 'PSL'] for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
IR_spectra_mix = IR_spectra_mix.loc[[x in poly_list 
                                     for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
IR_spectra_mix.loc[IR_spectra_mix['polymer'] == 'LDPE','polymer'] = 'PE'
IR_spectra_mix = IR_spectra_mix.sort_values('id').reset_index(drop=True)


#Raman_spectra_mix = pd.read_csv('processed_data_new/data_raman_mix_rebecca.csv')
Raman_spectra_mix = pd.read_csv('processed_data_new/data_raman_mix_rebecca_specy.csv')
#notes_mix_rest = np.array(Raman_spectra_mix['comments'])
notes_mix = Raman_spectra_mix.iloc[:,-5:]

notes_mix = (notes_mix.
                 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
                 drop(['polymer_x', 'abbr'], axis=1).
                 rename({'polymer_y':'polymer_overall'},axis=1).
                 merge(poly_abbr_table, left_on='polymer_raman', right_on='abbr').
                 drop(['polymer_raman', 'abbr'], axis=1).
                 rename({'polymer':'polymer_raman'},axis=1).
                 merge(poly_abbr_table, left_on='polymer_ir', right_on='abbr').
                 drop(['polymer_ir', 'abbr'], axis=1).
                 rename({'polymer':'polymer_ir'},axis=1))

notes_mix.loc[notes_mix['polymer_overall'] == 'LDPE','polymer_overall'] = 'PE'
notes_mix.loc[notes_mix['polymer_raman'] == 'LDPE','polymer_raman'] = 'PE'
notes_mix.loc[notes_mix['polymer_ir'] == 'LDPE','polymer_ir'] = 'PE'


#notes_mix_rest[[x is np.nan for x in notes_mix_rest]] = ''
del Raman_spectra_mix['Unnamed: 0']
Raman_spectra_mix.columns = list(np.float64(Raman_spectra_mix.columns[:-7])) + list(Raman_spectra_mix.columns[-7:])
Raman_spectra_mix = (Raman_spectra_mix.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
 iloc[:,list(range(Raman_spectra_mix.shape[1]-5))+[-1,-2]])

#poly_and_comments_list_rest = poly_and_comments_list.loc[[x not in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

notes_mix = notes_mix.sort_values('id').reset_index(drop=True)
Raman_spectra_mix = Raman_spectra_mix.sort_values('id').reset_index(drop=True)

Raman_spectra_mix_rest = Raman_spectra_mix.loc[[x not in poly_list 
                                                for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)
notes_mix_rest = notes_mix.loc[[x not in poly_list 
                                for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

#poly_and_comments_list = poly_and_comments_list.loc[[x in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

notes_mix = notes_mix.loc[[x in poly_list 
                           for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)
Raman_spectra_mix = Raman_spectra_mix.loc[[x in poly_list 
                                           for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

Raman_spectra_mix.loc[Raman_spectra_mix['polymer'] == 'LDPE','polymer'] = 'PE'

ids_ideal_set = [(x in poly_list) and (y in poly_list) and (z in poly_list) 
                 for x, y, z in zip(notes_mix['polymer_raman'], notes_mix['polymer_ir'], notes_mix['polymer_overall'])]
ids_ideal_set = np.logical_and([x not in notes_mix_fluor_ids 
                                for x in Raman_spectra_mix['id']],
                               ids_ideal_set)

Raman_spectra_mix['id'] = ["mix_"+str(x) for x in Raman_spectra_mix['id']]
IR_spectra_mix['id'] = ["mix_"+str(x) for x in IR_spectra_mix['id']]

Raman_spectra_mix_ideal = Raman_spectra_mix.loc[ids_ideal_set,:].reset_index(drop=True)
IR_spectra_mix_ideal = IR_spectra_mix.loc[ids_ideal_set,:].reset_index(drop=True)

Raman_spectra_mix_lq = Raman_spectra_mix.loc[~ids_ideal_set,:].reset_index(drop=True)
IR_spectra_mix_lq = IR_spectra_mix.loc[~ids_ideal_set,:].reset_index(drop=True)



IR_spectra_all = pd.concat([IR_spectra_mix_std_ideal, IR_spectra_mix_ideal,
                            IR_spectra_mix_std_lq, IR_spectra_mix_lq], axis=0).reset_index(drop=True)

Raman_spectra_all = pd.concat([Raman_spectra_mix_std_ideal, Raman_spectra_mix_ideal,
                               Raman_spectra_mix_std_lq, Raman_spectra_mix_lq], 
                              axis=0).reset_index(drop=True)
#wn = np.array(Raman_spectra_all.columns[:-4])

(x_train_ir, 
 x_train_raman, 
 label_train_IR, 
 label_train_Raman, 
 x_ir, 
 x_raman, 
 label_ir, 
 label_raman, 
 label_table,
 ids_test) = process_data(rebecca_data_ir, 
                             rebecca_data_raman,   
                             IR_spectra_all,
                             Raman_spectra_all)


quality = np.array(['ideal']*(IR_spectra_mix_std_ideal.shape[0] + IR_spectra_mix_ideal.shape[0])+ 
                   ['low_ir']*(IR_spectra_mix_std_lq.shape[0] + IR_spectra_mix_lq.shape[0]))

smx_all_IR = 1-np.power(smx_nn(x_train_ir, x_ir, label_train_IR),1)
smx_all_Raman = 1-np.power(smx_nn(x_train_raman, x_raman, label_train_Raman),1)
score_list = [smx_all_IR, smx_all_Raman]

smx_all_IR_all = 1-np.power(smx_nn_all(x_train_ir, x_ir, label_train_IR),1)
smx_all_Raman_all = 1-np.power(smx_nn_all(x_train_raman, x_raman, label_train_Raman),1)
score_all_list = [smx_all_IR_all, smx_all_Raman_all]

print(np.nanmean(np.argmin(score_list[0], 1) == label_ir), 
      np.nanmean(np.argmin(score_list[1], 1) == label_raman))

print(np.nanmean(np.argmin(score_list[0][quality=='ideal'], 1) == label_ir[quality=='ideal']), 
      np.nanmean(np.argmin(score_list[1][quality=='ideal'], 1) == label_raman[quality=='ideal']))

print(np.nanmean(np.argmin(score_list[0][quality=='low_ir'], 1) == label_ir[quality=='low_ir']), 
      np.nanmean(np.argmin(score_list[1][quality=='low_ir'], 1) == label_raman[quality=='low_ir']))


# Data Experiments Figure 3

IR_spectra_all_ = IR_spectra_all.copy()
Raman_spectra_all_ = Raman_spectra_all.copy()
IR_spectra_all_['quality'] = quality
Raman_spectra_all_['quality'] = quality

IR_spectra_all_.to_csv('results_archv_2/data_fig_3_ir_all_processed_real.csv')
Raman_spectra_all_.to_csv('results_archv_2/data_fig_3_raman_all_processed_real.csv')

top_hqi_values = 1-pd.DataFrame(np.concatenate(score_list, axis=1))
top_hqi_values.columns = [x + ' (IR)' for x in label_table['polymer']] + [x + ' (Raman)' for x in label_table['polymer']]
top_hqi_values = pd.concat([top_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
top_hqi_values.to_csv('results_archv_2/data_fig_3_top_hqi_values_real.csv')

all_hqi_values = 1-pd.DataFrame(np.concatenate(score_all_list, axis=0).T)
all_hqi_values.columns = list(rebecca_data_ir['id']) + list(rebecca_data_raman['id'])
all_hqi_values = pd.concat([all_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
all_hqi_values.to_csv('results_archv_2/data_fig_3_all_hqi_values_real.csv')

rebecca_data_ir.to_csv('results_archv_2/data_fig_3_processed_reference_library_ir.csv')
rebecca_data_raman.to_csv('results_archv_2/data_fig_3_processed_reference_library_raman.csv')


#Figure 3a real low quality

alphas = [0.05]
N_sim = 100

(df_res_lis, 
 df_res_sd_lis,
 df_res_cov_lis, 
 df_res_cov_sd_lis, 
 df_res_all_mean_lis, 
 df_res_cov_all_mean_lis, 
 df_res_all_lis, 
 df_res_cov_all_lis)  = experiment_fig_3(score_list,                                      
                                   label_ir,                                   
                                   quality,
                                   alphas=alphas,
                                   N_sim=N_sim, 
                                   quant_prop=.2, 
                                   cal_prop=0.8, 
                                   seed_init=0)

                                   
labels = ['mvcp_ideal', 
          'view_IR_ideal', 
          'view_Raman_ideal', 
          'mvcp_low_ir', 
          'view_IR_low_ir', 
          'view_Raman_low_ir']

set_size = [np.array(df_res_all_mean_lis[0][x]) for x in labels]#df_res_all_lis[0]
#emp_conf_list.append(df_res_cov_lis)
    
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 8))
title = ['none', 'low', 'medium', 'high']




#labels = list(df_res_all_lis[0].keys())
colors = ['peachpuff', 'orange', 'tomato', 'lightblue', 'blue', 'darkblue']

ax.set_ylabel('set size')
ax.set_title('set size')

bplot = ax.boxplot(set_size,
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()


cov = [np.array(df_res_cov_all_mean_lis[0][x]) for x in labels]#df_res_all_lis[0]
#emp_conf_list.append(df_res_cov_lis)
    
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 8))
title = ['none', 'low', 'medium', 'high']




#labels = list(df_res_all_lis[0].keys())
colors = ['peachpuff', 'orange', 'tomato', 'lightblue', 'blue', 'darkblue']

ax.set_ylabel('confidence')
ax.set_title('confidence')

bplot = ax.boxplot(cov,
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()


set_size_df = pd.DataFrame(set_size).T
set_size_df.columns = ["set size "+x[:-3] if x[-2:]=="ir" else "set size "+x for x in labels]
cov_df = pd.DataFrame(cov).T
cov_df.columns = ["confidence "+x[:-3] if x[-2:]=="ir" else "confidence "+x for x in labels]
pd.concat([set_size_df,cov_df], axis=1).to_csv('results_archv_2/data_fig_3a_ideal_low_quality.csv')


#Figure 3a augmented low quality envelops
set_size_list = []
cov_list = []

N_sim_split = 10

for k in range(N_sim_split):
    noise_IR = 3#1
    noise_Raman = 2#0.5
    n_ideal_mix = IR_spectra_mix_ideal.shape[0]
    n_ideal_mix_std = IR_spectra_mix_std_ideal.shape[0]
    n_ideal = n_ideal_mix_std + n_ideal_mix

    np.random.seed(k)

    idx_mix_std, idx_mix = (np.linspace(0,1,n_ideal_mix_std), 
                            np.linspace(0,1,n_ideal_mix))

    np.random.shuffle(idx_mix_std)
    np.random.shuffle(idx_mix)

    idx_ideal_mix_std, idx_ideal_mix = (idx_mix_std< 1/3, 
                                        idx_mix< 1/3)

    idx_low_ir_mix_std, idx_low_ir_mix = (np.logical_and(idx_mix_std > 1/3,
                                                         idx_mix_std < 2/3),
                                          np.logical_and(idx_mix > 1/3,
                                                         idx_mix < 2/3))

    idx_low_raman_mix_std, idx_low_raman_mix = (idx_mix_std> 2/3, 
                                                idx_mix > 2/3)

    np.random.shuffle(idx_ideal_mix_std)
    np.random.shuffle(idx_ideal_mix)

    np.random.shuffle(idx_low_ir_mix_std)
    np.random.shuffle(idx_low_ir_mix)

    np.random.shuffle(idx_ideal_mix_std)
    np.random.shuffle(idx_ideal_mix_std)


    IR_spectra_mix_std_lq_sint_ir = IR_spectra_mix_std_ideal.loc[idx_low_ir_mix_std,:].copy()
    IR_spectra_mix_std_lq_sint_raman = IR_spectra_mix_std_ideal.loc[idx_low_raman_mix_std,:].copy()

    IR_spectra_mix_lq_sint_ir = IR_spectra_mix_ideal.loc[idx_low_ir_mix,:].copy()
    IR_spectra_mix_lq_sint_raman = IR_spectra_mix_ideal.loc[idx_low_raman_mix,:].copy()

    Raman_spectra_mix_std_lq_sint_raman = Raman_spectra_mix_std_ideal.loc[idx_low_raman_mix_std,:].copy()
    Raman_spectra_mix_std_lq_sint_ir = Raman_spectra_mix_std_ideal.loc[idx_low_ir_mix_std,:].copy()

    Raman_spectra_mix_lq_sint_raman = Raman_spectra_mix_ideal.loc[idx_low_raman_mix,:].copy()
    Raman_spectra_mix_lq_sint_ir = Raman_spectra_mix_ideal.loc[idx_low_ir_mix,:].copy()

    #noise_IR_mix_std = noise_IR*np.random.normal(0,1,IR_spectra_mix_std_ideal.iloc[:,:-8].shape)
    #noise_IR_mix = noise_IR*np.random.normal(0,1,IR_spectra_mix_ideal.iloc[:,:-8].shape)

    noise_IR_mix_std = noise_IR*np.random.normal(0,
                                                 np.array(np.std(IR_spectra_mix_std_ideal.iloc[:,:-8],0)), 
                                                 size=IR_spectra_mix_std_lq_sint_ir.iloc[:,:-8].shape)

    noise_IR_mix = noise_IR*np.random.normal(0,
                                             np.array(np.std(IR_spectra_mix_ideal.iloc[:,:-8],0)), 
                                             size=IR_spectra_mix_lq_sint_ir.iloc[:,:-8].shape)

    noise_Raman_mix_std_ir = noise_Raman*np.random.normal(0,
                                                 np.array(np.std(Raman_spectra_mix_std_ideal.iloc[:,:-4],0)), 
                                                 size=Raman_spectra_mix_std_lq_sint_ir.iloc[:,:-4].shape)

    noise_Raman_mix_ir = noise_Raman*np.random.normal(0,
                                             np.array(np.std(Raman_spectra_mix_ideal.iloc[:,:-4],0)), 
                                             size=Raman_spectra_mix_lq_sint_ir.iloc[:,:-4].shape)
    
    noise_Raman_mix_std = noise_Raman*np.random.normal(0,
                                                 np.array(np.std(Raman_spectra_mix_std_ideal.iloc[:,:-4],0)), 
                                                 size=Raman_spectra_mix_std_lq_sint_raman.iloc[:,:-4].shape)

    noise_Raman_mix = noise_Raman*np.random.normal(0,
                                             np.array(np.std(Raman_spectra_mix_ideal.iloc[:,:-4],0)), 
                                             size=Raman_spectra_mix_lq_sint_raman.iloc[:,:-4].shape)


    IR_spectra_mix_std_lq_sint_ir.iloc[:,:-8] = IR_spectra_mix_std_lq_sint_ir.iloc[:,:-8] + noise_IR_mix_std
    IR_spectra_mix_lq_sint_ir.iloc[:,:-8] = IR_spectra_mix_lq_sint_ir.iloc[:,:-8] + noise_IR_mix
    
    
    #Raman_spectra_mix_std_lq_sint_ir.iloc[:,:-4] = Raman_spectra_mix_std_lq_sint_ir.iloc[:,:-4] + noise_Raman_mix_std_ir
    #Raman_spectra_mix_lq_sint_ir.iloc[:,:-4] = Raman_spectra_mix_lq_sint_ir.iloc[:,:-4] + noise_Raman_mix_ir
    
    Raman_spectra_mix_std_lq_sint_raman.iloc[:,:-4] = Raman_spectra_mix_std_lq_sint_raman.iloc[:,:-4] + noise_Raman_mix_std
    Raman_spectra_mix_lq_sint_raman.iloc[:,:-4] = Raman_spectra_mix_lq_sint_raman.iloc[:,:-4] + noise_Raman_mix


    IR_spectra_all = pd.concat([IR_spectra_mix_std_ideal.loc[idx_ideal_mix_std,:], 
                                IR_spectra_mix_ideal.loc[idx_ideal_mix,:],
                                IR_spectra_mix_std_lq_sint_ir, 
                                IR_spectra_mix_lq_sint_ir,    
                                IR_spectra_mix_std_lq_sint_raman, 
                                IR_spectra_mix_lq_sint_raman], axis=0).reset_index(drop=True)

    Raman_spectra_all = pd.concat([Raman_spectra_mix_std_ideal.loc[idx_ideal_mix_std,:], 
                                   Raman_spectra_mix_ideal.loc[idx_ideal_mix,:],
                                   Raman_spectra_mix_std_lq_sint_ir, 
                                   Raman_spectra_mix_lq_sint_ir,
                                   Raman_spectra_mix_std_lq_sint_raman, 
                                   Raman_spectra_mix_lq_sint_raman], axis=0).reset_index(drop=True)

    wn = np.array(Raman_spectra_all.columns[:-4])

    (x_train_ir, 
     x_train_raman, 
     label_train_IR, 
     label_train_Raman, 
     x_ir, 
     x_raman, 
     label_ir, 
     label_raman, 
     label_table,
     ids_test) = process_data(rebecca_data_ir, 
                                 rebecca_data_raman, 
                                 IR_spectra_all,
                                 Raman_spectra_all)
                                 

    #x_ir_rest, x_raman_rest, y_ir_rest = process_data_2(IR_spectra_mix_rest, Raman_spectra_mix_rest)


    quality = np.array(['ideal']*(np.sum(idx_ideal_mix_std)+np.sum(idx_ideal_mix))+ 
                       ['low_ir']*(np.sum(idx_low_ir_mix_std)+np.sum(idx_low_ir_mix))+ 
                       ['low_raman']*(np.sum(idx_low_raman_mix_std)+np.sum(idx_low_raman_mix)))

    smx_all_IR = 1-np.power(smx_nn(x_train_ir, x_ir, label_train_IR),1)
    smx_all_Raman = 1-np.power(smx_nn(x_train_raman, x_raman, label_train_Raman),1)
    score_list = [smx_all_IR, smx_all_Raman]

    #smx_all_IR_all = 1-np.power(smx_nn_all(x_train_ir, x_ir, label_train_IR),1)
    #smx_all_Raman_all = 1-np.power(smx_nn_all(x_train_raman, x_raman, label_train_Raman),1)
    #score_list_all = [smx_all_IR_all, smx_all_Raman_all]


    print('all',np.nanmean(np.argmin(score_list[0], 1) == label_ir), np.nanmean(np.argmin(score_list[1], 1) == label_raman))
    print('ideal',np.nanmean(np.argmin(score_list[0][quality=='ideal'], 1) == label_ir[quality=='ideal']), np.nanmean(np.argmin(score_list[1][quality=='ideal'], 1) == label_raman[quality=='ideal']))
    print('low_ir',np.nanmean(np.argmin(score_list[0][quality=='low_ir'], 1) == label_ir[quality=='low_ir']), np.nanmean(np.argmin(score_list[1][quality=='low_ir'], 1) == label_raman[quality=='low_ir']))
    print('low_raman',np.nanmean(np.argmin(score_list[0][quality=='low_raman'], 1) == label_ir[quality=='low_raman']), np.nanmean(np.argmin(score_list[1][quality=='low_raman'], 1) == label_raman[quality=='low_raman']))

    (df_res_lis, 
     df_res_sd_lis,
     df_res_cov_lis, 
     df_res_cov_sd_lis, 
     df_res_all_mean_lis, 
     df_res_cov_all_mean_lis, 
     df_res_all_lis, 
     df_res_cov_all_lis)  = experiment_fig_3(score_list,                                      
                                       label_ir,                                   
                                       quality,
                                       alphas=alphas,
                                       N_sim=N_sim, 
                                       quant_prop=.2, 
                                       cal_prop=0.8, 
                                       seed_init=0)

    labels = ['mvcp_ideal', 
              'view_IR_ideal', 
              'view_Raman_ideal', 
              'mvcp_low_ir', 
              'view_IR_low_ir', 
              'view_Raman_low_ir', 
              'mvcp_low_raman', 
              'view_IR_low_raman', 
              'view_Raman_low_raman']

    set_size = [np.array(df_res_all_mean_lis[0][x]) for x in labels]
    cov = [np.array(df_res_cov_all_mean_lis[0][x]) for x in labels]

    set_size_list.append(set_size)
    cov_list.append(cov)

IR_spectra_all.to_csv('results_archv_2/data_fig_3_ir_all_processed_augmented.csv')
Raman_spectra_all.to_csv('results_archv_2/data_fig_3_raman_all_processed_augmented.csv')

top_hqi_values = 1-pd.DataFrame(np.concatenate(score_list, axis=1))
top_hqi_values.columns = [x + ' (IR)' for x in label_table['polymer']] + [x + ' (Raman)' for x in label_table['polymer']]
top_hqi_values = pd.concat([top_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
top_hqi_values.to_csv('results_archv_2/data_fig_3_top_hqi_values_augmented.csv')

all_hqi_values = 1-pd.DataFrame(np.concatenate(score_all_list, axis=0).T)
all_hqi_values.columns = list(rebecca_data_ir['id']) + list(rebecca_data_raman['id'])
all_hqi_values = pd.concat([all_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
all_hqi_values.to_csv('results_archv_2/data_fig_3_all_hqi_values_augmented.csv')


labels = ['mvcp_ideal', 
          'view_IR_ideal', 
          'view_Raman_ideal', 
          'mvcp_low_ir', 
          'view_IR_low_ir', 
          'view_Raman_low_ir', 
          'mvcp_low_raman', 
          'view_IR_low_raman', 
          'view_Raman_low_raman']

#set_size = [np.array(df_res_all_mean_lis[0][x]) for x in labels]#df_res_all_lis[0]
#emp_conf_list.append(df_res_cov_lis)

set_size = list(np.mean(set_size_list,0))
cov = list(np.mean(cov_list,0))
    
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 8))
title = ['none', 'low', 'medium', 'high']

#labels = list(df_res_all_lis[0].keys())
colors = ['peachpuff', 'orange', 'tomato', 
          'lightblue', 'blue', 'darkblue',
          'lightgreen', 'green', 'darkgreen']

ax.set_ylabel('set size')
ax.set_title('set size')

bplot = ax.boxplot(set_size,
                   #list(np.array(set_size_list)[:,:,0].T),
                   #list(np.transpose(np.array(set_size_list),(0,2,1)).reshape(2500,9).T),
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.xticks(rotation=45, ha='right')
plt.show()


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(10, 8))
title = ['none', 'low', 'medium', 'high']




#labels = list(df_res_all_lis[0].keys())
colors = ['peachpuff', 'orange', 'tomato', 
          'lightblue', 'blue', 'darkblue',
          'lightgreen', 'green', 'darkgreen']

ax.set_ylabel('set size')
ax.set_title('set size')

bplot = ax.boxplot(cov,
                   #list(np.array(cov_list)[:,:,0].T),
                   #list(np.transpose(np.array(cov_list),(0,2,1)).reshape(2500,9).T),
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.xticks(rotation=45, ha='right')
plt.show()


set_size_df = pd.DataFrame(set_size).T
set_size_df.columns = ["set size "+x for x in labels]
cov_df = pd.DataFrame(cov).T
cov_df.columns = ["confidence "+x for x in labels]

pd.concat([set_size_df,cov_df], axis=1).to_csv('results_archv_2/data_fig_3a_ideal_augmented.csv')


#Figure 3b real low quality envelopes

IR_spectra_all = pd.concat([IR_spectra_mix_std_ideal, IR_spectra_mix_ideal,
                            IR_spectra_mix_std_lq, IR_spectra_mix_lq], axis=0).reset_index(drop=True)

Raman_spectra_all = pd.concat([Raman_spectra_mix_std_ideal, Raman_spectra_mix_ideal,
                               Raman_spectra_mix_std_lq, Raman_spectra_mix_lq], axis=0).reset_index(drop=True)

wn = np.array(Raman_spectra_all.columns[:-4])

(x_train_ir, 
 x_train_raman,
 label_train_IR,
 label_train_Raman, 
 x_ir, 
 x_raman, 
 label_ir, 
 label_raman, 
 label_table,
 ids_test) = process_data(rebecca_data_ir, 
                          rebecca_data_raman, 
                          IR_spectra_all,
                          Raman_spectra_all)

#x_ir_rest, x_raman_rest, y_ir_rest = process_data_2(IR_spectra_mix_rest, Raman_spectra_mix_rest)

quality = np.array(['ideal']*(IR_spectra_mix_std_ideal.shape[0] + IR_spectra_mix_ideal.shape[0])+ 
                   ['low_ir']*(IR_spectra_mix_std_lq.shape[0] + IR_spectra_mix_lq.shape[0]))

smx_all_IR = 1-np.power(smx_nn(x_train_ir, x_ir, label_train_IR),1)
smx_all_Raman = 1-np.power(smx_nn(x_train_raman, x_raman, label_train_Raman),1)
score_list = [smx_all_IR, smx_all_Raman]

print(np.nanmean(np.argmin(score_list[0], 1) == label_ir), np.nanmean(np.argmin(score_list[1], 1) == label_raman))
print(np.nanmean(np.argmin(score_list[0][quality=='ideal'], 1) == label_ir[quality=='ideal']), np.nanmean(np.argmin(score_list[1][quality=='ideal'], 1) == label_raman[quality=='ideal']))
print(np.nanmean(np.argmin(score_list[0][quality=='low_ir'], 1) == label_ir[quality=='low_ir']), np.nanmean(np.argmin(score_list[1][quality=='low_ir'], 1) == label_raman[quality=='low_ir']))


bad_examples = [360, 438, 414, 465, 417, 421]


for id_ in bad_examples:
    example_id = pd.concat([IR_spectra_mix_std, IR_spectra_mix], axis=0).reset_index(drop=True).iloc[id_,-2]
    score_example = [score_list[0][id_:id_+1], score_list[1][id_:id_+1]] 
    
    
    (ps_list, 
     ps_ir_list, 
     ps_raman_list, 
     env_list, 
     cal_smx_arr, 
     cal_idx_list, 
     q_ir_list, 
     q_raman_list) = envelop(score_list, 
                             label_ir, 
                             score_example,
                             alphas=[0.05],
                             label_table=label_table,
                             N_sim=1, 
                             quant_prop=.2, 
                             cal_prop=0.8, 
                             seed_init=10)
    
    hqi_ex = pd.DataFrame({'polymer':label_table['polymer'],
              'HQI (IR)':1-score_example[0][0], 
              'HQI (Raman)':1-score_example[1][0]})
    hqi_ex.to_csv('results_archv_2/data_fig_3b_hqi_example_id_'+str(example_id)+'.csv')

    pred_sets_ex = pd.concat([(pd.DataFrame(ps_list).T), 
                              (pd.DataFrame(ps_ir_list).T),
                              (pd.DataFrame(ps_raman_list).T)],axis=1).reset_index()
    
    pred_sets_ex['index'] = 100 - pred_sets_ex['index']
    pred_sets_ex.columns = ['confidence', 'MVCP (IR + Raman)', 'CP (IR)', 'CP (Raman)']
    pred_sets_ex.to_csv('results_archv_2/data_fig_3b_pred_sets_id_'+str(example_id)+'.csv')
    
    _ = plt.show()
    _ = plt.plot(IR_spectra_all.iloc[id_:id_+1,:-8].T)
    _ = plt.plot(Raman_spectra_all.iloc[id_:id_+1,:-8].T-5)
    _ = plt.show()
    


h_ir = plt.hist(1-cal_smx_arr[0],bins=20, density=False)
_ = plt.vlines(x= 1-q_ir_list[5][0],
                ymin=0,
                ymax=65,
                colors='red',
                alpha=1,
                linestyles='solid',
                label='95 IR Quantile')
_ = plt.legend()
_ = plt.show()

h_raman = plt.hist(1-cal_smx_arr[1],bins=h_ir[1], density=False)
_ = plt.vlines(x=1- q_raman_list[5][0],
                ymin=0,
                ymax=85,
                colors='red',
                alpha=1,
                linestyles='solid',
                label='95 IR Quantile')
_ = plt.legend()
_ = plt.show()

pd.DataFrame({'HQI':h_ir[1][:-1], 
              '# IR':h_ir[0],
              '# Raman':h_raman[0]
              }).to_csv('results_archv_2/data_fig_3b_histograms.csv')


for id_ in bad_examples:
    score_example = [score_list[0][id_:id_+1], score_list[1][id_:id_+1]] 
    plt.scatter(1-cal_smx_arr[0],
                1-cal_smx_arr[1], s=2)
    
    plt.scatter(1-score_example[0],
                1-score_example[1], s=10)
    
    for i in range(6):
        plt.text(1-score_example[0][0, i], 
                 1-score_example[1][0, i], label_table.iloc[i,1], fontsize=9, ha='right')
        
    env_data_list = []
    for env in env_list:
        env_data = pd.DataFrame(1-np.array(env_list[env][0]).T)
        env_data.columns = ['x - '+str(100-env), 'y - '+str(100-env)]
        _ = plt.plot(env_data['x - '+str(100-env)],
                     env_data['y - '+str(100-env)], label=str(100-env))
        env_data_list.append(env_data)
        
    _ = plt.vlines(x= 1-q_ir_list[5][0],
                    ymin=.6,
                    ymax=1,
                    colors='r',
                    alpha=.5,
                    linestyles='solid',
                    label='95 IR Quantile')
    
    _ = plt.hlines(y= 1-q_raman_list[5][0],
                                xmin=.6,
                                xmax=1,
                                colors='purple',
                                alpha=.5,
                                linestyles='solid',
                                label='95 Raman Quantile')
    
    _ = plt.legend()
    _ = plt.show()

cal_data = pd.DataFrame(1-cal_smx_arr.T)
cal_data.columns = ['IR', 'Raman']
cal_data.to_csv('results_archv_2/data_fig_3b_calibration.csv')

env_data = pd.concat(env_data_list,axis=1)
env_data.to_csv('results_archv_2/data_fig_3b_envelopes.csv')
