#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 22:35:23 2025

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
import time
from datetime import date

# Reference Libraries

rebecca_data_ir = pd.read_csv('processed_data_new/rebecca_data_ir_final.csv')
del rebecca_data_ir['Unnamed: 0']
rebecca_data_ir.columns = list(np.float64(rebecca_data_ir.columns[:-4])) + list(rebecca_data_ir.columns[-4:])

rebecca_data_raman = pd.read_csv('processed_data_new/rebecca_data_raman_final_specy.csv')
del rebecca_data_raman['Unnamed: 0']
rebecca_data_raman.columns = list(np.float64(rebecca_data_raman.columns[:-4])) + list(rebecca_data_raman.columns[-4:])




poly_abbr_table = pd.read_excel('data_new/CCOPTIR_20240604_MixedMPsStdsv2_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:9,-5:-3]
poly_abbr_table.columns = ['polymer', 'abbr']

poly_and_comments_list = pd.read_excel('data_new/CCOPTIR_20240604_MixedMPsStdsv2_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:,9:13]

poly_list = list(np.unique(rebecca_data_raman['polymer'])) + ['LDPE']

IR_spectra_mix_std = pd.read_csv('processed_data_new/data_ir_mix_std_rebecca.csv')
del IR_spectra_mix_std['Unnamed: 0']
#non_bg = IR_spectra['background'] == 'no_background'
IR_spectra_mix_std.columns = list(np.float64(IR_spectra_mix_std.columns[:-8])) + list(IR_spectra_mix_std.columns[-8:])

IR_spectra_mix_std = (IR_spectra_mix_std.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(IR_spectra_mix_std.shape[1]-3))+[-1,-3,-2]])

IR_spectra_mix_std_rest = IR_spectra_mix_std.loc[[x not in poly_list for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
#IR_spectra_mix_std = IR_spectra_mix_std.loc[[x in ['LDPE', 'PP', 'PSL'] for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
IR_spectra_mix_std = IR_spectra_mix_std.loc[[x in poly_list for x in IR_spectra_mix_std['polymer']],:].reset_index(drop=True)
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

Raman_spectra_mix_std_rest = Raman_spectra_mix_std.loc[[x not in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
notes_mix_std_rest = notes_mix_std.loc[[x not in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

#poly_and_comments_list = poly_and_comments_list.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

notes_mix_std = notes_mix_std.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
Raman_spectra_mix_std = Raman_spectra_mix_std.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

Raman_spectra_mix_std.loc[Raman_spectra_mix_std['polymer'] == 'LDPE','polymer'] = 'PE'

#ids_ideal_set = [(x in poly_list) and (y in poly_list) and (z in poly_list) for x, y, z in zip(notes_mix_std['polymer_raman'], notes_mix_std['polymer_ir'], notes_mix_std['polymer_overall'])]
#Raman_spectra_mix_std = Raman_spectra_mix_std.loc[ids_ideal_set,:].reset_index(drop=True)
#IR_spectra_mix_std = IR_spectra_mix_std.loc[ids_ideal_set,:].reset_index(drop=True)

Raman_spectra_mix_std = Raman_spectra_mix_std.query('id != 223').reset_index(drop=True)
IR_spectra_mix_std = IR_spectra_mix_std.query('id != 223').reset_index(drop=True)

Raman_spectra_mix_std['id'] = ["mix_std_"+str(x) for x in Raman_spectra_mix_std['id']]
IR_spectra_mix_std['id'] = ["mix_std_"+str(x) for x in IR_spectra_mix_std['id']]

poly_abbr_table = pd.read_excel('data_new/CCOPTIR_20240326_MixedMPsv7.3_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:9,-5:-3]
poly_abbr_table.columns = ['polymer', 'abbr']

poly_and_comments_list = pd.read_excel('data_new/CCOPTIR_20240326_MixedMPsv7.3_RP_mIRage_allccdata_manualsort.xlsx',
         sheet_name='Other cc data and sort').iloc[:,9:13]

poly_list = list(np.unique(rebecca_data_raman['polymer'])) + ['LDPE']

IR_spectra_mix = pd.read_csv('processed_data_new/data_ir_mix_rebecca.csv')
del IR_spectra_mix['Unnamed: 0']
#non_bg = IR_spectra['background'] == 'no_background'
IR_spectra_mix.columns = list(np.float64(IR_spectra_mix.columns[:-8])) + list(IR_spectra_mix.columns[-8:])

IR_spectra_mix = (IR_spectra_mix.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(IR_spectra_mix.shape[1]-3))+[-1,-3,-2]])

IR_spectra_mix_rest = IR_spectra_mix.loc[[x not in poly_list for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
#IR_spectra_mix = IR_spectra_mix.loc[[x in ['LDPE', 'PP', 'PSL'] for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
IR_spectra_mix = IR_spectra_mix.loc[[x in poly_list for x in IR_spectra_mix['polymer']],:].reset_index(drop=True)
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

Raman_spectra_mix_rest = Raman_spectra_mix.loc[[x not in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)
notes_mix_rest = notes_mix.loc[[x not in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

#poly_and_comments_list = poly_and_comments_list.loc[[x in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

notes_mix = notes_mix.loc[[x in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)
Raman_spectra_mix = Raman_spectra_mix.loc[[x in poly_list for x in Raman_spectra_mix['polymer']],:].reset_index(drop=True)

Raman_spectra_mix.loc[Raman_spectra_mix['polymer'] == 'LDPE','polymer'] = 'PE'

Raman_spectra_mix['id'] = ["mix_"+str(x) for x in Raman_spectra_mix['id']]
IR_spectra_mix['id'] = ["mix_"+str(x) for x in IR_spectra_mix['id']]

#ids_ideal_set = [(x in poly_list) and (y in poly_list) and (z in poly_list) for x, y, z in zip(notes_mix['polymer_raman'], notes_mix['polymer_ir'], notes_mix['polymer_overall'])]
#Raman_spectra_mix = Raman_spectra_mix.loc[ids_ideal_set,:].reset_index(drop=True)
#IR_spectra_mix = IR_spectra_mix.loc[ids_ideal_set,:].reset_index(drop=True)


IR_spectra_all = pd.concat([IR_spectra_mix_std, IR_spectra_mix], axis=0).reset_index(drop=True)
Raman_spectra_all = pd.concat([Raman_spectra_mix_std, Raman_spectra_mix], axis=0).reset_index(drop=True)

lower_range_raman_1 = 600
upper_range_raman_1 = 1850
lower_range_raman_2 = 2750
upper_range_raman_2 = 3500

wn_raman = np.array(Raman_spectra_all.columns[:-4])

idx_raman = list(np.logical_or(np.logical_and(wn_raman >lower_range_raman_1,
                           wn_raman <upper_range_raman_1
                           ),
                 np.logical_and(wn_raman >lower_range_raman_2,
                           wn_raman <upper_range_raman_2
                           ))) + [True]*4

(x_train_ir, x_train_raman, 
 label_train_IR, label_train_Raman, 
 x_ir, x_raman, 
 label_ir, label_raman, 
 label_table, ids_test) = process_data(rebecca_data_ir,                                            
                                       rebecca_data_raman.loc[:, idx_raman],                                               
                                       IR_spectra_all,                                               
                                       Raman_spectra_all.loc[:, idx_raman])


# Data Experiments Figure 2
smx_all_IR = 1-np.power(smx_nn(x_train_ir, x_ir, label_train_IR, tau=1.4427),1)
smx_all_Raman = 1-np.power(smx_nn(x_train_raman, x_raman, label_train_Raman, tau=1.4427),1)
#smx_all_IR = 1-np.power(smx_pcc(x_train_ir, x_ir, label_train_IR),1)
#smx_all_Raman = 1-np.power(smx_pcc(x_train_raman, x_raman, label_train_Raman),2)
score_list = [smx_all_IR, smx_all_Raman]

smx_all_IR_all = 1-np.power(smx_nn_all(x_train_ir, x_ir, label_train_IR, tau=1.4427),1)
smx_all_Raman_all = 1-np.power(smx_nn_all(x_train_raman, x_raman, label_train_Raman, tau=1.4427),1)
score_all_list = [smx_all_IR_all, smx_all_Raman_all]

# Get the current date
todays_date = date.today()
year = str(todays_date.year)
if todays_date.month < 10:
    month = '0'+str(todays_date.month)
else:
    month = str(todays_date.month)

if todays_date.day < 10:
    day = '0'+str(todays_date.day)
else:
    day = str(todays_date.day)

current_time = year+month+day
directory_name = 'results_final/'+current_time
if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
    
IR_spectra_all.to_csv(directory_name+'/data_fig_1-3_ir_all_processed.csv')
Raman_spectra_all.to_csv(directory_name+'/data_fig_1-3_raman_all_processed.csv')

top_hqi_values = 1-pd.DataFrame(np.concatenate(score_list, axis=1))
top_hqi_values.columns = [x + ' (IR)' for x in label_table['polymer']] + [x + ' (Raman)' for x in label_table['polymer']]
top_hqi_values = pd.concat([top_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
top_hqi_values.to_csv(directory_name+'/data_fig_1-3_top_hqi_values.csv')

all_hqi_values = 1-pd.DataFrame(np.concatenate(score_all_list, axis=0).T)
all_hqi_values.columns = list(rebecca_data_ir['id']) + list(rebecca_data_raman['id'])
all_hqi_values = pd.concat([all_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)
all_hqi_values.to_csv(directory_name+'/data_fig_1-3_all_hqi_values.csv')

rebecca_data_ir.to_csv(directory_name+'/data_fig_1-3_processed_reference_library_ir.csv')
rebecca_data_raman.to_csv(directory_name+'/data_fig_1-3_processed_reference_library_raman.csv')

# Figure 1

id_ = 519#116
example_id = IR_spectra_all.iloc[id_,-2]
score_example = [score_list[0][id_:id_+1], score_list[1][id_:id_+1]]

(ps_list, 
 ps_ir_list, 
 ps_raman_list, 
 env_list, 
 cal_smx_arr, 
 cal_idx_list, 
 q_ir_list, 
 q_raman_list) = envelop(score_list, label_ir, score_example,
                         alphas=[0.40,.20, .10, 0.05, 0.02],
                         label_table=label_table,
                         N_sim=1, 
                         quant_prop=.2, 
                         cal_prop=0.8, 
                         seed_init=0)


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

q_ir_list, q_raman_list
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


env_data = pd.concat(env_data_list,axis=1)
env_data.to_csv(directory_name+'/data_fig_1_envelopes.csv')

hqi_ex = pd.DataFrame({'polymer':label_table['polymer'],
                       'HQI (IR)':1-score_example[0][0], 
                       'HQI (Raman)':1-score_example[1][0]})
hqi_ex.to_csv(directory_name+'/data_fig_1_hqi_example_id_'+str(example_id)+'.csv')

pred_sets_ex = pd.concat([(pd.DataFrame(ps_list).T), 
                          (pd.DataFrame(ps_ir_list).T),
                          (pd.DataFrame(ps_raman_list).T)],axis=1).reset_index()
pred_sets_ex['index'] = 100 - pred_sets_ex['index']
pred_sets_ex.columns = ['confidence', 'MVCP (IR + Raman)', 'CP (IR)', 'CP (Raman)']
pred_sets_ex.to_csv(directory_name+'/data_fig_1_pred_sets_id_'+str(example_id)+'.csv')


# Figure 2

idx_quant,idx = cal_idx_list[5][0]
label_cal = label_ir[~idx_quant][idx]

cal_data = pd.DataFrame(1-cal_smx_arr.T)
cal_data.columns = ['IR', 'Raman']
cal_data['polymer'] = list(label_table.iloc[label_cal, 1])
cal_data.to_csv(directory_name+'/data_fig_2_calibration.csv')

bins_min = np.min([np.min(1-cal_smx_arr[0]), np.min(1-cal_smx_arr[1])])
bins_max = np.max([np.max(1-cal_smx_arr[0]), np.max(1-cal_smx_arr[1])])
bins_list = np.linspace(bins_min, bins_max, 20)


for i in np.unique(label_cal):
    h_ir = plt.hist(cal_data.iloc[np.where(label_cal==i)[0],0],#1-cal_smx_arr[0],
                    bins=bins_list, 
                    density=False)
    
    _ = plt.vlines(x= 1-q_ir_list[5][0],
                    ymin=0,
                    ymax=10,
                    colors='red',
                    alpha=1,
                    linestyles='solid',
                    label='95 IR Quantile')
    _ = plt.legend()
    _ = plt.show()
    
    h_raman = plt.hist(cal_data.iloc[np.where(label_cal==i)[0],1],#1-cal_smx_arr[1],
                       bins=bins_list, 
                       density=False)
    
    _ = plt.vlines(x=1- q_raman_list[5][0],
                    ymin=0,
                    ymax=10,
                    colors='red',
                    alpha=1,
                    linestyles='solid',
                    label='95 IR Quantile')
    _ = plt.legend()
    _ = plt.show()
    
    pd.DataFrame({'HQI beg bin':h_ir[1][:-1], 
                  'HQI end bin':h_ir[1][1:], 
                  '# IR':h_ir[0],
                  '# Raman':h_raman[0]
                  }).to_csv(directory_name+'/data_fig_2_histograms_'+label_table.iloc[i,1]+'.csv')

h_ir = plt.hist(cal_data.iloc[:,0],#1-cal_smx_arr[0],
                bins=bins_list, 
                density=False)

_ = plt.vlines(x= 1-q_ir_list[5][0],
                ymin=0,
                ymax=10,
                colors='red',
                alpha=1,
                linestyles='solid',
                label='95 IR Quantile')
_ = plt.legend()
_ = plt.show()

h_raman = plt.hist(cal_data.iloc[:,1],#1-cal_smx_arr[1],
                   bins=bins_list, 
                   density=False)

_ = plt.vlines(x=1- q_raman_list[5][0],
                ymin=0,
                ymax=10,
                colors='red',
                alpha=1,
                linestyles='solid',
                label='95 IR Quantile')
_ = plt.legend()
_ = plt.show()

pd.DataFrame({'HQI beg bin':h_ir[1][:-1], 
              'HQI end bin':h_ir[1][1:], 
              '# IR':h_ir[0],
              '# Raman':h_raman[0]
              }).to_csv(directory_name+'/data_fig_2_histograms.csv')

q_raman = np.array([q_raman_list[k][0] for k in q_raman_list.keys()])      
q_ir = np.array([q_ir_list[k][0] for k in q_ir_list.keys()])

quantiles = pd.DataFrame({'HQI Quantile (IR)':1-q_ir,
                            'HQI Quantile (Raman)':1-q_raman,
                            'Confidence':100-np.array([k for k in q_raman_list.keys()])
                            })
quantiles.to_csv(directory_name+'/data_fig_2_quantiles.csv')

# Figure 3a

alphas = np.arange(0,.2,.01)
N_sim = 100


(df_res_lis, 
 df_res_sd_lis, 
 df_res_cov_lis, 
 df_res_cov_sd_lis, 
 df_res_all_mean_lis, 
 df_res_cov_all_mean_lis, 
 df_res_all_lis, 
 df_res_cov_all_lis,
 df_res_labels_all,
 df_res_pred_set_all_lis)  = experiment_fig_1_2(score_list,                            
                                           label_ir,                                       
                                           alphas,
                                           N_sim, 
                                           quant_prop=.2,   
                                           cal_prop=0.8, 
                                           seed_init=0)


df_res_all = pd.concat(df_res_lis, axis=0)
df_res_all['confidence'] = (1-alphas)*100
_ = plt.plot(np.round(df_res_all, 2).set_index('confidence'))
_ = plt.legend(['view_IR', 'view_Raman', 'mvcp'])
_ = plt.xlabel("Confidence")
_ = plt.ylabel("Prediction Sets Size")
_ = plt.show()

df_res_all = pd.concat([pd.concat(df_res_lis, axis=0), pd.concat(df_res_sd_lis, axis=0)],axis=1)
df_res_all.columns = ["set size "+x +" (mean)" for x in df_res_all.columns[:3]] + ["set size "+x+" (standard deviation)" for x in df_res_all.columns[3:]]
df_res_all['confidence'] = (1-alphas)*100
df_res_all

df_res_all.to_csv(directory_name+'/data_fig_3a.csv')



# Figure 3b

(df_res_lis, 
 df_res_sd_lis, 
 df_res_cov_lis, 
 df_res_cov_sd_lis, 
 df_res_all_mean_lis, 
 df_res_cov_all_mean_lis,
 df_res_all_lis, 
 df_res_cov_all_lis,
 df_res_labels_all,
 df_res_pred_set_all_lis)  = experiment_fig_1_2(score_list,                                      
                                           label_ir,                                   
                                           [0.05],                                 
                                           N_sim,                          
                                           quant_prop=.2,                         
                                           cal_prop=0.8, 
                                           seed_init=0)
          

fig, ax = plt.subplots(nrows=2, figsize=(6, 10))
title = ['none', 'low', 'medium', 'high']
set_size = [np.array(df_res_all_mean_lis[0][x]) for x in df_res_all_mean_lis[0]]
cov = [np.array(df_res_cov_all_mean_lis[0][x]) for x in df_res_cov_all_mean_lis[0]]


labels = list(df_res_all_mean_lis[0].keys())
colors = ['peachpuff', 'orange', 'tomato']

ax[0].set_ylabel('set size')
ax[0].set_title('set size')

bplot = ax[0].boxplot(set_size,
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    
    
labels = list(df_res_cov_all_mean_lis[0].keys())   
ax[1].set_ylabel('empirical confidence')
ax[1].set_title('empirical confidence')

bplot = ax[1].boxplot(cov,
                       patch_artist=True,
                       labels=labels)  # will be used to label x-ticks

    # fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()             

pd.concat([pd.DataFrame({'set_size_'+x:set_size[i] for i, x in enumerate(df_res_all_mean_lis[0])}),
              pd.DataFrame({'cov_'+x:cov[i] for i, x in enumerate(df_res_cov_all_mean_lis[0])})], axis=1).to_csv(directory_name+'/data_fig_3b.csv')    
                          

# Figures SI

def figure_SI(method):
    table_fig2d_list = []

    table_fig2d_aux = pd.DataFrame(np.zeros((7,3)))
    table_fig2d_aux.columns = ['number of spectra', 
                               'correct label in prediction set', 
                               'correct label not in prediction set']

    set_size_list_df = pd.DataFrame({'set_size':np.arange(7)})
    set_size_list_df

    for df_ss, df_cov in zip(df_res_all_lis[0][method], df_res_cov_all_lis[0][method]):
        table_fig2d = pd.DataFrame({'set_size':df_ss,
                                    'cov':df_cov})

        table_fig2d = pd.concat([table_fig2d.groupby('set_size').count(),
                                 table_fig2d.groupby('set_size').mean(),
                                 1-table_fig2d.groupby('set_size').mean()], axis=1)

        table_fig2d.columns = ['number of spectra', 
                               'correct label in prediction set', 
                               'correct label not in prediction set']
        
        table_fig2d = table_fig2d.reset_index()
        table_fig2d = pd.merge(set_size_list_df, 
                               table_fig2d, 
                               on='set_size', how='left')
        table_fig2d[np.isnan(table_fig2d)] = 0
        table_fig2d_aux = table_fig2d_aux + table_fig2d.iloc[:,1:]
        table_fig2d_list.append(table_fig2d)

    table_fig2d_aux = table_fig2d_aux / np.array([np.array([50]*7),
              np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list])),
              np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list]))]).T

    table_fig2d_aux['set_size'] = range(7)
    table_fig2d_aux = table_fig2d_aux.set_index('set_size')

    table_fig2d_aux_std = pd.DataFrame(np.sqrt(np.sum([np.array(1*(x.iloc[:,1]>0))[:,None]*(x.iloc[:,1:]-table_fig2d_aux)**2 for x in table_fig2d_list], 0)/
                         np.array([np.array([50]*7),
                                   np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list])),
                                   np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list]))]).T))

    table_fig2d_aux_std.columns = table_fig2d_aux.columns

    table_fig2d_aux_std['set_size'] = range(7)
    table_fig2d_aux_std = table_fig2d_aux_std.set_index('set_size')
    table_fig2d_aux_std.columns = [x+' (standard deviation)' for x in table_fig2d_aux_std.columns]

    table_fig2d_aux.columns = [x+' (mean)' for x in table_fig2d_aux.columns]

    return pd.concat([table_fig2d_aux, table_fig2d_aux_std], axis=1)


def figure_SI_2(method):
    table_fig2d_list = []

    table_fig2d_aux = pd.DataFrame(np.zeros((7,3)))
    table_fig2d_aux.columns = ['number of spectra', 
                               'correct label in prediction set', 
                               'correct label not in prediction set']

    set_size_list_df = pd.DataFrame({'set_size':np.arange(7)})
    set_size_list_df

    for df_ss, df_cov in zip(df_res_all_lis[0][method], df_res_cov_all_lis[0][method]):
        table_fig2d = pd.DataFrame({'set_size':df_ss,
                                    'cov':df_cov})

        table_fig2d = pd.concat([table_fig2d.groupby('set_size').count(),
                                 table_fig2d.groupby('set_size').mean(),
                                 1-table_fig2d.groupby('set_size').mean()], axis=1)

        table_fig2d.columns = ['number of spectra', 
                               'correct label in prediction set', 
                               'correct label not in prediction set']
        
        table_fig2d = table_fig2d.reset_index()
        table_fig2d = pd.merge(set_size_list_df, 
                               table_fig2d, 
                               on='set_size', how='left')
        table_fig2d[np.isnan(table_fig2d)] = 0
        table_fig2d_aux = table_fig2d_aux + table_fig2d.iloc[:,1:]
        table_fig2d_list.append(table_fig2d)

    table_fig2d_aux = table_fig2d_aux / np.array([np.array([50]*7),
              np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list])),
              np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list]))]).T

    table_fig2d_aux['set_size'] = range(7)
    table_fig2d_aux = table_fig2d_aux.set_index('set_size')

    table_fig2d_aux_std = pd.DataFrame(np.sqrt(np.sum([np.array(1*(x.iloc[:,1]>0))[:,None]*(x.iloc[:,1:]-table_fig2d_aux)**2 for x in table_fig2d_list], 0)/
                         np.array([np.array([50]*7),
                                   np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list])),
                                   np.array(sum([df.iloc[:,2:].sum(1) for df in table_fig2d_list]))]).T))

    table_fig2d_aux_std.columns = table_fig2d_aux.columns

    table_fig2d_aux_std['set_size'] = range(7)
    table_fig2d_aux_std = table_fig2d_aux_std.set_index('set_size')
    table_fig2d_aux_std.columns = [x+' (standard deviation)' for x in table_fig2d_aux_std.columns]

    table_fig2d_aux.columns = [x+' (mean)' for x in table_fig2d_aux.columns]

    return pd.concat([table_fig2d_aux, table_fig2d_aux_std], axis=1)


for method in ['mvcp', 'view_IR', 'view_Raman']:
    table_SI_2 = pd.merge(pd.concat([pd.DataFrame({'label':np.where(x)[1],
     'n':1}).groupby('label').sum() for x in df_res_pred_set_all_lis[0][method]], axis=1).mean(1).reset_index(),
    pd.concat([pd.DataFrame({'label':x, 'n':1}).groupby('label').sum() for x in df_res_labels_all[0]], axis=1).mean(1).reset_index(), on='label').merge(
    label_table, on='label')
    table_SI_2.columns = ['label', 'number_label_in_pred_sets', 'number_label', 'polymer']
    table_SI_2.to_csv(directory_name+'/data_fig_SI_2_'+method+'.csv')

for method in ['mvcp', 'view_IR', 'view_Raman']:
    figure_SI(method).to_csv(directory_name+'/data_fig_SI_2d_'+method+'.csv')


# Figure 2b

#id_ = 68
#score_example = [score_list[0][id_:id_+1], score_list[1][id_:id_+1]] # id == 138
#print(pd.concat([IR_spectra_mix_std, IR_spectra_mix], axis=0).reset_index(drop=True).iloc[id_:id_+1,:]['id'])
