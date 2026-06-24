#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 12:39:44 2025

@author: eochoa
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from utils import *
#from utils_cpdir import *
from utils_mvcp import *
from utils import *
from utils_experiments import *
import os
from tqdm import tqdm
from datetime import date

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


del Raman_spectra_mix_std['Unnamed: 0']
Raman_spectra_mix_std.columns = list(np.float64(Raman_spectra_mix_std.columns[:-7])) + list(Raman_spectra_mix_std.columns[-7:])
Raman_spectra_mix_std = (Raman_spectra_mix_std.
 merge(poly_abbr_table, left_on='polymer', right_on='abbr').
 drop(['polymer_x', 'abbr'],axis=1).
 rename({'polymer_y':'polymer'},axis=1).
iloc[:,list(range(Raman_spectra_mix_std.shape[1]-5))+[-1,-2]])


notes_mix_std = notes_mix_std.sort_values('id').reset_index(drop=True)
Raman_spectra_mix_std = Raman_spectra_mix_std.sort_values('id').reset_index(drop=True)

Raman_spectra_mix_std_rest = Raman_spectra_mix_std.loc[[x not in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
notes_mix_std_rest = notes_mix_std.loc[[x not in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)



notes_mix_std = notes_mix_std.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)
Raman_spectra_mix_std = Raman_spectra_mix_std.loc[[x in poly_list for x in Raman_spectra_mix_std['polymer']],:].reset_index(drop=True)

Raman_spectra_mix_std.loc[Raman_spectra_mix_std['polymer'] == 'LDPE','polymer'] = 'PE'



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


Raman_spectra_mix = pd.read_csv('processed_data_new/data_raman_mix_rebecca_specy.csv')
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


IR_spectra_amb = pd.read_csv('processed_data_new/data_ir_amb_rebecca.csv')
del IR_spectra_amb['Unnamed: 0']
#non_bg = IR_spectra['background'] == 'no_background'
IR_spectra_amb.columns = list(np.float64(IR_spectra_amb.columns[:-8])) + list(IR_spectra_amb.columns[-8:])
IR_spectra_amb_rest = IR_spectra_amb.loc[[x not in ['LDPE', 'PP', 'PSL'] for x in IR_spectra_amb['polymer']],:].reset_index(drop=True)
#IR_spectra_amb = IR_spectra_amb.loc[[x in ['LDPE', 'PP', 'PSL', 'AMB'] for x in IR_spectra_amb['polymer']],:].reset_index(drop=True)
#IR_spectra_amb = IR_spectra_amb.loc[[x in ['AMB','LDPE', 'PP', 'PSL'] for x in IR_spectra_amb['polymer']],:].reset_index(drop=True)
IR_spectra_amb.loc[IR_spectra_amb['polymer'] == 'LDPE','polymer'] = 'PE'
IR_spectra_amb.loc[IR_spectra_amb['polymer'] == 'AMB','polymer'] = 'z_AMB'
IR_spectra_amb.loc[IR_spectra_amb['polymer'] == 'LOW','polymer'] = 'z_LOW'
IR_spectra_amb.loc[IR_spectra_amb['polymer'] == 'MIX','polymer'] = 'z_MIX'
IR_spectra_amb.loc[IR_spectra_amb['polymer'] == 'UNS','polymer'] = 'z_UNS'


Raman_spectra_amb = pd.read_csv('processed_data_new/data_raman_amb_rebecca_specy.csv')
del Raman_spectra_amb['Unnamed: 0']
Raman_spectra_amb.columns = list(np.float64(Raman_spectra_amb.columns[:-7])) + list(Raman_spectra_amb.columns[-7:])
Raman_spectra_amb_rest = Raman_spectra_amb.loc[[x not in ['LDPE', 'PP', 'PSL'] for x in Raman_spectra_amb['polymer']],:].reset_index(drop=True)
#Raman_spectra_amb = Raman_spectra_amb.loc[[x in ['LDPE', 'PP', 'PSL', 'AMB'] for x in Raman_spectra_amb['polymer']],:].reset_index(drop=True)
#Raman_spectra_amb = Raman_spectra_amb.loc[[x in ['AMB','LDPE', 'PP', 'PSL'] for x in Raman_spectra_amb['polymer']],:].reset_index(drop=True)
Raman_spectra_amb.loc[Raman_spectra_amb['polymer'] == 'LDPE','polymer'] = 'PE'
Raman_spectra_amb.loc[Raman_spectra_amb['polymer'] == 'AMB','polymer'] = 'z_AMB'
Raman_spectra_amb.loc[Raman_spectra_amb['polymer'] == 'LOW','polymer'] = 'z_LOW'
Raman_spectra_amb.loc[Raman_spectra_amb['polymer'] == 'MIX','polymer'] = 'z_MIX'
Raman_spectra_amb.loc[Raman_spectra_amb['polymer'] == 'UNS','polymer'] = 'z_UNS'


poly_list_amb = poly_list+['z_AMB']+['AMB']
ids_ideal_set = [(x in poly_list_amb) and (y in poly_list_amb) and (z in poly_list_amb) for x, y, z in zip(Raman_spectra_amb['polymer_raman'], Raman_spectra_amb['polymer_ir'], Raman_spectra_amb['polymer'])]

IR_spectra_amb = IR_spectra_amb.iloc[:,list(range(IR_spectra_amb.shape[1]-8))+[-8,-7,-5,-4,-3,-6,-2,-1]]
Raman_spectra_amb = Raman_spectra_amb.iloc[:,list(range(Raman_spectra_amb.shape[1]-4))+[-1]]
idx_nan_raman = np.isnan(Raman_spectra_amb.iloc[:,:-4]).sum(1)>0

Raman_spectra_amb = Raman_spectra_amb.loc[~idx_nan_raman,:].reset_index(drop=True)
IR_spectra_amb = IR_spectra_amb.loc[~idx_nan_raman,:].reset_index(drop=True)


Raman_spectra_amb['id'] = ["amb_"+str(x) for x in Raman_spectra_amb['id']]
IR_spectra_amb['id'] = ["amb_"+str(x) for x in IR_spectra_amb['id']]

IR_spectra_amb.to_csv(directory_name+'/data_fig_5_ir_amb_processed_.csv')
Raman_spectra_amb.to_csv(directory_name+'/data_fig_5_raman_amb_processed.csv')

rebecca_data_ir = rebecca_data_ir.query('polymer in ["PE", "PP", "PSL"]').reset_index(drop=True)
rebecca_data_raman = rebecca_data_raman.query('polymer in ["PE", "PP", "PSL"]').reset_index(drop=True)
IR_spectra_all = pd.concat([IR_spectra_mix_std.query('polymer in ["PE", "PP", "PSL"]'), 
                            IR_spectra_mix], axis=0).reset_index(drop=True)

Raman_spectra_all = pd.concat([Raman_spectra_mix_std.query('polymer in ["PE", "PP", "PSL"]'), 
                               Raman_spectra_mix], axis=0).reset_index(drop=True)

lower_range_raman_1 = 600
upper_range_raman_1 = 1850
lower_range_raman_2 = 2750
upper_range_raman_2 = 3500

wn_raman = np.float64(rebecca_data_raman.columns[:-4])

idx_raman = list(np.logical_or(np.logical_and(wn_raman >lower_range_raman_1,
                               wn_raman <upper_range_raman_1
                               ),
                     np.logical_and(wn_raman >lower_range_raman_2,
                               wn_raman <upper_range_raman_2
                               ))) + [True]*4

(x_train_ir, 
 x_train_raman, 
 label_train_IR, 
 label_train_Raman, 
 x_ir, x_raman, label_ir, label_raman, 
 label_table, ids_cal) = process_data(rebecca_data_ir,
                                      rebecca_data_raman.loc[:, idx_raman],
                                      IR_spectra_all,
                                      Raman_spectra_all.loc[:, idx_raman]
                                      )

                                      
(_, _, _, _, 
 x_amb_ir, x_amb_raman, label_amb_ir, 
 label_amb_raman, 
 label_amb_table, ids_test) = process_data(rebecca_data_ir,
                                           rebecca_data_raman.loc[:, idx_raman],
                                           IR_spectra_amb,
                                           Raman_spectra_amb.loc[:, idx_raman]
                                           )
                                           

smx_all_IR = 1-np.power(smx_nn(x_train_ir, x_ir, label_train_IR, tau=1.4427),1)
smx_all_Raman = 1-np.power(smx_nn(x_train_raman, x_raman, label_train_Raman, tau=1.4427),1)

smx_all_IR_test = 1-np.power(smx_nn(x_train_ir, x_amb_ir, label_train_IR, tau=1.4427),1)
smx_all_Raman_test = 1-np.power(smx_nn(x_train_raman, x_amb_raman, label_train_Raman, tau=1.4427),1)

smx_all_IR_test_all = 1-np.power(smx_nn_all(x_train_ir, x_amb_ir, label_train_IR, tau=1.4427),1)
smx_all_Raman_test_all = 1-np.power(smx_nn_all(x_train_raman, x_amb_raman, label_train_Raman, tau=1.4427),1)

score_list = [smx_all_IR, smx_all_Raman]
score_list_test = [smx_all_IR_test, smx_all_Raman_test]
score_all_list = [smx_all_IR_test_all, smx_all_Raman_test_all]

top_hqi_values = 1-pd.DataFrame(np.concatenate(score_list_test, axis=1))
top_hqi_values.columns = [x + ' (IR)' for x in label_table['polymer']] + [x + ' (Raman)' for x in label_table['polymer']]
top_hqi_values = pd.concat([top_hqi_values, IR_spectra_amb.iloc[:,-6:-1]], axis=1)
top_hqi_values = top_hqi_values.sort_values(by='polymer')
top_hqi_values.to_csv(directory_name+'/data_fig_5_top_hqi_values.csv')

all_hqi_values = 1-pd.DataFrame(np.concatenate(score_all_list, axis=0).T)
all_hqi_values.columns = list(rebecca_data_ir['id']) + list(rebecca_data_raman['id'])
all_hqi_values = pd.concat([all_hqi_values, IR_spectra_amb.iloc[:,-6:-1]], axis=1)
all_hqi_values = all_hqi_values.sort_values(by='polymer')
all_hqi_values.to_csv(directory_name+'/data_fig_5_all_hqi_values.csv')


TPR = defaultdict(list)
FPR = defaultdict(list)
F_1 = defaultdict(list)
U_quant_env_list = defaultdict(list)
prediction_sets_list = defaultdict(list)
emp_cov_list = defaultdict(list)

N_sim = 100
N_alpha = 99
for i in tqdm(range(N_sim)):
    seed = i
    K = 2
    quant_smx_list = []
    rest_smx_list = []
    cal_smx_list = []
    val_smx_list = []
    U_list = gen_u(K, seed=0)

    
    label_cal_ir = label_ir
    val_labels = None

    for i in range(K):
        quant_smx, cal_smx, quant_labels, cal_labels, idx_quant = split_cal_test(score_list[i], label_cal_ir, cal_prop=0.2, seed=seed)

        val_smx, _ = score_list_test[i], None

        quant_smx_list.append(quant_smx)
        cal_smx_list.append(cal_smx)
        val_smx_list.append(val_smx)

    n = cal_labels.shape[0]

    for alpha in (np.linspace(0.01,.99,N_alpha)):

        prediction_sets, U_quant_env = mvcp_2(quant_smx_list, 
                                                     quant_labels,   
                                                     cal_smx_list,   
                                                     cal_labels,   
                                                     val_smx_list,   
                                                     alpha,     
                                                     val_labels,                 
                                                     U=U_list[0])

        U_quant_env_list[round(alpha*100)].append(U_quant_env)
        prediction_sets_n_mvcp = [list(label_table.loc[ps,'polymer']) for ps in prediction_sets]
        prediction_sets_size = np.array([len(ps) for ps in prediction_sets_n_mvcp])
        
        prediction_sets_list[round(alpha*100)].append(prediction_sets.sum(1).mean())
        
        
        emp_cov_list[round(alpha*100)].append(
            np.mean(prediction_sets[label_amb_ir<3][np.arange(sum(label_amb_ir<3)),
                                            label_amb_ir[label_amb_ir<3]])
            )
            
        #TPR[round(alpha*100)].append(np.mean(prediction_sets[label_amb_ir<3,:][np.arange(label_amb_ir[label_amb_ir<3].shape[0]), 
        #                                                             label_amb_ir[label_amb_ir<3]]))
        
        TPR[round(alpha*100)].append(
            np.mean(prediction_sets[label_amb_ir<3,:].sum(1) > 0)
            )
        
        FPR[round(alpha*100)].append(np.sum(label_amb_ir[prediction_sets_size>0] >= 3) / sum(label_amb_ir >= 3))
        
        #Precision = np.sum(prediction_sets[label_amb_ir<3,:][np.arange(label_amb_ir[label_amb_ir<3].shape[0]), 
        #                                                             label_amb_ir[label_amb_ir<3]]) / sum(prediction_sets.sum(1)>0)
        
        #Recall = np.mean(prediction_sets[label_amb_ir<3,:][np.arange(label_amb_ir[label_amb_ir<3].shape[0]), 
        #                                                             label_amb_ir[label_amb_ir<3]])
        
        Precision = np.sum(prediction_sets[label_amb_ir<3,:].sum(1) > 0) / sum(prediction_sets.sum(1)>0)
        
        Recall = np.mean(prediction_sets[label_amb_ir<3,:].sum(1) > 0)
        
        F_1[round(alpha*100)].append(
        2 * (Precision * Recall) / (Precision + Recall)
        )

id_ = 20
score_example = [score_list_test[0][id_:id_+1], 
                 score_list_test[1][id_:id_+1]] 

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
                         alphas=list(np.round(np.linspace(0.01,.99,N_alpha), 2)),
                         label_table=label_table,
                         N_sim=N_sim, 
                         quant_prop=.2, 
                         cal_prop=0.8, 
                         seed_init=0, 
                         plot=False,
                         cp=False)

df_calibration = pd.DataFrame({
            'IR':1-cal_smx_arr[0],
            'Raman':1-cal_smx_arr[1]
            })

df_calibration.to_csv(directory_name+'/data_fig_5_calibration.csv')
                         
for x in env_list[10]:
    plt.scatter(1-x[0],
                1-x[1], color='blue', alpha=.05)

_ = plt.scatter(
    1-np.mean(env_list[10], 0)[0],
    1-np.mean(env_list[10], 0)[1], color='red')


for i, alpha in enumerate(env_list.keys()):
    if i == 0:
        df_envelopes = pd.DataFrame({
            'x - '+str(100-alpha) + ' (mean)':1-np.mean(env_list[alpha],0)[0],
            'y - '+str(100-alpha) + ' (mean)':1-np.mean(env_list[alpha],0)[1]
                      })
    else:
        df_envelopes = pd.concat([df_envelopes,
                        pd.DataFrame({
                            'x - '+str(100-alpha) + ' (mean)':1-np.mean(env_list[alpha],0)[0],
                            'y - '+str(100-alpha) + ' (mean)':1-np.mean(env_list[alpha],0)[1]
                      })], axis=1)
for i, alpha in enumerate(env_list.keys()):#[2, 5, 10, 20, 40]:#enumerate(U_quant_env_list.keys()):
    df_envelopes = pd.concat([
        df_envelopes,
        pd.DataFrame({
                            'x - '+str(100-alpha) + ' (standard deviation)':np.std(env_list[alpha],0)[0],
                            'y - '+str(100-alpha) + ' (standard deviation)':np.std(env_list[alpha],0)[1]
                      })], axis=1)

        
    
df_envelopes.to_csv(directory_name+'/data_fig_5_envelopes.csv')


_ = plt.show()

fig, ax1 = plt.subplots()

# Plot first dataset (left y-axis)
ax1.plot(pd.DataFrame(prediction_sets_list).mean(), color="blue", label="Prediction set size")
#ax1.set_xlabel("X")
#ax1.set_ylabel("sin(x)", color="blue")
#ax1.tick_params(axis="y", labelcolor="blue")

# Create second y-axis (right side)
ax2 = ax1.twinx()
ax2.plot(pd.DataFrame(FPR).mean(), color="red", label="FPR")
ax2.plot(pd.DataFrame(TPR).mean(), color="orange", label="TPR")
ax2.plot(pd.DataFrame(emp_cov_list).mean(), color="yellow", label="Empirical coverage")
#ax2.set_ylabel("exp(0.3x)", color="red")
#ax2.tick_params(axis="y", labelcolor="red")

# Title
#plt.title("Plot with Two Y-Axes")
plt.legend()

plt.show()

#_ = plt.plot(pd.DataFrame(prediction_sets_list).mean())

#_ = plt.plot(pd.DataFrame(FPR).mean())

#_ = plt.plot(pd.DataFrame(TPR).mean())

#_ = plt.show()

_ = plt.plot(pd.DataFrame(FPR).mean(),
             pd.DataFrame(TPR).mean())

_ = plt.show()

pd.DataFrame({'Set size (mean)':pd.DataFrame(prediction_sets_list).mean(),
              'Empirical confidence (mean)':pd.DataFrame(emp_cov_list).mean(),
              'TPR (mean)':pd.DataFrame(TPR).mean(),
              'FPR (mean)':pd.DataFrame(FPR).mean(),
              'F_1 score (mean)':pd.DataFrame(F_1).mean(),
              'Set size (standard deviation)':pd.DataFrame(prediction_sets_list).std(),
              'Empirical confidence (standard deviation)':pd.DataFrame(emp_cov_list).std(),
              'TPR (standard deviation)':pd.DataFrame(TPR).std(),
              'FPR (standard deviation)':pd.DataFrame(FPR).std(),
              'F_1 score (standard deviation)':pd.DataFrame(F_1).std()  
              }).to_csv(directory_name+'/data_fig_5_TPR_FPR.csv')
