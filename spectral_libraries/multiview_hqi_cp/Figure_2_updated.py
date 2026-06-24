#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 21:38:40 2026

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
    

top_hqi_values = 1-pd.DataFrame(np.concatenate(score_list, axis=1))
top_hqi_values.columns = [x + ' (IR)' for x in label_table['polymer']] + [x + ' (Raman)' for x in label_table['polymer']]
top_hqi_values = pd.concat([top_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)

all_hqi_values = 1-pd.DataFrame(np.concatenate(score_all_list, axis=0).T)
all_hqi_values.columns = list(rebecca_data_ir['id']) + list(rebecca_data_raman['id'])
all_hqi_values = pd.concat([all_hqi_values, IR_spectra_all.iloc[:,-6:-1]], axis=1)



id_ = 519#116
example_id = IR_spectra_all.iloc[id_,-2]
score_example = [score_list[0][id_:id_+1], score_list[1][id_:id_+1]]


for n_th in [4, 5, 6, 20, 400]:
    (ps_list, 
     ps_ir_list, 
     ps_raman_list, 
     env_list, 
     cal_smx_arr, 
     cal_idx_list, 
     q_ir_list, 
     q_raman_list,
     beta_list,
     t_list,
     q_array_list) = envelop_3(score_list, label_ir, score_example,
                             alphas=[0.40,.20, .10, 0.05, 0.02],
                             label_table=label_table,
                             N_sim=1, 
                             quant_prop=.2, 
                             cal_prop=0.8, 
                             seed_init=0,
                            n_th=n_th)

    K = 2
    U_list = gen_u(K, n_th=n_th, seed=0)

    idx_quant = cal_idx_list[20][0][0]
    idx = cal_idx_list[20][0][1]

    score_list_cal = [score_list[0][~idx_quant][idx],
                      score_list[1][~idx_quant][idx]]
    label_list_cal = label_ir[~idx_quant][idx]

    score_list_quant = [score_list[0][idx_quant],
                      score_list[1][idx_quant]]
    label_list_quant = label_ir[idx_quant]

    quant_smx_arr = np.array(score_list_quant)[:,
                             np.arange(label_list_quant.shape[0]),                           
                             label_list_quant]
    
    df_projections = pd.DataFrame()
    df_orth_list = [] 
    df_orth = pd.DataFrame()
    env_th = np.array(env_list[5][0]).T #@ u_theta
    directions = np.where(
            np.array([np.max([np.sum(np.abs((env_th - env_th[i])@ U_list[0][l][:,None])<1e-12) for i in range(env_th.shape[0])]) for l in range(n_th)])>1
        )[0]
    #directions = np.concatenate([directions, [294]])
    directions = directions[np.argsort(U_list[0][directions] @ np.array([1,0]))]
    plt.figure(figsize=(6, 6))


    for l in directions:#[180, 350, 150, 48]: #233 #75, 96
        #print(l)

        plt.plot(1-env_list[5][0][0]/t_list[5][0][1], 
                 1-env_list[5][0][1]/t_list[5][0][1], color='green', label='raw')
        
        plt.plot(1-env_list[5][0][0], 
                 1-env_list[5][0][1], color='purple', label='calibrated')

        score_list_quant_projected = project_u_mv(score_list_quant, U_list[0][l])[0][np.arange(label_list_quant.shape[0]),
                                                        label_list_quant]

        u_theta = U_list[0][l][:,None]
        score_list_quant_projected_vect = (u_theta*score_list_quant_projected).T

        n = score_list_quant_projected.shape[0]
        beta = beta_list[5][0]
        q_level = np.clip(np.ceil((n+1)*(1-beta))/n, 0, 1)
        quant_proj = np.quantile(score_list_quant_projected, q_level, method='higher')

        id_point = np.where(score_list_quant_projected == quant_proj)[0][0]
        print(id_point, quant_proj)
        #plt.figure(figsize=(6, 6))

        #plt.scatter(1-quant_smx_arr[0,:],
        #            1-quant_smx_arr[1,:], alpha=.2)
        #plt.scatter(1-u_theta[0],
        #            1-u_theta[1])

        plt.scatter(1-score_list_quant_projected_vect[:,0],
                    1-score_list_quant_projected_vect[:,1],
                    c=np.linalg.norm(1-score_list_quant_projected_vect, axis=1)/np.sqrt(2))

        plt.scatter(1-u_theta[0]*np.linspace(0,1.5,1000),
                    1-u_theta[1]*np.linspace(0,1.5,1000),
                    s=2)

        plt.scatter(1-score_list_quant_projected_vect[id_point,0],
                    1-score_list_quant_projected_vect[id_point,1],
                    color='red')

        orth_vect = np.array([score_list_quant_projected_vect[id_point,0],
        score_list_quant_projected_vect[id_point,1]])[:,None] + (np.concatenate([-u_theta[1],u_theta[0]])[:,None]*np.linspace(-1,1, 100))


        plt.plot(1-orth_vect[0,:],
                    1-orth_vect[1,:], 
                 #color='orange'
                # s=1
                )

        df_aux = pd.DataFrame(1-score_list_quant_projected_vect)
        df_aux.columns = ['IR - '+str(l), 'Raman - '+str(l)]
        df_aux['threshold_'+str(l)] = 0
        df_aux.iloc[id_point,2] = 1
        df_projections = pd.concat([df_projections, df_aux], axis=1)

        df_aux = pd.DataFrame(1-orth_vect.T)
        df_aux.columns = ['IR', 'Raman']
        df_aux = df_aux.query('IR>0 and IR < 1 and Raman >0 and Raman < 1').reset_index(drop=True)
        df_aux.columns = ['IR - '+str(l), 'Raman - '+str(l)]
        df_orth_list.append(df_aux)
        #df_orth = pd.concat([df_orth, df_aux], axis=1)

        plt.xlim(-.01,1.01)
        plt.ylim(-.01,1.01)
        #plt.xlim(.3,1)
        #plt.ylim(.3,.6)
        #plt.show()
    df_orth = pd.concat(df_orth_list, axis=1)
    df_orth.to_csv(directory_name+'/data_fig_calibration_b_orthogonal_'+str(n_th)+'.csv')
    df_projections.to_csv(directory_name+'/data_fig_calibration_b_'+str(n_th)+'.csv')
    
    df_env = pd.DataFrame({
    'IR':1-env_list[5][0][0], 
    'Raman':1-env_list[5][0][1]
    })
    df_env.to_csv(directory_name+'/data_fig_calibration_envelope_'+str(n_th)+'.csv')
    
    df_env = pd.DataFrame({
    'IR':1-env_list[5][0][0]/t_list[5][0][1], 
    'Raman':1-env_list[5][0][1]/t_list[5][0][1]
    })
    df_env.to_csv(directory_name+'/data_fig_calibration_envelope_raw_'+str(n_th)+'.csv')
    
    plt.legend()
    plt.show()