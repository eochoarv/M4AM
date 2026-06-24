#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:30:32 2025

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
import os
from tqdm import tqdm


def envelop(score_list, label_ir, score_example, alphas, label_table, N_sim=1, quant_prop=.2, cal_prop=.8, seed_init=0, plot=True, cp=True, n_th=400):
    env_list = defaultdict(list)
    ps_list = defaultdict(list)
    ps_ir_list = defaultdict(list)
    ps_raman_list = defaultdict(list)
    q_ir_list = defaultdict(list)
    q_raman_list = defaultdict(list)
    cal_idx_list = defaultdict(list)
    beta_list = defaultdict(list)
        
    for alpha in tqdm(alphas):
        set_size = defaultdict(list)
        set_size_all = defaultdict(list)
        zero_sets = defaultdict(list)
        cov = defaultdict(list)
        cov_cal = defaultdict(list)
        set_size_by_label_hqi = defaultdict(list)
        
        K = 2
        #U_list = gen_u(K, n_th=400, seed=0)
        U_list = gen_u(K, n_th=n_th, seed=0)
        seed = seed_init
        for l in range(N_sim):
            seed = seed + l

            #score_list, y_train, y_test, top_views = simulate_s_2(N, d, d_v, L, scale, K, N_train, seed)
            #top_views = np.array(acc_list)
            quant_smx_list = []
            rest_smx_list = []
            cal_smx_list = []
            val_smx_list = []


            for i in range(K):

                quant_smx, rest_smx, quant_labels, rest_labels, idx_quant = split_cal_test(score_list[i], label_ir, cal_prop=quant_prop, seed=seed)
                cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(rest_smx, rest_labels, cal_prop=cal_prop, seed=seed)
                

                quant_smx_list.append(quant_smx)
                rest_smx_list.append(rest_smx)
                cal_smx_list.append(cal_smx)
                val_smx_list.append(val_smx)

            n = cal_labels.shape[0]
            #n = quant_labels.shape[0]
            if cp:
                prediction_sets, U_quant_env = mvcp_2(quant_smx_list,
                                                                 quant_labels,
                                                                 cal_smx_list, #quant_smx_list,
                                                                 cal_labels, #quant_labels,
                                                                 score_example,#val_smx_list, #rest_smx_list, 
                                                                 alpha, 
                                                                 val_labels=None, #rest_labels,
                                                                 U=U_list[0])
    
                
                s_u = rest_smx_list[0]
    
                cal_ir_smx, val_ir_smx, cal_ir_labels, val_ir_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
                n = cal_ir_smx.shape[0]
                prediction_sets_ir, qhat_ir = conform_prediction(cal_ir_smx, score_example[0], cal_ir_labels, n, alpha, val_labels=None, cond=0)
    
                s_u = rest_smx_list[1]
    
                cal_raman_smx, val_raman_smx, cal_raman_labels, val_raman_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
                n = cal_raman_smx.shape[0]
                prediction_sets_raman, qhat_raman = conform_prediction(cal_raman_smx, score_example[1], cal_raman_labels, n, alpha, val_labels=None, cond=0)            
            else:
                prediction_sets = [[]]
                prediction_sets_ir = [[]]
                prediction_sets_raman = [[]]
                
                qhat_ir = 0
                qhat_raman = 0
                
            
            ps_ir_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_ir[0], 'polymer']))            
            ps_raman_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_raman[0], 'polymer']))            
            ps_list[int(100*alpha)].append(list(label_table.loc[prediction_sets[0], 'polymer']))
            

            beta, q_array, U = calibrate_beta_mv(alpha, quant_smx_list, quant_labels, U=U_list[0])

            n = cal_smx_list[0].shape[0]
            t = np.max([project_u_mv(cal_smx_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
            q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
            t_quant = np.quantile(t, q_level, method='higher')
            cal_smx_arr = np.array([cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                                    cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]])
            
            cal_idx_list[int(100*alpha)].append((idx_quant,idx))
            
        
            idx_quant_reg = np.argsort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_x = t_quant*np.sort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_y = t_quant*(cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]/t)[idx_quant_reg]
            
            q_ir_list[int(100*alpha)].append(qhat_ir)
            q_raman_list[int(100*alpha)].append(qhat_raman)
            env_list[int(100*alpha)].append([quant_reg_x, quant_reg_y])
            beta_list[int(100*alpha)].append(beta)
            
            
            #for th in range(U_list[0].shape[0]):
            #    u = U_list[0][th]
            #    plt.scatter(u[0]*q_array[th],
            #                u[1]*q_array[th],
            #                c='yellow',
            #                s=1)
            
            #plt.scatter(U_quant_env[:,0],
            #            U_quant_env[:,1],
            #                c='green',
            #                s=1)
            
            if plot:
                plt.scatter(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                            cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels], 
                            label='calibration set')
                
                print("MVCP", ps_list[int(100*alpha)][-1])
                print("IR", ps_ir_list[int(100*alpha)][-1])
                print("Raman", ps_raman_list[int(100*alpha)][-1])
                
                _ = plt.plot(quant_reg_x,
                         quant_reg_y, 
                         label=str(int(100*(1-alpha)))+' quantile region',
                         c='orange',
                         linewidth=3.0
                     )
                
                _ = plt.vlines(x= qhat_ir,
                    ymin=0,
                    ymax=.25,
                    colors='r',
                    alpha=.5,
                    linestyles='solid',
                    label='95 IR Quantile')
    
                _ = plt.hlines(y= qhat_raman,
                                xmin=0,
                                xmax=.7,
                                colors='purple',
                                alpha=.5,
                                linestyles='solid',
                                label='95 Raman Quantile')
    
                _ = plt.scatter(score_example[0],
                            score_example[1], c='r')
                
                for i in range(score_example[0].shape[1]):
                    _ = plt.text(score_example[0][0, i], 
                             score_example[1][0, i], label_table.iloc[i,1], fontsize=9, ha='right')
    
                _ = plt.xlabel("IR 1-HQI")
                _ = plt.ylabel("Raman 1-HQI")
    
                _ = plt.legend()
                _ = plt.show()


    

    return ps_list, ps_ir_list, ps_raman_list, env_list, cal_smx_arr, cal_idx_list, q_ir_list, q_raman_list, beta_list


def envelop_2(score_list, label_ir, alphas, label_table, N_sim=1, quant_prop=.2, cal_prop=.8, seed_init=0, quality=None):
    env_list = defaultdict(list)
    ps_list = defaultdict(list)
    ps_ir_list = defaultdict(list)
    ps_raman_list = defaultdict(list)
    q_ir_list = defaultdict(list)
    q_raman_list = defaultdict(list)
    cal_idx_list = defaultdict(list)
    cal_smx_arr_list = defaultdict(list)
        
    for alpha in alphas:
        set_size = defaultdict(list)
        set_size_all = defaultdict(list)
        zero_sets = defaultdict(list)
        cov = defaultdict(list)
        cov_cal = defaultdict(list)
        set_size_by_label_hqi = defaultdict(list)
        K = 2
        U_list = gen_u(K, n_th=400, seed=0)
        seed = seed_init
        for l in tqdm(range(N_sim)):
            seed = seed + l

            #score_list, y_train, y_test, top_views = simulate_s_2(N, d, d_v, L, scale, K, N_train, seed)
            #top_views = np.array(acc_list)
            quant_smx_list = []
            rest_smx_list = []
            cal_smx_list = []
            val_smx_list = []


            for i in range(K):

                quant_smx, rest_smx, quant_labels, rest_labels, idx_quant = split_cal_test(score_list[i], label_ir, cal_prop=quant_prop, seed=seed)
                cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(rest_smx, rest_labels, cal_prop=cal_prop, seed=seed)

                if quality is not None:
                    quality_quant = quality[idx_quant]
                    quality_rest = quality[~idx_quant]
                    quality_cal = quality_rest[idx]
                    quality_val = quality_rest[~idx]
                else:
                    quality_val = np.array(['low_ir']*val_smx.shape[0])

                quant_smx_list.append(quant_smx)
                rest_smx_list.append(rest_smx)
                cal_smx_list.append(cal_smx)
                val_smx_list.append(val_smx)

            n = cal_labels.shape[0]
            #n = quant_labels.shape[0]

            prediction_sets, empirical_coverage, U_quant_env = mvcp_2(quant_smx_list,
                                                             quant_labels,
                                                             cal_smx_list, #quant_smx_list,
                                                             cal_labels, #quant_labels,
                                                             val_smx_list, #rest_smx_list, 
                                                             alpha, 
                                                             val_labels, #rest_labels,
                                                             U=U_list[0])
            
            
            #set_size['mvcp'].append(
            #        prediction_sets.sum(1).mean()
            #    )
            #cov['mvcp'].append(
            #        empirical_coverage
            #    )

            s_u = rest_smx_list[0]
            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
            n = cal_smx.shape[0]
            (prediction_sets_ir, empirical_coverage_ir), qhat_ir = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            #set_size['view_IR'].append(prediction_sets.sum(1).mean())
            #cov['view_IR'].append(empirical_coverage)

            s_u = rest_smx_list[1]
            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=0.8, seed=seed)
            n = cal_smx.shape[0]
            (prediction_sets_raman, empirical_coverage_raman), qhat_raman = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            #set_size['view_Raman'].append(prediction_sets.sum(1).mean())
            #cov['view_Raman'].append(empirical_coverage)
            
            
            #ps_ir_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_ir, 'polymer']))            
            #ps_raman_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_raman, 'polymer']))            
            #ps_list[int(100*alpha)].append(list(label_table.loc[prediction_sets, 'polymer']))
            
            ps_ir_list[int(100*alpha)].append(prediction_sets_ir[quality_val=="low_ir"].sum(1).mean())
            ps_raman_list[int(100*alpha)].append(prediction_sets_raman[quality_val=="low_ir"].sum(1).mean())
            ps_list[int(100*alpha)].append(prediction_sets[quality_val=="low_ir"].sum(1).mean())
            

            beta, q_array, U = calibrate_beta_mv(alpha, quant_smx_list, quant_labels, U=U_list[0])

            n = cal_smx_list[0].shape[0]
            t = np.max([project_u_mv(cal_smx_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
            q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
            t_quant = np.quantile(t, q_level, method='higher')
            cal_smx_arr = np.array([cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                                    cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]])
            
            cal_smx_arr_list[int(100*alpha)].append(cal_smx_arr)
            
            cal_idx_list[int(100*alpha)].append((idx_quant,idx))
            #plt.scatter(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
            #            cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels], 
            #            label='calibration set')

            
            idx_quant_reg = np.argsort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_x = t_quant*np.sort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_y = t_quant*(cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]/t)[idx_quant_reg]
            
            q_ir_list[int(100*alpha)].append(qhat_ir)
            q_raman_list[int(100*alpha)].append(qhat_raman)
            env_list[int(100*alpha)].append([quant_reg_x, quant_reg_y])
            
            print("MVCP", ps_list[int(100*alpha)][-1])
            print("IR", ps_ir_list[int(100*alpha)][-1])
            print("Raman", ps_raman_list[int(100*alpha)][-1])
            
            
            #plt.plot(quant_reg_x,
            #         quant_reg_y, 
            #         label=str(int(100*(1-alpha)))+' quantile region',
            #         c='orange',
            #         linewidth=3.0
            #     )
            
            #_ = plt.vlines(x= qhat_ir,
            #    ymin=0,
            #    ymax=.25,
            #    colors='r',
            #    alpha=.5,
            #    linestyles='solid',
            #    label='95 IR Quantile')

            #_ = plt.hlines(y= qhat_raman,
            #                xmin=0,
            #                xmax=.7,
            #                colors='purple',
            #                alpha=.5,
            #                linestyles='solid',
            #                label='95 Raman Quantile')

            #plt.xlabel("IR 1-HQI")
            #plt.ylabel("Raman 1-HQI")

            #plt.legend()
            #plt.show()

    return ps_list, ps_ir_list, ps_raman_list, env_list, cal_smx_arr_list, cal_idx_list, q_ir_list, q_raman_list

def envelop_3(score_list, label_ir, score_example, alphas, label_table, N_sim=1, quant_prop=.2, cal_prop=.8, seed_init=0, plot=True, cp=True, n_th=400):
    env_list = defaultdict(list)
    t_list = defaultdict(list)
    q_array_list = defaultdict(list)
    ps_list = defaultdict(list)
    ps_ir_list = defaultdict(list)
    ps_raman_list = defaultdict(list)
    q_ir_list = defaultdict(list)
    q_raman_list = defaultdict(list)
    cal_idx_list = defaultdict(list)
    beta_list = defaultdict(list)
        
    for alpha in tqdm(alphas):
        set_size = defaultdict(list)
        set_size_all = defaultdict(list)
        zero_sets = defaultdict(list)
        cov = defaultdict(list)
        cov_cal = defaultdict(list)
        set_size_by_label_hqi = defaultdict(list)
        
        K = 2
        U_list = gen_u(K, n_th=n_th, seed=0)
        seed = seed_init
        for l in range(N_sim):
            seed = seed + l

            #score_list, y_train, y_test, top_views = simulate_s_2(N, d, d_v, L, scale, K, N_train, seed)
            #top_views = np.array(acc_list)
            quant_smx_list = []
            rest_smx_list = []
            cal_smx_list = []
            val_smx_list = []


            for i in range(K):

                quant_smx, rest_smx, quant_labels, rest_labels, idx_quant = split_cal_test(score_list[i], label_ir, cal_prop=quant_prop, seed=seed)
                cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(rest_smx, rest_labels, cal_prop=cal_prop, seed=seed)
                

                quant_smx_list.append(quant_smx)
                rest_smx_list.append(rest_smx)
                cal_smx_list.append(cal_smx)
                val_smx_list.append(val_smx)

            n = cal_labels.shape[0]
            #n = quant_labels.shape[0]
            if cp:
                prediction_sets, U_quant_env = mvcp_2(quant_smx_list,
                                                                 quant_labels,
                                                                 cal_smx_list, #quant_smx_list,
                                                                 cal_labels, #quant_labels,
                                                                 score_example,#val_smx_list, #rest_smx_list, 
                                                                 alpha, 
                                                                 val_labels=None, #rest_labels,
                                                                 U=U_list[0])
    
                
                s_u = rest_smx_list[0]
    
                cal_ir_smx, val_ir_smx, cal_ir_labels, val_ir_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
                n = cal_ir_smx.shape[0]
                prediction_sets_ir, qhat_ir = conform_prediction(cal_ir_smx, score_example[0], cal_ir_labels, n, alpha, val_labels=None, cond=0)
    
                s_u = rest_smx_list[1]
    
                cal_raman_smx, val_raman_smx, cal_raman_labels, val_raman_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
                n = cal_raman_smx.shape[0]
                prediction_sets_raman, qhat_raman = conform_prediction(cal_raman_smx, score_example[1], cal_raman_labels, n, alpha, val_labels=None, cond=0)            
            else:
                prediction_sets = [[]]
                prediction_sets_ir = [[]]
                prediction_sets_raman = [[]]
                
                qhat_ir = 0
                qhat_raman = 0
                
            
            ps_ir_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_ir[0], 'polymer']))            
            ps_raman_list[int(100*alpha)].append(list(label_table.loc[prediction_sets_raman[0], 'polymer']))            
            ps_list[int(100*alpha)].append(list(label_table.loc[prediction_sets[0], 'polymer']))
            

            beta, q_array, U = calibrate_beta_mv(alpha, quant_smx_list, quant_labels, U=U_list[0])

            n = cal_smx_list[0].shape[0]
            t = np.max([project_u_mv(cal_smx_list, U[th,:])[0][np.arange(n),cal_labels]/q for th, q in enumerate(q_array)], 0)
            q_level = np.clip(np.ceil((n+1)*(1-alpha))/n, 0, 1)
            t_quant = np.quantile(t, q_level, method='higher')
            cal_smx_arr = np.array([cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels],
                                    cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]])
            
            cal_idx_list[int(100*alpha)].append((idx_quant,idx))
            
        
            idx_quant_reg = np.argsort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_x = t_quant*np.sort(cal_smx_list[0][np.arange(cal_labels.shape[0]),cal_labels]/t)#[::-1]
            quant_reg_y = t_quant*(cal_smx_list[1][np.arange(cal_labels.shape[0]),cal_labels]/t)[idx_quant_reg]
            
            q_ir_list[int(100*alpha)].append(qhat_ir)
            q_raman_list[int(100*alpha)].append(qhat_raman)
            env_list[int(100*alpha)].append([quant_reg_x, quant_reg_y])
            t_list[int(100*alpha)].append([t, t_quant])
            q_array_list[int(100*alpha)].append(q_array)
            beta_list[int(100*alpha)].append(beta)
            

    return ps_list, ps_ir_list, ps_raman_list, env_list, cal_smx_arr, cal_idx_list, q_ir_list, q_raman_list, beta_list, t_list, q_array_list


def process_data(rebecca_data_ir, rebecca_data_raman, IR_spectra, Raman_spectra, range_r=[550, 3500]):
    
    #range_r = [550, 3500] 
    
    wn_rb = np.array(rebecca_data_ir.columns[:-4])
    idx_rb = list(np.logical_or(np.logical_and(wn_rb > 980, wn_rb < 1800),
                           np.logical_and(wn_rb > 2720, wn_rb < 2950)
                          ) ) + [False]*4
    
    wn_rb_r = np.array(rebecca_data_raman.columns[:-4])
    idx_rb_r = list(np.logical_and(wn_rb_r > range_r[0], wn_rb_r < range_r[1])) + [False]*4


    wn_rb_amb = np.array(IR_spectra.columns[:-8])
    idx_rb_amb = list(np.logical_or(np.logical_and(wn_rb_amb > 980, wn_rb_amb < 1800),
                           np.logical_and(wn_rb_amb > 2720, wn_rb_amb < 2950) #2993.64990234375-10
                          ) ) + [False]*8
    
    wn_rb_amb_r = np.array(Raman_spectra.columns[:-4])
    idx_rb_amb_r = list(np.logical_and(wn_rb_amb_r > range_r[0], wn_rb_amb_r < range_r[1])) + [False]*4
    
    x_train_ir = rebecca_data_ir.iloc[:,idx_rb].to_numpy()
    x_train_ir = (x_train_ir - x_train_ir.mean(1)[:, None]) / x_train_ir.std(1)[:, None]
    
    y_train_ir = rebecca_data_ir.iloc[:,-2]

    x_train_raman = rebecca_data_raman.iloc[:,idx_rb_r].to_numpy()
    x_train_raman = (x_train_raman - x_train_raman.mean(1)[:, None]) / x_train_raman.std(1)[:, None]
    
    y_train_raman = rebecca_data_raman.iloc[:,-2]

    x_ir = IR_spectra.iloc[:,idx_rb_amb].to_numpy()
    x_ir = (x_ir - x_ir.mean(1)[:, None]) / x_ir.std(1)[:, None]    

    x_raman = Raman_spectra.iloc[:,idx_rb_amb_r].to_numpy()
    x_raman = (x_raman - x_raman.mean(1)[:, None]) / x_raman.std(1)[:, None]
    y_test = IR_spectra.iloc[:,-3]
    
    idx_test = np.sum(np.isnan(x_raman), axis=1) == 0
    ids_test = IR_spectra.loc[idx_test,'id']
    
    x_ir = x_ir[idx_test,:]
    x_raman = x_raman[idx_test,:]
    
    y_raman = y_test[idx_test]
    y_ir = y_test[idx_test]
    
    #y_test[y_test == 'LDPE'] = 'PE'
    
    poly_list = list(np.unique(np.concatenate([y_train_ir, y_train_raman, y_ir]))) 
    label_table = pd.DataFrame({'label':np.arange(len(poly_list)), 'polymer':poly_list})
    
    label_train_Raman = np.array(pd.DataFrame(y_train_raman).merge(label_table, left_on="polymer", right_on="polymer")['label'])
    label_train_IR = np.array(pd.DataFrame(y_train_ir).merge(label_table, left_on="polymer", right_on="polymer")['label'])
    
    label_ir = np.array(pd.DataFrame(y_ir).merge(label_table, how='left', left_on="polymer", right_on="polymer")['label'])
    
    label_raman = np.array(pd.DataFrame(y_raman).merge(label_table, how='left', left_on="polymer", right_on="polymer")['label'])
    
    return x_train_ir, x_train_raman, label_train_IR, label_train_Raman, x_ir, x_raman, label_ir, label_raman, label_table, ids_test


def experiment_fig_1_2(score_list, label_ir, alphas, N_sim, quant_prop=.2, cal_prop=.8, seed_init=0, n_th=400):
    df_res_lis = []
    df_res_sd_lis = []
    df_res_all_lis = []
    df_res_cov_all_lis = []
    df_res_all_mean_lis = []
    df_res_cov_all_mean_lis = []
    df_res_pred_set_all_lis = []
    df_res_labels_all = []
    df_res_cov_lis = []
    df_res_cov_sd_lis = []

    for alpha in tqdm(alphas):
        set_size = defaultdict(list)
        set_size_all = defaultdict(list)
        zero_sets = defaultdict(list)
        cov = defaultdict(list)
        cov_all = defaultdict(list)
        cov_cal = defaultdict(list)
        set_size_by_label_hqi = defaultdict(list)
        pred_set_all = defaultdict(list)
        val_labels_list = []
        
        K = 2
        U_list = gen_u(K, n_th=n_th, seed=0)
        seed = seed_init
        for l in range(N_sim):
            seed = seed + l

            #score_list, y_train, y_test, top_views = simulate_s_2(N, d, d_v, L, scale, K, N_train, seed)
            #top_views = np.array(acc_list)
            quant_smx_list = []
            rest_smx_list = []
            cal_smx_list = []
            val_smx_list = []


            for i in range(K):

                quant_smx, rest_smx, quant_labels, rest_labels, idx_quant = split_cal_test(score_list[i], label_ir, cal_prop=quant_prop, seed=seed)
                cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(rest_smx, rest_labels, cal_prop=cal_prop, seed=seed)
                

                quant_smx_list.append(quant_smx)
                rest_smx_list.append(rest_smx)
                cal_smx_list.append(cal_smx)
                val_smx_list.append(val_smx)

            n = cal_labels.shape[0]
            #n = quant_labels.shape[0]

            prediction_sets, empirical_coverage, U_quant_env = mvcp_2(quant_smx_list,
                                                             quant_labels,
                                                             cal_smx_list, #quant_smx_list,
                                                             cal_labels, #quant_labels,
                                                             val_smx_list, #rest_smx_list, 
                                                             alpha, 
                                                             val_labels, #rest_labels,
                                                             U=U_list[0])

            
            
            pred_set_all['mvcp'].append(prediction_sets)
            val_labels_list.append(val_labels)

            set_size['mvcp'].append(
                    prediction_sets.sum(1).mean()
                )
            set_size_all['mvcp'].append(
                    prediction_sets.sum(1)
                )
            cov_all['mvcp'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels]
                )
            zero_sets['mvcp'].append(
                    np.mean(prediction_sets.sum(1) == 0)
                )
            cov['mvcp'].append(
                    empirical_coverage
                )

            set_size_by_label_hqi['mvcp'].append(
                    pd.DataFrame({'label':val_labels, #rest_labels,
                                  'set_size':prediction_sets.sum(1)}).groupby('label').mean()
                )

            s_u = rest_smx_list[0]

            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
            #cal_smx, val_smx, cal_labels, val_labels = quant_smx_list[0], rest_smx_list[0], quant_labels, rest_labels
            n = cal_smx.shape[0]
            (prediction_sets, empirical_coverage), qhat = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            
            pred_set_all['view_IR'].append(prediction_sets)
            set_size['view_IR'].append(prediction_sets.sum(1).mean())

            set_size_all['view_IR'].append(
                    prediction_sets.sum(1)
                )
            cov_all['view_IR'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels]
                )

            zero_sets['view_IR'].append(np.mean(prediction_sets.sum(1) == 0))
            cov['view_IR'].append(empirical_coverage)

            set_size_by_label_hqi['view_IR'].append(
                    pd.DataFrame({'label':val_labels, 
                                  'set_size':prediction_sets.sum(1)}).groupby('label').mean()
                )

            s_u = rest_smx_list[1]

            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=0.8, seed=seed)
            #cal_smx, val_smx, cal_labels, val_labels = quant_smx_list[1], rest_smx_list[1], quant_labels, rest_labels
            n = cal_smx.shape[0]
            (prediction_sets, empirical_coverage), qhat = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            
            pred_set_all['view_Raman'].append(prediction_sets)
            set_size['view_Raman'].append(prediction_sets.sum(1).mean())

            set_size_all['view_Raman'].append(
                    prediction_sets.sum(1)
                )
            cov_all['view_Raman'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels]
                )

            zero_sets['view_Raman'].append(np.mean(prediction_sets.sum(1) == 0))
            cov['view_Raman'].append(empirical_coverage)
        
        df_res_pred_set_all_lis.append(pred_set_all)
        df_res_all_lis.append(set_size_all)
        df_res_cov_all_lis.append(cov_all)
        
        df_res_labels_all.append(val_labels_list)
        
        df_res_all_mean_lis.append(set_size)
        df_res_cov_all_mean_lis.append(cov)
        
        df_res = pd.DataFrame(pd.DataFrame(set_size).mean()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0).loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(set_size).std()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0).loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_sd_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(cov).mean()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0).loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_cov_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(cov).std()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0).loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_cov_sd_lis.append(df_res)
        
    return df_res_lis, df_res_sd_lis, df_res_cov_lis, df_res_cov_sd_lis, df_res_all_mean_lis, df_res_cov_all_mean_lis, df_res_all_lis, df_res_cov_all_lis, df_res_labels_all, df_res_pred_set_all_lis


def experiment_fig_3(score_list, label_ir, quality, alphas, N_sim, quant_prop=.2, cal_prop=.8, seed_init=0):
    df_res_lis = []
    df_res_sd_lis = []
    df_res_all_lis = []
    df_res_cov_lis = []
    df_res_cov_sd_lis = []
    
    df_res_cov_all_lis = []
    df_res_all_mean_lis = []
    df_res_cov_all_mean_lis = []
    

    for alpha in tqdm(alphas):
        set_size = defaultdict(list)
        set_size_all = defaultdict(list)
        zero_sets = defaultdict(list)
        cov = defaultdict(list)
        cov_all = defaultdict(list)
        cov_cal = defaultdict(list)
        set_size_by_label_hqi = defaultdict(list)        
        
        K = 2
        U_list = gen_u(K, seed=0)
        seed = seed_init
        for l in range(N_sim):
            seed = seed + l

            #score_list, y_train, y_test, top_views = simulate_s_2(N, d, d_v, L, scale, K, N_train, seed)
            #top_views = np.array(acc_list)
            quant_smx_list = []
            rest_smx_list = []
            cal_smx_list = []
            val_smx_list = []


            for i in range(K):

                quant_smx, rest_smx, quant_labels, rest_labels, idx_quant = split_cal_test(score_list[i], label_ir, cal_prop=quant_prop, seed=seed)
                quality_quant = quality[idx_quant]
                quality_rest = quality[~idx_quant]

                cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(rest_smx, rest_labels, cal_prop=cal_prop, seed=seed)
                
                quality_cal = quality_rest[idx]
                quality_val = quality_rest[~idx]
                
                quant_smx_list.append(quant_smx)
                rest_smx_list.append(rest_smx)
                cal_smx_list.append(cal_smx)
                val_smx_list.append(val_smx)

            n = cal_labels.shape[0]
            #n = quant_labels.shape[0]

            prediction_sets, empirical_coverage, _ = mvcp_2(quant_smx_list,
                                                             quant_labels,
                                                             cal_smx_list,# quant_smx_list,
                                                             cal_labels,#quant_labels,
                                                             val_smx_list,#rest_smx_list,
                                                             alpha, 
                                                             val_labels,#rest_labels,
                                                             U=U_list[0])


            set_size['mvcp_ideal'].append(
                    prediction_sets[quality_val=="ideal"].sum(1).mean()
                )
            set_size_all['mvcp_ideal'].append(
                    prediction_sets[quality_val=="ideal"].sum(1)
                )
            
            set_size['mvcp_low_ir'].append(
                    prediction_sets[quality_val=="low_ir"].sum(1).mean()
                )
            set_size_all['mvcp_low_ir'].append(
                    prediction_sets[quality_val=="low_ir"].sum(1)
                )
            
            set_size['mvcp_low_raman'].append(
                    prediction_sets[quality_val=="low_raman"].sum(1).mean()
                )
            set_size_all['mvcp_low_raman'].append(
                    prediction_sets[quality_val=="low_raman"].sum(1)
                )
            
            zero_sets['mvcp'].append(
                    np.mean(prediction_sets.sum(1) == 0)
                )
            
            
            empirical_coverage_ideal = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"])
            empirical_coverage_low_ir = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"])
            empirical_coverage_low_raman = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"])
            
            cov['mvcp_ideal'].append(
                    empirical_coverage_ideal
                )
            
            cov['mvcp_low_ir'].append(
                    empirical_coverage_low_ir
                )
            
            cov['mvcp_low_raman'].append(
                    empirical_coverage_low_raman
                )
            
            
            cov_all['mvcp_ideal'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"]
                )
            
            cov_all['mvcp_low_ir'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"]
                )
            
            cov_all['mvcp_low_raman'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"]
                )

            set_size_by_label_hqi['mvcp'].append(
                    pd.DataFrame({'label':val_labels,#rest_labels,
                                  'set_size':prediction_sets.sum(1)}).groupby('label').mean()
                )

            s_u = rest_smx_list[0]
                #print(top_views[:(j+1)])
                #print(np.argmax(top_views[:(j+1)]))

            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
            #cal_smx, val_smx, cal_labels, val_labels = quant_smx_list[0], rest_smx_list[0], quant_labels, rest_labels
            n = cal_smx.shape[0]
            (prediction_sets, empirical_coverage), qhat = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            
            set_size['view_IR_ideal'].append(prediction_sets[quality_val=="ideal"].sum(1).mean())
            set_size['view_IR_low_ir'].append(prediction_sets[quality_val=="low_ir"].sum(1).mean())
            set_size['view_IR_low_raman'].append(prediction_sets[quality_val=="low_raman"].sum(1).mean())
            
            set_size_all['view_IR_ideal'].append(
                    prediction_sets[quality_val=="ideal"].sum(1)
                )
            set_size_all['view_IR_low_ir'].append(
                    prediction_sets[quality_val=="low_ir"].sum(1)
                )
            
            set_size_all['view_IR_low_raman'].append(
                    prediction_sets[quality_val=="low_raman"].sum(1)
                )

            zero_sets['view_IR'].append(np.mean(prediction_sets.sum(1) == 0))
            
            empirical_coverage_ideal = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"])
            empirical_coverage_low_ir = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"])
            empirical_coverage_low_raman = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"])
            
            cov['view_IR_ideal'].append(empirical_coverage_ideal)
            cov['view_IR_low_ir'].append(empirical_coverage_low_ir)
            cov['view_IR_low_raman'].append(empirical_coverage_low_raman)
            
            cov_all['view_IR_ideal'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"]
                )
            
            cov_all['view_IR_low_ir'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"]
                )
            cov_all['view_IR_low_raman'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"]
                )
            
            set_size_by_label_hqi['view_IR'].append(
                    pd.DataFrame({'label':val_labels, 
                                  'set_size':prediction_sets.sum(1)}).groupby('label').mean()
                )

            s_u = rest_smx_list[1]
                #print(top_views[:(j+1)])
                #print(np.argmax(top_views[:(j+1)]))

            cal_smx, val_smx, cal_labels, val_labels, idx = split_cal_test(s_u, rest_labels, cal_prop=cal_prop, seed=seed)
            #cal_smx, val_smx, cal_labels, val_labels = quant_smx_list[1], rest_smx_list[1], quant_labels, rest_labels
            n = cal_smx.shape[0]
            (prediction_sets, empirical_coverage), qhat = conform_prediction(cal_smx, val_smx, cal_labels, n, alpha, val_labels, cond=0)
            
            set_size['view_Raman_ideal'].append(prediction_sets[quality_val=="ideal"].sum(1).mean())
            set_size['view_Raman_low_ir'].append(prediction_sets[quality_val=="low_ir"].sum(1).mean())
            set_size['view_Raman_low_raman'].append(prediction_sets[quality_val=="low_raman"].sum(1).mean())
            
            set_size_all['view_Raman_ideal'].append(
                    prediction_sets[quality_val=="ideal"].sum(1)
                )
            set_size_all['view_Raman_low_ir'].append(
                    prediction_sets[quality_val=="low_ir"].sum(1)
                )
            
            set_size_all['view_Raman_low_raman'].append(
                    prediction_sets[quality_val=="low_raman"].sum(1)
                )

            zero_sets['view_Raman'].append(np.mean(prediction_sets.sum(1) == 0))
            
            empirical_coverage_ideal = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"])
            empirical_coverage_low_ir = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"])
            empirical_coverage_low_raman = np.mean(prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"])
            
            cov['view_Raman_ideal'].append(empirical_coverage_ideal)
            cov['view_Raman_low_ir'].append(empirical_coverage_low_ir)
            cov['view_Raman_low_raman'].append(empirical_coverage_low_raman)
            
            cov_all['view_Raman_ideal'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="ideal"]
                )
            
            cov_all['view_Raman_low_ir'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_ir"]
                )
            
            cov_all['view_Raman_low_raman'].append(
                    prediction_sets[np.arange(len(val_labels)), val_labels][quality_val=="low_raman"]
                )
            
        df_res_all_lis.append(set_size_all)
        df_res_cov_all_lis.append(cov_all)
        
        df_res_all_mean_lis.append(set_size)
        df_res_cov_all_mean_lis.append(cov)
        
        
        df_res = pd.DataFrame(pd.DataFrame(set_size).mean()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0)#.loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(set_size).std()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0)#.loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_sd_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(cov).mean()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0)#.loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_cov_lis.append(df_res)

        df_res = pd.DataFrame(pd.DataFrame(cov).std()).reset_index()
        df_res['M'] = 1
        df_res = df_res.pivot(index='M', columns='index', values=0)#.loc[:,['view_IR','view_Raman', 'mvcp']]
        df_res = df_res.reset_index(drop=True)

        df_res_cov_sd_lis.append(df_res)

    return df_res_lis, df_res_sd_lis, df_res_cov_lis, df_res_cov_sd_lis, df_res_all_mean_lis, df_res_cov_all_mean_lis, df_res_all_lis, df_res_cov_all_lis

