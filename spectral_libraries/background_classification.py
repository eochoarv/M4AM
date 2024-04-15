#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:20:58 2023

@author: eochoa
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle

path = sys.argv[1]
spectra_df = pd.read_csv(path + 'data_all_spectral.csv')
del spectra_df['Unnamed: 0']

spectra_df

spectra_df_bg = spectra_df.iloc[:,:-4].loc[:, ~np.isnan(spectra_df.iloc[:1,:-4]).to_numpy().flatten()]
spectra_df_bg.iloc[np.where(np.isnan(spectra_df_bg.to_numpy()))[0],
np.where(np.isnan(spectra_df_bg.to_numpy()))[1]] = 0
#spectra_df_bg = np.array(spectra_df_bg)

kmeans_1 = pickle.load(open('bg_classifier_1.sav', 'rb'))
kmeans_2 = pickle.load(open('bg_classifier_2.sav', 'rb'))

pred = kmeans_1.predict(spectra_df_bg)

spectra_bg_1 = spectra_df.loc[pred == 0, 'id']

kmeans_2 = KMeans(n_clusters=10, random_state=0).fit(spectra_df_bg.loc[(pred == 0),:])
pred_2 = kmeans_2.predict(spectra_df_bg.loc[(pred == 0),:])

spectra_bg_2 = spectra_bg_1[np.array([x not in [1,2,3,4,5,7,9] for x in pred_2])]

spectra_df.query('id not in @spectra_bg_1').reset_index(drop=True).to_csv(path + 'data_non_background.csv')
spectra_df.query('id in @spectra_bg_1').reset_index(drop=True).to_csv(path + '/data_background.csv')