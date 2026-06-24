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
import Orange
from orangecontrib.spectroscopy import data
from orangecontrib.spectroscopy.data import getx
from Orange.data.io import FileFormat


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

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def read_spectra(self):
        channels = self.get_channels()

        for c, label in channels.items():
            if label.decode("utf-8") == self.sheet:
                self.data_signal = c
                break
        else:
            self.data_signal = list(channels.keys())[0]

        import h5py
        hdf5_file = h5py.File(self.filename,'r')
        keys = list(hdf5_file.keys())

        hyperspectra = False
        intensities = []
        wavenumbers = []
        x_locs = []
        y_locs = []

        # load measurements
        for meas_name in filter(lambda s: s.startswith('Measurement'), keys):
            hdf5_meas = hdf5_file[meas_name]

            meas_keys = list(hdf5_meas.keys())
            meas_attrs = hdf5_meas.attrs

            # check if this measurement contains the selected data channel
            selected_signal = False
            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                if hdf5_chan.attrs.keys().__contains__('DataSignal') and hdf5_chan.attrs['DataSignal'] == self.data_signal:
                    selected_signal = True
                    break
            if not selected_signal:
                continue

            # build range arrays
            spec_vals = []
            try:
                if meas_attrs.keys().__contains__('RangeWavenumberStart'):
                    wn_start = meas_attrs['RangeWavenumberStart'][0]
                    wn_end = meas_attrs['RangeWavenumberEnd'][0]
                    wn_points = meas_attrs['RangeWavenumberPoints'][0]
                    spec_vals = np.linspace(wn_start, wn_end, wn_points)
            except:
                raise IOError("Error reading wavenumber range from " + self.filename)

            pos_vals = []
            try:
                if meas_attrs.keys().__contains__('RangeXStart'):
                    x_start = meas_attrs['RangeXStart'][0]
                    x_points = meas_attrs['RangeXPoints'][0]
                    x_incr = meas_attrs['RangeXIncrement'][0]
                    x_end = x_start + x_incr * (x_points - 1)
                    x_min = min(x_start, x_end)
                    if meas_attrs.keys().__contains__('RangeYStart'):
                        y_start = meas_attrs['RangeYStart'][0]
                        y_points = meas_attrs['RangeYPoints'][0]
                        y_incr = meas_attrs['RangeYIncrement'][0]
                        y_end = y_start + y_incr * (y_points - 1)
                        y_min = min(y_start, y_end)

                        # construct the positions array
                        for iY in range(int(y_points)):
                            y = y_min + iY * abs(y_incr)
                            for iX in range(int(x_points)):
                                x = x_min + iX * abs(x_incr)
                                pos_vals.append([x, y])
                        pos_vals = np.array(pos_vals)
                else:
                    pos_vals = np.array([1])
            except:
                raise IOError("Error reading position data from " + self.filename)

            hyperspectra = pos_vals.shape[0] > 1

            # ignore backgrounds and unchecked data
            if not hyperspectra:
                if meas_attrs.keys().__contains__('IsBackground') and meas_attrs['IsBackground'][0]:
                    continue
                if meas_attrs.keys().__contains__('Checked') and not meas_attrs['Checked'][0]:
                    continue

            if len(wavenumbers) == 0:
                wavenumbers = spec_vals

            if hyperspectra:
                x_len = meas_attrs['RangeXPoints'][0]
                y_len = meas_attrs['RangeYPoints'][0]
                x_locs = pos_vals[:x_len,0]
                y_indices = np.round(np.linspace(0, pos_vals.shape[0] - 1, y_len)).astype(int)
                y_locs = pos_vals[y_indices,1]
            else:
                x_locs.append(meas_attrs['LocationX'][0])
                y_locs.append(meas_attrs['LocationY'][0])

            # load channels
            for chan_name in filter(lambda s: s.startswith('Channel'), meas_keys):
                hdf5_chan = hdf5_meas[chan_name]
                chan_attrs = hdf5_chan.attrs

                signal = chan_attrs['DataSignal']
                if signal != self.data_signal:
                    continue

                data = hdf5_chan['Raw_Data']
                if hyperspectra:
                    rows = meas_attrs['RangeYPoints'][0]
                    cols = meas_attrs['RangeXPoints'][0]
                    intensities = np.reshape(data, (rows,cols,data.shape[1])) # organized rows, columns, wavelengths
                    break
                else:
                    intensities.append(data[0,:])

        #intensities = np.array(intensities)
        #features = np.array(wavenumbers)
        intensities = intensities
        features = np.array(wavenumbers)
        x_locs = np.array(x_locs).flatten()
        y_locs = np.array(y_locs).flatten()
        if hyperspectra:
            return _spectra_from_image(intensities, features, x_locs, y_locs)
        else:
            spectra = intensities

            # locations
            x_loc = y_loc = np.arange(len(spectra))
            metas = np.array([x_locs[x_loc], y_locs[y_loc]]).T

            domain = Orange.data.Domain([], None,
                                        metas=[Orange.data.ContinuousVariable.make("map_x"),
                                               Orange.data.ContinuousVariable.make("map_y")]
                                        )
            data = Orange.data.Table.from_numpy(domain, X=np.zeros((len(spectra), 0)),
                                                metas=np.asarray(metas, dtype=object))
            return features, spectra, data