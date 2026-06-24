from pathlib import Path
import os
import numpy as np
import pandas as pd
from typing import Tuple
import json
import shutil
import random
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import glob
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline, splrep, splev
from scipy import stats

import Orange ###pip install orange3; pip install Orange3-Associate
from Orange.data.io import FileFormat
from orangecontrib.spectroscopy import data ###pip install Orange-Spectroscopy
from orangecontrib.spectroscopy.data import getx
from utils import *

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
            return ''#_spectra_from_image(intensities, features, x_locs, y_locs)
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


####Dataset Generate
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_valid(y: NDArray[np.float32]):
    assert y.ndim == 1, f"y should be an 1-D array"
    rng = np.random.default_rng()
    return rng

def minmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    is_valid(x)
    return (x - x.min()) / (x.max() - x.min())

def standardize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    is_valid(x)
    return (x - x.mean()) / x.std()

def adj_negative(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x = x + abs(x.min()) if x.min() < 0 else x
    return x

def iModPolyFit(x, y, n: int):
    if not all([isinstance(x, np.ndarray), isinstance(y, np.ndarray)]):
        x, y = np.array(x), np.array(y)
    wavesOriginal = x
    devPrev = 0
    firstIter = True
    criteria = False
    while not criteria:
        paramVector = np.polynomial.Polynomial.fit(x, y, n)
        mod_poly = paramVector(x)
        residuals = y - mod_poly
        dev_curr = residuals.std(ddof=1)

        if firstIter:
            peaks = []
            for i in range(y.shape[0]):
                if y[i] > mod_poly[i] + dev_curr:
                    peaks.append(i)
            peaks = np.array(peaks)
            if peaks.shape[0]>0:
                y = np.delete(y, peaks)
                mod_poly = np.delete(mod_poly, peaks)
                x = np.delete(x, peaks)
            else:
                return np.zeros((len(y),))
            firstIter = False
        for j in range(y.shape[0]):
            if mod_poly[j] + dev_curr > y[j]:
                pass
            else:
                y[j] = mod_poly[j]
        criteria = abs((dev_curr - devPrev) / dev_curr) <= 0.05

        if criteria:
            t = np.interp(wavesOriginal, x, y)
            return t.flatten()
        devPrev = dev_curr


def specy_prep(wavenum, absorb, minmax_ind: bool = False, smooth: int = 0, baseline: int = 0, adj_neg: bool = False):
    if adj_neg:
        absorb = adj_negative(absorb)
    if smooth == 0:
        smoothed = absorb
    else:
        smoothed: np.ndarray = savgol_filter(x=absorb,window_length=11,polyorder=smooth,)
    if baseline != 0:
        baselineRemove = smoothed - iModPolyFit(wavenum, smoothed, baseline)
    else:
        baselineRemove = smoothed
    if minmax_ind == True:
        baselineRemove = minmax(baselineRemove)
    else:
        baselineRemove = standardize(baselineRemove)
    return baselineRemove


def rubberband(x, y, minmax_ind: bool = False,):
    x = x.reshape(-1,1)
    temp = y.reshape(-1,1)
    merged = np.concatenate((x, temp), axis = 1)
    v = ConvexHull(merged).vertices
    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    # Leave only the ascending part
    v = v[:v.argmax()]
    x_final = merged[:,0]
    y_final = merged[:,1]
    # Create baseline using linear interpolation between vertices
    rubber = np.interp(x_final, x_final[v], y_final[v])
    y_final = y_final - rubber
    return standardize(y_final)

def joe_rubberband(wavelengths, y, upper = False, noise = 0, spline = True, minmax_ind: bool = False,):
    if upper:
        y = -y ###negate, and then flip back after performing the rubberband
    y =  joe_dot_rubberband(wavelengths, y, noise = noise, spline = spline)
    if upper:
        y = -y
    if minmax_ind:
        return minmax(y)
    else:
        return standardize(y)

def joe_dot_rubberband(x, y, noise, spline):
    x = x.reshape(-1,1)
    temp = y.reshape(-1,1)
    merged = np.concatenate((temp, x), axis = 1)
    v = list(ConvexHull(merged).vertices)
    v.reverse()
    v = np.array(v)
    v = np.roll(v, -v.argmin())
    v = v[:v.argmax()]
    y_final = merged[:,0]
    x_final = merged[:,1]
    # Create baseline using linear interpolation between vertices
    rubber = np.interp(x_final, x_final[v], y_final[v])
    y_flat = y.flatten()
    if spline:
        result = [a-b for a, b in zip(y_flat, rubber)]
        idxs_small = [i for i in range(len(result)) if result[i] <=0]
        num_idxs = len(idxs_small)
        if num_idxs >3:
            ###Apply a smoothing spline to these points
            tmp_spline = UnivariateSpline(x[idxs_small], y_flat[idxs_small], k=3)
            y_spline = tmp_spline(x).flatten() # Predicted y values using the spline
        elif num_idxs >1:
            tmp_spline = UnivariateSpline(x[idxs_small], y_flat[idxs_small], k=1)
            y_spline = tmp_spline(x).flatten() 
        else:
            y_spline = rubber ##use the functional rubberband correction instead.
        y_final = y_flat - y_spline
    else:
        y_final = y_flat - rubber
    return y_final

def nitrogen_peak_calibration(xa, ya):
    lower_normality = min(enumerate(xa), key=lambda x:abs(x[1]-2280))[0]
    upper_normality = min(enumerate(xa), key=lambda x:abs(x[1]-2380))[0]
    center_idx = min(enumerate(xa), key=lambda x:abs(x[1]-2330))[0] - lower_normality
    upper_idx = min(enumerate(xa), key=lambda x:abs(x[1]-2360))[0] - lower_normality
    lower_idx = min(enumerate(xa), key=lambda x:abs(x[1]-2300))[0] - lower_normality
    idxs_to_keep = list(range(lower_normality,upper_normality + 1))
    ya_subset = ya[idxs_to_keep]
    row_max_idx = np.argmax(ya_subset)
    ya = list(ya)
    if(row_max_idx==center_idx):
        ###No transformation needed. 
        new_spectra = ya
    elif (row_max_idx>center_idx and row_max_idx<=upper_idx):
        ###left shift. example: idx = 1
        elements_to_chop_left = row_max_idx-center_idx
        new_spectra = ya[elements_to_chop_left:] + ya[-1:]*elements_to_chop_left
    elif(row_max_idx>=lower_idx and row_max_idx<center_idx):
        ###right shift. example: idx = 5
        elements_to_chop_right = center_idx-row_max_idx
        new_spectra = ya[:1]*elements_to_chop_right + ya[:-elements_to_chop_right]
    else:
        ##expanded check. example: idx = 0, 21
        if max(ya_subset)==min(ya_subset): ###That is the only case where there isn't a unique max; I checked.
            new_spectra = ya
        else:
            non_min_val_loc = [idx for idx in range(len(ya_subset)) if ya_subset[idx] != min(ya_subset)]
            data2 = [ya_subset[z] for z in non_min_val_loc]
            data4 = [(element - np.mean(data2)) / np.std(data2) for element in data2]
            D3, p3 = stats.kstest(data4, "norm")
            if p3 > 0.05:
                ###Go ahead with a normal shifting procedure based on the appropriate direction.
                if row_max_idx>center_idx:
                    elements_to_chop_left = row_max_idx-center_idx
                    new_spectra = ya[elements_to_chop_left:] + ya[-1:]*elements_to_chop_left
                else:
                    elements_to_chop_right = center_idx-row_max_idx
                    new_spectra = ya[:1]*elements_to_chop_right + ya[:-elements_to_chop_right]
            else:
                new_spectra = ya
    return np.array(new_spectra)


def silicon_peak_calibration(xa, ya, si_peak = 520):
    center_idx = min(enumerate(xa), key=lambda x:abs(x[1]-520))[0] ###520 = true silicon peak
    silicon_peak_pos = min(enumerate(xa), key=lambda x:abs(x[1]-si_peak))[0]
    ya = list(ya)
    if(silicon_peak_pos==center_idx):
        ###No transformation needed. 
        new_spectra = ya
    elif (silicon_peak_pos>center_idx):
        ###left shift. example: idx = 1
        elements_to_chop_left = silicon_peak_pos-center_idx
        new_spectra = ya[elements_to_chop_left:] + ya[-1:]*elements_to_chop_left
    elif(silicon_peak_pos<center_idx):
        ###right shift. example: idx = 5
        elements_to_chop_right = center_idx-silicon_peak_pos
        new_spectra = ya[:1]*elements_to_chop_right + ya[:-elements_to_chop_right]
    return np.array(new_spectra)


###Functions updated/replaced: dict to interpolated array, spectral preproc loop, table to dict, preprocess data ir, 
###preprocess data raman, transform csv to single view, create raman ir merges.  


def export_preprocessed_data(export_file_name, len_vector, interpolated_spectra, XX, transpose = False):
    labels = np.array(len_vector).reshape(len(len_vector),1)
    merged = np.concatenate((labels, interpolated_spectra), axis = 1)
    merged = pd.DataFrame(merged)
    col_list = ["Label"] + list(XX)
    merged.columns = col_list
    if transpose == True:
        merged = merged.transpose()
    merged.to_csv(export_file_name)
    return merged

def ptir_studio_to_csv(filename, spectral_type):
    absolute_filename = FileFormat.locate(filename)
    reader = FileFormat.get_reader(absolute_filename)
    if spectral_type == "IR":
        sheet = 'OPTIR (mV)'
    else:
        sheet = 'Raman Intensity'
    reader.select_sheet(sheet)
    spectra = read_spectra(reader)
    spectra_data = pd.DataFrame(spectra[1])
    coords = spectra[2].to_pandas_dfs()[2].reset_index(drop=True)
    spectra_data = spectra_data.reset_index(drop=True)
    spectra_data.columns = spectra[0]
    ###Will work on a better solution later, but this should work for now
    label= ["0" if "PSL" in filename else "1" if "PE" in filename else "2" if "PP" in filename
                    else "3" if "NH3SO5" in filename else "4" if "NaNO3" in filename else "5" if "sucrose" in filename
                    else "6"]
    label_list = label * spectra_data.shape[0]
    return spectra_data, coords, label_list

def check_matching_coordinates(coords_IR, coords_Raman):
    IR_coords_df = pd.DataFrame(coords_IR)
    Raman_coords_df = pd.DataFrame(coords_Raman)
    if IR_coords_df.shape != Raman_coords_df.shape:
        return [-99]
    mismatch_list = []
    for row in range(IR_coords_df.shape[0]):
        for column in range(2):
            if IR_coords_df.iloc[row,column] != Raman_coords_df.iloc[row, column]:
                mismatch_list.append(row)
    return mismatch_list

def calibrate_and_interpolate(args, df):
    if args.no_interpolate == False:
        XX = np.arange(args.range_min, args.range_max, args.interpolate_interval).astype(np.float32)
    else:
        XX = np.array(df.columns)
    xa_array = np.array(df.columns)
    xa = df.columns
    data_mpid_0 = np.array(df)
    z = data_mpid_0.shape[0]
    interpolated_spectra = np.zeros((z, XX.shape[0]))
    interpolated_spectra = interpolated_spectra.copy()
    for i in range(z):
        ya = data_mpid_0[i,]
        if args.spectral_type == "Raman":
            if args.rubberband == False:
                ya = specy_prep(xa_array, ya, minmax_ind = args.minmax, smooth=args.smooth, baseline=args.baseline)
            else:
                ya = joe_rubberband(xa_array, ya, minmax_ind = args.minmax)
                #ya = rubberband(xa_array, ya, minmax_ind = args.minmax)
            if args.nitrogen_calibration == True:
                ya = nitrogen_peak_calibration(xa, ya)
            ya = silicon_peak_calibration(xa, ya, args.si_peak)
        if args.no_interpolate == False:
            ya = np.interp(XX, xa, ya)
        interpolated_spectra[i,] = ya
    return interpolated_spectra, XX

def preprocess_data_ir(args, ir_data, export_filename):
    ir_2, tmp = calibrate_and_interpolate(args, ir_data) 
    idxs1 = [index for index in range(len(tmp)) if (tmp[index]>=948 and tmp[index]<=1860)] 
    idxs2 = [index for index in range(len(tmp)) if (tmp[index]>=2698 and tmp[index]<=3002)] 
    subset = idxs1 + idxs2
    XX = tmp[subset]
    ir_subset = ir_2[:,subset].copy()
    for row in range(ir_subset.shape[0]):
        if args.minmax:
            ir_subset[row,:] = minmax(ir_subset[row,:])
        else:
            ir_subset[row,:] = standardize(ir_subset[row,:])
    rng = np.random.default_rng()
    return ir_subset, XX, rng

def preprocess_data_raman(args, raman_data, export_filename):
    raman_2, XX = calibrate_and_interpolate(args, raman_data)
    rng = np.random.default_rng()
    return raman_2, XX, rng

def remove_na_rows(ir_2, raman_2, raman_labels):
    na_rows_ir = np.isnan(ir_2).all(axis = 1)
    valid_row_idxs_ir = [i for i, x in enumerate(na_rows_ir) if not x]
    ir_2 = ir_2.iloc[valid_row_idxs_ir,:]
    raman_2 = raman_2.iloc[valid_row_idxs_ir,:]
    na_rows_raman = np.isnan(raman_2).any(axis = 1)
    valid_row_idxs_raman = [i for i, x in enumerate(na_rows_raman) if not x]
    ir_2 = ir_2.iloc[valid_row_idxs_raman,:]
    raman_2 = raman_2.iloc[valid_row_idxs_raman,:]
    ###Update labels also by removing as needed
    raman_labels = [raman_labels[x] for x in valid_row_idxs_ir]
    raman_labels = [raman_labels[x] for x in valid_row_idxs_raman]
    return ir_2, raman_2, raman_labels


def ptir_to_interpolated_processed_csv(args, directory_name, ir_export_filename, raman_export_filename, file_list = [], transpose = True):
    directory = os.fsencode(directory_name)
    directory2 = os.fsdecode(directory)
    label_list = []
    merge_frames_exist = False ###creating merge frames on the 1st pass once know XX_ir and XX_raman.
    
    ###If no file list is provided, include all files in directory. 
    if file_list == []:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            file_list.append(filename)
    
    for filename in file_list:
        filename2 = '%s%s' %(directory2, filename)
        ir_data, ir_coords, ir_labels = ptir_studio_to_csv(filename2, "IR")
        raman_data, raman_coords, raman_labels = ptir_studio_to_csv(filename2, "Raman")
        check = check_matching_coordinates(ir_coords, raman_coords)
        if check != []:
            print("Row Mismatch", filename2)
        else:
            ###Set args for spectral type, range min, range max, and interpolate interval
            ###Run NA checks before merging
            ir_1, raman_1, raman_labels = remove_na_rows(ir_data, raman_data, raman_labels)
            if (ir_1.shape[0]!= ir_data.shape[0] or raman_1.shape[0]!= raman_data.shape[0]):
                print("NA Rows Removed but no other issues", filename2)
                
            args.spectral_type, args.range_min = "Raman", args.range_min_raman
            args.range_max, args.interpolate_interval = args.range_max_raman, args.interpolate_interval_raman
            ###The order within this function is: (1) Perform Baseline correction, (2) Perform Nitrogen Peak Calibration, 
            ##(3) Perform Silicon Peak Calibration, (4) Interpolate, and (5) export to the filename 
            raman_2, XX_raman, rng = preprocess_data_raman(args, raman_1, raman_export_filename)
            ###Update args for spectral type, range min, range max, and interpolate interval
            args.spectral_type, args.range_min = "IR", args.range_min_ir
            args.range_max, args.interpolate_interval = args.range_max_ir, args.interpolate_interval_ir
            ###The order within this function is: (1) Interpolate, (2) subset to omit the middle region of data that isn't 
            ###collected by M4AM, and (4) export to the filename 
            ###Don't need to use any of the saved arguments from the function; those are just there for use in later steps,
            ###and so that you don't have to see all of the input printing to the screen. 
            ir_2, XX_ir, rng = preprocess_data_ir(args, ir_1, ir_export_filename)
            
            if merge_frames_exist == False:
                    raman_merge_frame = np.zeros((0, len(XX_raman)))
                    ir_merge_frame = np.zeros((0, len(XX_ir)))
                    merge_frames_exist = True
            
            ###If we don't interpolate, and there are differences in the # of wave numbers, we'll run into an issue here.
            ###If there are the same number of wave # but the wave #'s are different, the wave #'s of the 1st will be used.
            raman_merge_frame = np.concatenate((raman_merge_frame,raman_2)) 
            ir_merge_frame = np.concatenate((ir_merge_frame,ir_2)) 
            label_list.append(raman_labels) ###the labels will be the same for both Raman & IR, so doesn't matter.
    ###Flatten list of lists
    label_list_flat = [x for xs in label_list for x in xs]
    label_list_flat = [int(element) for element in label_list_flat]
    export_preprocessed_data(ir_export_filename, label_list_flat, ir_merge_frame, XX_ir, transpose)
    export_preprocessed_data(raman_export_filename, label_list_flat, raman_merge_frame, XX_raman, transpose)

    return raman_merge_frame, ir_merge_frame, label_list_flat, XX_ir, XX_raman, rng