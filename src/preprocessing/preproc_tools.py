# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:56:39 2023

@author: WookS
"""
from scipy import signal
from scipy.interpolate import interp1d
# from scipy.interpolate import pchip
from scipy.stats import linregress
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d
import pywt

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor



def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, use_lfilter=True):
    
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filter_func = signal.lfilter if use_lfilter else signal.filtfilt   
    y = filter_func(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5, use_lfilter=True):
    
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    b, a = butter_lowpass(cutoff, fs, order=order)
    filter_func = signal.lfilter if use_lfilter else signal.filtfilt   
    y = filter_func(b, a, data)
    return y


def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


def ms_to_npts(ms, sfreq):
    # return int(sfreq * ms // 1000)
    return int(round(sfreq * ms / 1000))

def npts_to_ms(npts, sfreq):
    return npts * 1000 / sfreq


def make_odd(x):
    if x % 2 == 0:  # Check if x is even
        x += 1  # Increment x by 1 to make it odd
    return x


def median_normalisation(data):
    median_data = np.median(data)
    data -= median_data
    return data


def mean_normalisation(data):
    mean_data = np.mean(data)
    data -= mean_data
    return data


def detrend_signal(time, input_signal):
    # import pdb; pdb.set_trace()
    # Fit a linear trend
    slope, intercept, _, _, _ = linregress(time, input_signal)
    
    # Subtract the linear trend
    detrended_signal = input_signal - (slope * time + intercept)
    
    return detrended_signal




# def interpolate_outliers(series, threshold=3, method='linear', show_plot=False):
#     # import pdb; pdb.set_trace()
#     z_scores = np.abs((series - series.mean()) / series.std())
#     outliers = z_scores > threshold
#     if show_plot:
#         plt.figure()
#         plt.plot(series, label='Original Data')
#     num_outliers_removed = sum(outliers)
#     series[outliers] = np.nan
#     series.interpolate(method=method, limit_direction='both', inplace=True)
#     if show_plot:
#         plt.plot(series, label='Interpolated Data')
#         plt.title(f'Number of Outliers Removed: {num_outliers_removed}')
#         plt.legend()  # Add legend to the plot
#     return series, num_outliers_removed


def interpolate_outliers(series, threshold=3, method='pchip', show_plot=False):
    data = series.values  # Convert to a numpy array for performance
    
    # Calculate z-scores using numpy
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    
    # Identify outliers using numpy
    outliers = z_scores > threshold
    
    if show_plot:
        plt.figure()
        plt.plot(data, label='Original Data')
    
    num_outliers_removed = np.sum(outliers)
    
    # Replace outliers with NaN using numpy
    data[outliers] = np.nan
    
    # Create a pandas Series again
    series = pd.Series(data)
    
    # Interpolate missing values using pandas
    series.interpolate(method=method, limit_direction='both', inplace=True)
    
    if show_plot:
        plt.plot(series, label='Interpolated Data')
        plt.title(f'Number of Outliers Removed: {num_outliers_removed}')
        plt.legend()
    
    return series, num_outliers_removed


# def remove_spikes_by_group(df, grouping, input_signal, clean_method, medfilt_size, 
#                            titration_thresholds=[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10], 
#                            show_plot=False, outvar_suffix='_cleaned'):
#     # import pdb; pdb.set_trace()
#     if clean_method == 'medfilt':
#         df[f'{input_signal}{outvar_suffix}'] = df.groupby(grouping)[input_signal].transform(signal.medfilt, medfilt_size)
    
#     # thresholds = [20, 19, 18, 17, 16, 15, 14, 13, 12]
#     elif clean_method == 'titration_thresholding':
   
#         df[f'{input_signal}{outvar_suffix}'] = df[input_signal].copy()
    
#         for threshold in titration_thresholds:
#             grouped = df.groupby(grouping)
#             df[f'{input_signal}{outvar_suffix}'] = grouped.apply(
#                 lambda group: interpolate_outliers(
#                     group[f'{input_signal}{outvar_suffix}'].copy(), threshold=threshold, method='pchip', show_plot=show_plot)[0]
#                 ).reset_index(level=0, drop=True)            
            
#     return df



def remove_spikes(df, input_signal, clean_method, medfilt_size=3, 
                           titration_thresholds=[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10], 
                           show_plot=False):
    
    def medfilt_and_plot(signal_to_be_processed, medfilt_size, show_plot=True):
        if show_plot:
            plt.figure()
            plt.plot(signal_to_be_processed, label='Original Data')
            
        signal_to_be_processed = signal.medfilt(signal_to_be_processed, medfilt_size)
        
        if show_plot:
            plt.plot(signal_to_be_processed, label='Median-Filtered Data')
            plt.legend()
        
        return signal_to_be_processed
            
    
    # import pdb; pdb.set_trace()
    if df is None:
        signal_to_be_processed = input_signal
    else:
        signal_to_be_processed = df[input_signal].values
    

    # thresholds = [20, 19, 18, 17, 16, 15, 14, 13, 12]
    if clean_method == 'titration_thresholding':
        
        with tqdm(titration_thresholds) as progress_bar:
            for threshold in progress_bar:
                # Update the description
                progress_bar.set_description("Performing titration thresholding")
                
                # Your existing code for processing the signal goes here
                signal_to_be_processed = interpolate_outliers(pd.Series(signal_to_be_processed), threshold=threshold, method='pchip', show_plot=show_plot)[0]
             
                
             
    elif clean_method == 'medfilt':
        signal_to_be_processed = medfilt_and_plot(signal_to_be_processed, medfilt_size, show_plot=show_plot)
        
        
        
        
    elif clean_method == 'titration_thresholding_then_medfilt':
        
        with tqdm(titration_thresholds) as progress_bar:
            for threshold in progress_bar:
                # Update the description
                progress_bar.set_description("Performing titration thresholding")
                
                # Your existing code for processing the signal goes here
                signal_to_be_processed = interpolate_outliers(pd.Series(signal_to_be_processed), threshold=threshold, method='pchip', show_plot=show_plot)[0]
        
        signal_to_be_processed = medfilt_and_plot(signal_to_be_processed, medfilt_size, show_plot=show_plot)
        
        
        
        
    elif clean_method == 'medfilt_then_titration_thresholding':
        
        signal_to_be_processed = medfilt_and_plot(signal_to_be_processed, medfilt_size, show_plot=show_plot)
        
        with tqdm(titration_thresholds) as progress_bar:
            for threshold in progress_bar:
                # Update the description
                progress_bar.set_description("Performing titration thresholding")
                
                # Your existing code for processing the signal goes here
                signal_to_be_processed = interpolate_outliers(pd.Series(signal_to_be_processed), threshold=threshold, method='pchip', show_plot=show_plot)[0]
        

    return np.array(signal_to_be_processed)



def check_for_missing_values(df, input_signal):
    missing_values = df[df.isna().any(axis=1)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[input_signal], marker='o', linestyle='-')
    plt.xlabel('index')
    plt.ylabel('Value')
    
    # Add vertical lines for missing values
    for my_index in missing_values.index:
        plt.axvline(x=my_index, color='red', linestyle='--', linewidth=1)
        
        
        
# Common Average Referencing
def common_average_reference(rec1, rec2, method='mean'):
    
    data_stacked = np.vstack((rec1, rec2))
    
    if method == 'mean':
        mean_median_func = np.mean
    elif method == 'median':
        mean_median_func = np.median
    else:
        raise ValueError("'method' should be either 'mean' or 'median'!")
        
    # Calculate the mean across all channels for each time point
    mean_or_median_across_channels = mean_median_func(data_stacked, axis=0)
    
    # Subtract the mean from each channel
    data_car = data_stacked - mean_or_median_across_channels
    
    # Separate the CAR-corrected data back into individual recordings
    rec1, rec2 = data_car[0], data_car[1]
    
    return rec1, rec2


## overly simplistic?
def perform_laplacian_referencing(rec1, rec2):
    # Calculate the Laplacian-referenced data
    rec_center = (rec1 + rec2) / 2
    laplacian_data = rec1 - 2 * rec_center + rec2
    
    # Reverse Laplacian referencing to recover original data
    rec1_recovered = 2 * rec_center - laplacian_data
    rec2_recovered = rec1_recovered + laplacian_data
    
    return rec1_recovered, rec2_recovered, laplacian_data


def calculate_laplacian_referenced_signals(signal1, signal2, electrode_spacing_mm=3.25, sigma=1.0):
    """
    Calculate Laplacian-referenced signals for two peripheral nerve recordings.

    Args:
    signal1 (numpy.ndarray): The first peripheral nerve recording.
    signal2 (numpy.ndarray): The second peripheral nerve recording.
    electrode_spacing_mm (float): The distance between electrodes in millimeters.
    sigma (float): Standard deviation for the Gaussian kernel (controls smoothing).

    Returns:
    (numpy.ndarray, numpy.ndarray): Laplacian-referenced signals for signal1 and signal2.
    """
    # Create a Gaussian-weighted Laplacian kernel
    kernel_size = int(6 * sigma * electrode_spacing_mm)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.linspace(-3 * sigma * electrode_spacing_mm, 3 * sigma * electrode_spacing_mm, kernel_size)
    gaussian_kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma) ** 2)
    laplacian_kernel = np.gradient(np.gradient(gaussian_kernel))

    # Apply the Laplacian kernel to both signals
    laplacian_signal1 = convolve2d(signal1.reshape(1, -1), laplacian_kernel.reshape(1, -1), mode='same')[0]
    laplacian_signal2 = convolve2d(signal2.reshape(1, -1), laplacian_kernel.reshape(1, -1), mode='same')[0]

    return laplacian_signal1, laplacian_signal2



def run_singular_value_decomposition(rec1, rec2, threshold=0.1):
    """
    Remove muscle artifact noise from two peripheral nerve recordings using SVD.

    Args:
    rec1 (numpy.ndarray): The first peripheral nerve recording.
    rec2 (numpy.ndarray): The second peripheral nerve recording.
    threshold (float): Threshold for retaining significant components based on singular values.
                      Components with singular values greater than this threshold are retained.

    Returns:
    (numpy.ndarray, numpy.ndarray): Denoised recordings for rec1 and rec2.
    """
    # Concatenate the two recordings into a data matrix
    data_matrix = np.vstack((rec1, rec2)).T  # Transpose to have time samples as rows

    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(data_matrix, full_matrices=False)

    # Determine the number of significant components to retain
    num_significant_components = np.sum(S > threshold)

    # Keep only the significant components in U, S, and VT
    U_reduced = U[:, :num_significant_components]
    S_reduced = np.diag(S[:num_significant_components])
    VT_reduced = VT[:num_significant_components, :]

    # Reconstruct the denoised data matrix
    denoised_data_matrix = U_reduced @ S_reduced @ VT_reduced

    # Extract the denoised recordings from the denoised data matrix
    denoised_rec1 = denoised_data_matrix[:, 0]
    denoised_rec2 = denoised_data_matrix[:, 1]

    return denoised_rec1, denoised_rec2



def pca_denoise(rec1, rec2, num_components_to_retain=1, wavelet_threshold=0.1):
    """
    PCA and Wavelet Transform for denoising peripheral nerve recordings.

    Args:
    rec1 (numpy.ndarray): The first peripheral nerve recording.
    rec2 (numpy.ndarray): The second peripheral nerve recording.
    num_components_to_retain (int): Number of principal components to retain during PCA.
    wavelet_threshold (float): Threshold for wavelet denoising.

    Returns:
    (numpy.ndarray, numpy.ndarray): Denoised recordings for rec1 and rec2.
    """
    # import pdb; pdb.set_trace()
    # Combine the two recordings into a data matrix
    data_matrix = np.vstack((rec1, rec2)).T

    # Step 1: Perform PCA to denoise the data matrix
    mean_centered_data_matrix = data_matrix - np.mean(data_matrix, axis=0)
    cov_matrix = np.cov(mean_centered_data_matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :num_components_to_retain]
    denoised_data_matrix = mean_centered_data_matrix.dot(principal_components).dot(principal_components.T) + np.mean(data_matrix, axis=0)

    # Step 2: Perform wavelet denoising on the denoised data
    # denoised_rec1 = pywt.threshold(denoised_data_matrix[:, 0], wavelet_threshold, mode='soft')
    # denoised_rec2 = pywt.threshold(denoised_data_matrix[:, 1], wavelet_threshold, mode='soft')

    return denoised_data_matrix[:, 0], denoised_data_matrix[:, 1]
