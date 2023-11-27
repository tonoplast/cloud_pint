# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 03:45:28 2023

@author: sungw
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
# from scipy import signal
from src.preprocessing.preproc_tools import butter_bandpass_filter
from src.preprocessing.cross_correlation import movingxcor, get_peak_output, peak_output_into_df, plot_xcor_info #get_correlation #, check_rho_based_on_peak_detection_window
from src.preprocessing.preproc_tools import remove_spikes, median_normalisation, downsample, common_average_reference, perform_laplacian_referencing, calculate_laplacian_referenced_signals, pca_denoise
# from src.utils.feature_extractor import get_features_from_timeseries
from tqdm import tqdm





# Time parameters for bladder pressure
sampling_rate_bladder = 30_000  # Sampling rate for bladder pressure in Hz
duration_bladder = 100  # Duration of the bladder pressure recording in seconds

# Time parameters for neural recording
sampling_rate_neural = 30_000  # Sampling rate for neural recording in Hz
duration_neural = 100  # Duration of the neural recording in seconds

lag_ms = -0.5

time_bladder = np.linspace(0, duration_bladder, int(sampling_rate_bladder * duration_bladder), endpoint=False)

# Simulate bladder pressure variations (as in your code)
pressure = np.zeros(len(time_bladder))
# Define events such as filling and voiding
filling_start_time = duration_bladder * 0.1
filling_end_time = duration_bladder * 0.5
voiding_start_time = duration_bladder * 0.6
# Simulate bladder filling phase with an exponential rate
filling_duration = filling_end_time - filling_start_time
filling_rate = np.exp(np.linspace(0, 4, int(filling_duration * sampling_rate_bladder)))
pressure[(time_bladder >= filling_start_time) & (time_bladder < filling_end_time)] = (
    filling_rate / filling_rate[-1]
)
# Simulate voiding phase
voiding_rate = -0.2
voiding_duration = duration_bladder - voiding_start_time
voiding_pressure = np.arange(0, voiding_duration * sampling_rate_bladder) * voiding_rate / sampling_rate_bladder
voiding_pressure[voiding_pressure < 0] = 0
pressure[time_bladder >= voiding_start_time] = voiding_pressure
# Add noise to the bladder pressure recordings

noise_amplitude = 0.01
pressure += np.random.normal(0, noise_amplitude, len(time_bladder))


time_neural = np.linspace(0, duration_neural, int(sampling_rate_neural * duration_neural), endpoint=False)



# Simulate sporadic action potentials (spikes) with voiding peak time alignment
spike_times = np.sort(np.random.uniform(filling_end_time - 0.1, filling_end_time + 0.1, 20))  # Generate 20 random spike times around the peak time
spike_amplitude = 1.0  # Amplitude of the spikes
spikes = np.zeros(len(time_neural))

# Place spikes at the specified times
spike_indices = np.round(spike_times * sampling_rate_neural).astype(int)
spikes[spike_indices] = spike_amplitude

bladder = spikes

# Simulate baseline noise for neural recording
# Combine baseline noise and spikes to simulate the first neural recording
bladder += np.random.normal(0, 0.1, len(time_neural))

# Create a second neural recording with a 0.5 ms lag
lag_samples = int(lag_ms/1000 * sampling_rate_neural)  # 0.5 ms lag in samples
spine = np.roll(bladder, lag_samples)  # Shift the neural1 signal by the lag

# Combine the two neural recordings
# neural_combined = neural1 + neural2

# Repeat the signals twice by concatenating them
time_bladder = np.concatenate([time_bladder, time_bladder + duration_bladder])
pressure = np.concatenate([pressure, pressure])
time_neural = np.concatenate([time_neural, time_neural + duration_neural])
bladder = np.concatenate([bladder, bladder])
spine = np.concatenate([spine, spine])


# # Plot the simulated nerve recording
# plt.figure(figsize=(10, 4))
# plt.plot(time_neural, bladder, label="Nerve Recording (Bladder)")
# plt.plot(time_neural, spine, label="Nerve Recording (Spine)")

# # Plot the simulated bladder pressure recording
# plt.plot(time_bladder, pressure, label="Bladder Pressure")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.title(f"Simulated Bladder Pressure Recording with Two Neural Recordings and a {lag_ms} ms Lag (Repeated Twice)")
# plt.grid(True)
# plt.show()






clean_raw_data = True
clean_xcor_matrix = False





save_dirnames_and_peak_windows = {
    ## Data type: [[peak window], [non_moving_peak, grouped_peak, individual_peak, first_on_left_peak]]
    # 'StaticPeak_0.0': [[-1.0, 1.0], [False, False, False, False]],
    # 'StaticPeak_0.5': [[0.49, 0.51], [False, False, False, False]],
    # 'MovingPeak': [[0.1, 0.65], [False, False, False, False]],
    # 'MovingPeakFirstOnLeft': [[-1.0, 1.0], [False, False, False, True]], ## range is irrelevant but required -- need to capture max and find left
    'NonMovingPeak': [[0.1, 0.65], [True, True, False, False]],
    # 'NonMovingPeakIndividual': [[0.1, 0.65], [True, False, True, False]],
    }


# =============================================================================
# Clean raw data if wanted -- didn't yield good results but don't know the optimal cleaning settings
# =============================================================================

clean_method = 'titration_thresholding_then_medfilt' # titration_thresholding // medfilt // titration_thresholding_then_medfilt // medfilt_then_titration_thresholding
medfilt_size = 3
# titration_thresholds = [20, 19, 18, 17, 16, 15, 14, 13]
# titration_thresholds = [20, 18, 16, 14, 12, 10]
titration_thresholds = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

referencing_method = 'car' # car / laplacian

# =============================================================================
# # for neural data from cross correlation
# =============================================================================
# peak_type='abs' #'abs', 'neg', 'pos' ## redundant
sum_all_width = False ## summing between the window list or below
sum_half_width = 0.2 ## find peak, get 0.2 on both sides and summing
sfreq = 30_000
box_s = 0.5
corsize_s = 0.1

# we extract features from cross correlation so that we can try add into the regression models
# extract_features_from_xcor = False

## for different filtering
# sf_downsampled_ratio = 0.05
# rolling_windows = [30] ## rolling window width

# showing plots 
plot_3d=True #3d plot that will make a lot of sense
show_plot=False # plot of bladder pressure, and two neural recordings
plot_main=True # cross correlation plot
plot_sns=True # cross correlation plot 90 degrees
plot_corr=False # showing individual correlation plot (this will overload the pc)
plot_peaks_in_ts=False # showing peak finds in time series that aligns with bladder pressure (variance of cross correlation) 


# =============================================================================
# getting the actual cross correlation data with output flipped if negative corr after xcorr
# =============================================================================
# data = filtered_files[1]

        
## we should pick something random for baseline data since we don't actually know where it should be
select_random_peak = False

   
# spine = butter_bandpass_filter(spine, 110, 5000, sfreq, 3, use_lfilter=True)
# bladder = butter_bandpass_filter(bladder, 110, 5000, sfreq, 3, use_lfilter=True)
# pressure_filt = butter_bandpass_filter(pressure, 1, 500, sfreq, 3)
           
## copying this so that we still clean it, get the flip, but use raw for uncleaed data
spine_original = spine.copy()
bladder_original = bladder.copy()

## Cleaning it for all since that allows 'flipping'
# if clean_raw_data:
# =============================================================================
#     Start cleaning raw data
# =============================================================================

spine = remove_spikes(df=None, 
              input_signal=spine, 
              clean_method=clean_method, 
              medfilt_size=medfilt_size, 
              titration_thresholds=titration_thresholds,
              show_plot=False)


bladder = remove_spikes(df=None, 
              input_signal=bladder, 
              clean_method=clean_method, 
              medfilt_size=medfilt_size, 
              titration_thresholds=titration_thresholds,
              show_plot=False)


if referencing_method == 'car':
    # common average reference
    spine, bladder = common_average_reference(spine, bladder)

elif referencing_method == 'laplacian_simple':
    # laplacian reference
    spine, bladder, _ = perform_laplacian_referencing(spine, bladder)

elif referencing_method == 'laplacian_weighted':
    # laplacian reference
    spine, bladder = calculate_laplacian_referenced_signals(spine, bladder, electrode_spacing_mm=3.25, sigma=1.0)

# elif referencing_method == 'svd':
#     spine, bladder = run_singular_value_decomposition(spine, bladder, threshold=0.1)

elif referencing_method == 'pca':
    spine, bladder = pca_denoise(spine, bladder)
                                            
# =============================================================================
#     End Cleaning
# =============================================================================


            
# =============================================================================
# different peak type
# =============================================================================
with tqdm(save_dirnames_and_peak_windows.items()) as progress_bar:
    df_peak_info = []
    for data_type, data_input in progress_bar:
        progress_bar.set_description(f"Getting peak info from {data_type}")
        
    # for data_type, data_input in tqdm(save_dirnames_and_peak_windows.items()):
    
        peak_window = data_input[0]
        non_moving_peak = data_input[1][0]
        grouped_peak = data_input[1][1]
        individual_peak = data_input[1][2]
        get_left_peak_next_to_zero_corr = data_input[1][3]
        
        # =============================================================================
        # cross correlation -- only running it for the first loop
        # =============================================================================
        # we should only extract it once because it should be the same across different data_type
        first_in_dict_items = next(iter(save_dirnames_and_peak_windows.items()))
        
        ## using first in the input data to flip or not flip the data
        ## NOTE: This WILL flip either spine or bladder, and 'flipped' will tell you which has been flipped.
        if (first_in_dict_items[0] == data_type):

            peak_window = first_in_dict_items[1][0]
            non_moving_peak = first_in_dict_items[1][1][0]
            grouped_peak = first_in_dict_items[1][1][1]
            individual_peak = first_in_dict_items[1][1][2]
            get_left_peak_next_to_zero_corr = first_in_dict_items[1][1][3]

  
            # median norm
            spine = median_normalisation(spine)
            bladder = median_normalisation(bladder)
            pressure = median_normalisation(pressure)
                        
            
            ### check plot of spine bladder differences
            # import matplotlib.pyplot as plt

            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            # ax1.plot((spine-bladder), color='red')
            # ax1.set_ylabel('Diff between neural recordings')
            # ax2.plot(pressure, color='blue')
            # ax2.set_ylabel('Bladder Pressure')         
            # plt.xlabel('Time')                
            # plt.show()

            
            # =============================================================================
            # # cross correlation to check if should flip or not flip
            # =============================================================================
            all_cor, lags = movingxcor(x=spine, y=bladder, pressure=pressure, sfreq=sfreq, box_s=box_s)
            
            ## flipping method
            # max_sum_xcor = np.sum([np.max(i) for i in all_cor]) / len(all_cor)
            # min_sum_xcor = np.sum([np.min(i) for i in all_cor]) / len(all_cor)
            
            # flipped_spine = False
            # if abs(max_sum_xcor) < abs(min_sum_xcor):
            #     flipped_spine = True
            #     spine = spine * -1
            #     spine = median_normalisation(spine)
            
            if not clean_raw_data:
                spine = spine_original.copy()
                bladder = bladder_original.copy()
                # if flipped_spine:
                #     spine = spine * -1
                spine = median_normalisation(spine)
                bladder = median_normalisation(bladder)
                
            
            # actual cross correlation using flipped signal (spine flipped)
            all_cor, lags = movingxcor(x=spine, y=bladder, pressure=pressure, sfreq=sfreq, box_s=box_s)
            
            # =============================================================================
            # Cleaning the cross correlation matrix (in time, not in lag)           
            # =============================================================================
            # if clean_xcor_matrix:
            #     def wrapper_remove_spikes_function(input_signal):
            #         input_signal = remove_spikes(df=None, 
            #                       input_signal=input_signal, 
            #                       clean_method=clean_method, 
            #                       medfilt_size=medfilt_size, 
            #                       titration_thresholds=titration_thresholds,
            #                       show_plot=False)
            #         return input_signal

            #     all_cor = np.apply_along_axis(wrapper_remove_spikes_function, axis=0, arr=np.vstack(all_cor))
            # =============================================================================

            
            ## saving cross correlation matrix as numpy
            xcor_matrix = np.array(all_cor)
            # feature_regression_dir.mkdir(parents=True, exist_ok=True)
            # np.save(feature_regression_dir.joinpath(data.stem + '_xcor_matrix.npy'), xcor_matrix, allow_pickle=True)
            
            # downsample
            samples = len(xcor_matrix)
            downsampled_spine = downsample(spine, samples)
            downsampled_bladder = downsample(bladder, samples)    
            maxtime = len(spine) / sfreq
            time = np.linspace(0, maxtime, samples)
            sf_downsampled = samples/(time[-1] - time[0])
            
            # # feature extraction paused / deprecated -- we do it separately later since this takes a long time
            # xcor_reatures = get_features_from_timeseries(xcor_matrix, fs=sf_downsampled) if extract_features_from_xcor else ()
        
        # # feature extraction paused / deprecated -- we do it separately later since this takes a long time
        # extract_only_once = True if (first_in_dict_items[0] == data_type) & extract_features_from_xcor else False
        
        ## peak output
        ## using peak_type as 'pos' since we filpped it already, and it should be positive
        output, downsampled_pressure = get_peak_output(all_cor, pressure, sfreq=sfreq, corsize_s=corsize_s, 
                                         window_distance_from_zero=peak_window, get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr, 
                                         sum_all_width=sum_all_width, sum_half_width=sum_half_width, 
                                         peak_type='pos', non_moving_peak=non_moving_peak, select_random_peak=select_random_peak, 
                                         grouped_peak=grouped_peak, individual_peak=individual_peak, 
                                         plot_peaks_in_ts=plot_peaks_in_ts, plot_corr=False)
        
        df_peak_info_temp = peak_output_into_df(output, sfreq, corsize_s, data_type)
        df_peak_info.append(df_peak_info_temp)
        
        # plotting
        ## wanted to see nonmovingpeak plots
        # plot_main_intermediate = True if data_type == 'NonMovingPeakIndividual' else plot_main
        # plot_main_intermediate = plot_main if data_type == 'NonMovingPeakIndividual' else plot_main

        plot_xcor_info(all_cor, data_type, lags, time, 
                       df_peak_info_temp[f'peak_indices-{data_type}'], 
                       df_peak_info_temp[f'peak_ms-{data_type}'], 
                       maxtime, 
                       df_peak_info_temp[f'neural_act-{data_type}'],
                       df_peak_info_temp[f'neural_area-{data_type}'],
                       downsampled_pressure,
                       window_distance_from_zero=peak_window, 
                       # vmin=-1.1, vmax=1.1,
                       plot_3d=plot_3d, plot_main=False, plot_sns=False)
        
        
        
        
        
        
        
        
        
# # Plot the simulated nerve recording
# plt.figure(figsize=(10, 4))
# plt.plot(time, downsampled_bladder, label="Nerve Recording (Bladder)")
# plt.plot(time, downsampled_spine, label="Nerve Recording (Spine)")

# # Plot the simulated bladder pressure recording
# plt.plot(time, downsampled_pressure, label="Bladder Pressure")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.title(f"Simulated Bladder Pressure Recording with Two Neural Recordings and a {lag_ms} ms Lag (Repeated Twice)")
# plt.grid(True)
# plt.show()


        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
    