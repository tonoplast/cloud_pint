# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:27:16 2023

@author: WookS
"""
## myogenic activity is close to 0 (corr value of 1-ish) so we need to avoid that


from pathlib import Path
import pandas as pd
import numpy as np
from src.preprocessing.preproc_tools import butter_bandpass_filter
from src.preprocessing.cross_correlation import movingxcor, get_peak_output, peak_output_into_df, plot_xcor_info #get_correlation #, check_rho_based_on_peak_detection_window
from src.preprocessing.preproc_tools import remove_spikes, median_normalisation, downsample, common_average_reference, perform_laplacian_referencing, calculate_laplacian_referenced_signals, pca_denoise
from tqdm import tqdm

data_drive = 'O'
base_dir = Path(fr'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')

input_dir = base_dir.joinpath('AutoCurate')
## rename here if you want to try different things
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned') ## clean raw, clean cross-correlation matrix
# processed_dir = base_dir.joinpath('Processed_raw_cleaned') ## clean raw only
# processed_dir = base_dir.joinpath('Processed_raw_uncleaned') ## no cleaning

feature_regression_dir = processed_dir.joinpath('Features').joinpath('Regression')
neural_dir = processed_dir.joinpath('Neural')

# not taking baseline, and not taking "12_013.acute.saline.0.4" BUT could potentially take in baseline too for NN to learn
parquet_files = list(input_dir.glob("*parquet"))
filtered_files = [file for file in parquet_files if not ("12_013.acute.saline.0.4" in file.stem)]



# This is not very well coded. It is just wrapped into 'run_load_and_crosscorrelate' function at the momment
# so that it's easy to just run
## TODO: best to move it into a config file of some sort to clean it up

def run_load_and_crosscorrelate(filtered_files):
    # =============================================================================
    # # saving cross correlation data folder name and peak window -- make sure that you add / delete as you want
    # =============================================================================
    ## this is for a-delta range (-0.1 to -0.65) / -0.5 / peak / left peak / 
    save_dirnames_and_peak_windows = {
        ## Data type: [[peak window], [non_moving_peak, grouped_peak, individual_peak, first_on_left_peak]]
        'StaticPeak_0.0': [[-1.0, 1.0], [False, False, False, False]],
        'StaticPeak_0.5': [[0.49, 0.51], [False, False, False, False]],
        # 'MovingPeak': [[0.1, 0.65], [False, False, False, False]],
        'MovingPeakFirstOnLeft': [[-1.0, 1.0], [False, False, False, True]], ## range is irrelevant but required -- need to capture max and find left
        'NonMovingPeak': [[0.1, 0.65], [True, True, False, False]],
        'NonMovingPeakIndividual': [[0.1, 0.65], [True, False, True, False]],
        }
    
    
    # =============================================================================
    # Clean raw data if wanted -- didn't yield good results but don't know the optimal cleaning settings
    # =============================================================================
    clean_raw_data = True
    clean_xcor_matrix = True
    clean_method = 'titration_thresholding_then_medfilt' # titration_thresholding // medfilt // titration_thresholding_then_medfilt // medfilt_then_titration_thresholding
    medfilt_size = 3
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
    
    # showing plots 
    plot_3d=False #3d plot that will make a lot of sense
    show_plot=False # plot of bladder pressure, and two neural recordings
    plot_main=False # cross correlation plot
    plot_sns=False # cross correlation plot 90 degrees
    plot_corr=False # showing individual correlation plot (this will overload the pc)
    plot_peaks_in_ts=False # showing peak finds in time series that aligns with bladder pressure (variance of cross correlation) 
        
    # =============================================================================
    # getting the actual cross correlation data with output flipped if negative corr after xcorr
    # =============================================================================
    with tqdm(filtered_files) as progress_bar:
        for data in progress_bar:
            progress_bar.set_description(f"Processing:\n{data.stem}")
            
    
            # import pdb; pdb.set_trace()
            df = pd.read_parquet(data)
            
            ## we should pick something random for baseline data since we don't actually know where it should be
            select_random_peak = True if '0.1' in data.stem else False
            
            pressure = df['Pressure'].to_numpy()
            spine = df['nSpine'].to_numpy()
            bladder = df['nBladder'].to_numpy()
            
            spine = butter_bandpass_filter(spine, 110, 5000, sfreq, 3, use_lfilter=True)
            bladder = butter_bandpass_filter(bladder, 110, 5000, sfreq, 3, use_lfilter=True)
                       
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
                spine, bladder = common_average_reference(spine, bladder, method='mean')
            
            # =============================================================================
            # These are not used / tested properly, but options are there if you want to
            # further experiment on cleaning process. check 'src.preprocessing.preproc_tool'
            # =============================================================================
            elif referencing_method == 'laplacian_simple':
                # laplacian reference
                spine, bladder, _ = perform_laplacian_referencing(spine, bladder)

            elif referencing_method == 'laplacian_weighted':
                # laplacian reference
                spine, bladder = calculate_laplacian_referenced_signals(spine, bladder, electrode_spacing_mm=3.25, sigma=1.0)
            
            elif referencing_method == 'pca':
                spine, bladder = pca_denoise(spine, bladder)
                                                        
            # =============================================================================
            #     End Cleaning
            # =============================================================================
            
            timestamp = df.index.values
            
            del df
            
            ms_to_s = 1e6
            if show_plot:
                from src.preprocessing.cross_correlation import showdata
                showdata(timestamp/ms_to_s, timestamp/ms_to_s, pressure, spine, bladder, show_diff=True)
        
                        
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
                                    
                        
                        # =============================================================================
                        # # cross correlation to check if should flip or not flip
                        # =============================================================================
                        all_cor, lags = movingxcor(x=spine, y=bladder, pressure=pressure, sfreq=sfreq, box_s=box_s)
                        
                        ## flipping method
                        max_sum_xcor = np.sum([np.max(i) for i in all_cor]) / len(all_cor)
                        min_sum_xcor = np.sum([np.min(i) for i in all_cor]) / len(all_cor)
                        
                        flipped_spine = False
                        if abs(max_sum_xcor) < abs(min_sum_xcor):
                            flipped_spine = True
                            spine = spine * -1
                            spine = median_normalisation(spine)
                        
                        if not clean_raw_data:
                            spine = spine_original.copy()
                            bladder = bladder_original.copy()
                            if flipped_spine:
                                spine = spine * -1
                            spine = median_normalisation(spine)
                            bladder = median_normalisation(bladder)
                            
                        
                        # actual cross correlation using flipped signal (spine flipped)
                        all_cor, lags = movingxcor(x=spine, y=bladder, pressure=pressure, sfreq=sfreq, box_s=box_s)
                        
                        # =============================================================================
                        # Cleaning the cross correlation matrix (in time, not in lag)           
                        # =============================================================================
                        if clean_xcor_matrix:
                            def wrapper_remove_spikes_function(input_signal):
                                input_signal = remove_spikes(df=None, 
                                              input_signal=input_signal, 
                                              clean_method=clean_method, 
                                              medfilt_size=medfilt_size, 
                                              titration_thresholds=titration_thresholds,
                                              show_plot=False)
                                return input_signal

                            all_cor = np.apply_along_axis(wrapper_remove_spikes_function, axis=0, arr=np.vstack(all_cor))
                        # =============================================================================

                        xcor_matrix = np.array(all_cor)
                        
                        # downsample
                        samples = len(xcor_matrix)
                        downsampled_spine = downsample(spine, samples)
                        downsampled_bladder = downsample(bladder, samples)    
                        maxtime = len(spine) / sfreq
                        time = np.linspace(0, maxtime, samples)
                        sf_downsampled = samples/(time[-1] - time[0])
                        
                    
                    ## peak output
                    counter_for_plot = 1

                    ## using peak_type as 'pos' since we filpped it already, and it should be positive
                    output, downsampled_pressure = get_peak_output(all_cor, pressure, sfreq=sfreq, corsize_s=corsize_s, 
                                                     window_distance_from_zero=peak_window, get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr, 
                                                     sum_all_width=sum_all_width, sum_half_width=sum_half_width, 
                                                     peak_type='pos', non_moving_peak=non_moving_peak, select_random_peak=select_random_peak, 
                                                     grouped_peak=grouped_peak, individual_peak=individual_peak, 
                                                     plot_peaks_in_ts=plot_peaks_in_ts, plot_corr=plot_corr)
                    
                    df_peak_info_temp = peak_output_into_df(output, sfreq, corsize_s, data_type)
                    df_peak_info.append(df_peak_info_temp)
                    
                    # plotting
                    ## wanted to see nonmovingpeak plots
                    plot_main_intermediate = plot_main if data_type == 'NonMovingPeakIndividual' else plot_main

                    plot_xcor_info(all_cor, data_type, lags, time, 
                                   df_peak_info_temp[f'peak_indices-{data_type}'], 
                                   df_peak_info_temp[f'peak_ms-{data_type}'], 
                                   maxtime, 
                                   df_peak_info_temp[f'neural_act-{data_type}'],
                                   df_peak_info_temp[f'neural_area-{data_type}'],
                                   downsampled_pressure,
                                   window_distance_from_zero=peak_window, 
                                   vmin=-0.002, vmax=0.002, cmap='viridis',
                                   ylim_min=-1, ylim_max=1,
                                   show_peak_latencies=True,
                                   plot_3d=plot_3d, plot_main=plot_main_intermediate, plot_sns=plot_sns)
            

            df_peak_info = pd.concat(df_peak_info, axis=1)
            
            df_xcor_matrix = pd.DataFrame(xcor_matrix, columns=np.round(lags*1000, 2))
                
            # saving downsampled data into dataframe
            df_main_info = pd.DataFrame({
                               'source': data.stem, 
                               'time': time, 
                               'bladder_pressure': downsampled_pressure,
                               'spine': downsampled_spine,
                               'bladder': downsampled_bladder,
                               'sf': sf_downsampled,
                               'flipped_spine': flipped_spine,
                               })
            
            
            
            df_info_with_xcor_matrix = pd.concat([df_main_info, df_peak_info, df_xcor_matrix], axis=1)
        
            # make dir and save
            neural_dir.mkdir(parents=True, exist_ok=True)
            df_info_with_xcor_matrix.to_parquet(neural_dir.joinpath(data.stem + '.parquet'), index=False)
  

def main():
    run_load_and_crosscorrelate(filtered_files)


if __name__ == "__main__":
    main()              


