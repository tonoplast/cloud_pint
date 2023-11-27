# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:18:11 2023

@author: WookS
"""
import numpy as np
import pandas as pd
from scipy import signal
from src.preprocessing.preproc_tools import round_decimals_up, butter_bandpass_filter, downsample, ms_to_npts, npts_to_ms, median_normalisation, mean_normalisation, interpolate_outliers
from src.utils.smoothing_methods import SmoothingMethods
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
from scipy.ndimage import gaussian_filter1d
from src.utils.feature_extractor import get_features_from_timeseries
from src.preprocessing.preproc_tools import remove_spikes
from pathlib import Path

smoother = SmoothingMethods()

def xcor(x, y, sfreq=30000, start_lag=-0.05, end_lag=0.05):
    
    cor = signal.correlate(x, y)
    
    ## I was experimenting with normalisation
    # cor = signal.correlate(x-np.mean(x), y - np.mean(y))
    # cor = signal.correlate(x-np.median(x), y - np.median(y))

    lags = signal.correlation_lags(len(x), len(y))
    lags = lags / sfreq
    # cor /= np.max(cor)

    startIndex = np.searchsorted(lags, start_lag)
    endIndex = np.searchsorted(lags, end_lag)

    cor = cor[startIndex:endIndex+1] 
    cor -= np.median(cor)
    lags = lags[startIndex:endIndex+1]

    return cor, lags

# don't worry about this, it was there for me to try and do some plotting
counter_for_plot = 1

def find_left_peak_with_area(time_series_data, sfreq, #title,
                             window_distance_from_zero=[0.1, 0.65], get_left_peak_next_to_zero_corr=False,
                             sum_all_width=False, sum_half_width=0.2, 
                             plot_corr=False, peak_type='abs', pre_defined_peak=None):
    # import pdb; pdb.set_trace()
    global counter_for_plot
    
    '''
    window_distance_from_zero: pre-determined timelag (distance?) from middle (0 lag), so we can find a peak within this window. unit: ms
    Adelta between 5 m/s and 35 m/s: lag range of 0.1 ms - 0.65 ms
    C fiber between .5 m/s and 2 m/s: lag range of 1.625 - 6.5 ms
    '''

    # invert it??
    if peak_type == 'abs':
        time_series_data_mod = abs(time_series_data)
    elif peak_type == 'neg':  
        time_series_data_mod = time_series_data*-1
    elif peak_type == 'pos':
        time_series_data_mod = time_series_data
    else:
        raise ValueError("Bad imput. It should be either 'abs', 'pos' or 'neg'!")
       
    # window_distance_from_zero=[0.1, 0.65]
    
    window_distance_from = window_distance_from_zero[1]
    window_distance_to = window_distance_from_zero[0]
    
    ## used for peak distance but perhaps not necessary
    # window_size = window_distance_from - window_distance_to
    # window_size_npts = ms_to_npts(window_size, sfreq)
    
    middle_index = len(time_series_data) // 2
    window_distance_from_zero_npts = range(middle_index - ms_to_npts(window_distance_from, sfreq), middle_index - ms_to_npts(window_distance_to, sfreq)+1)
        
    if window_distance_from != window_distance_to:
        peaks, _ = signal.find_peaks(time_series_data_mod[window_distance_from_zero_npts])
    
    
    if get_left_peak_next_to_zero_corr:
        ## first left peak after the highest (which should be middle)
        peaks_all, _ = signal.find_peaks(time_series_data_mod[window_distance_from_zero_npts])
        if peaks_all.any():
            highest_peak_index = peaks_all[np.argmax(time_series_data_mod[window_distance_from_zero_npts][peaks_all])]
            left_peaks = peaks_all[peaks_all < highest_peak_index]
            if len(left_peaks) > 0:
                peaks = [left_peaks[-1]]  # The last peak to the left
                # pre_defined_peak = peaks[0]
            else:
                peaks = []  # No left peaks found
        else:
            peaks = []

    
    # import pdb; pdb.set_trace()
    # =============================================================================
    # if more than 0, then get max or mid point -- choose random point
    # =============================================================================
    found_peak = True
    if len(peaks) > 0:
        # peaks_max = peaks[np.argmax(peaks)]
        peaks_max = peaks[np.argmax(time_series_data_mod[window_distance_from_zero_npts][peaks])] 
    else:
        ### Max may not be good -- so we choose random peak
        # peaks_max = np.argmax(time_series_data_mod[window_distance_from_zero_npts])
        # peaks_max = (window_distance_from_zero_npts[-1] - window_distance_from_zero_npts[0])//2
        np.random.seed(42)
        peak_indices_random = np.random.choice(window_distance_from_zero_npts, size=1, replace=True)[0]
        peaks_max = peak_indices_random - window_distance_from_zero_npts[0]
        found_peak = False

    # peaks_max_index = peaks_max + window_distance_from_zero_npts[0]
    peaks_max_index = pre_defined_peak if pre_defined_peak else window_distance_from_zero_npts[peaks_max]

    peak_value = time_series_data[peaks_max_index]
    midpoint_of_window = int(round((window_distance_from_zero_npts[0] + window_distance_from_zero_npts[-1]) / 2))
    
    if sum_all_width:
        area_to_be_trapzed = time_series_data[window_distance_from_zero_npts]
    else:
        ## this is there because we don't want to get the area that is before - 0.1
        minimum_distance_allowed_from_zero = middle_index - ms_to_npts(sum_half_width+0.1, sfreq)
        
        if midpoint_of_window > minimum_distance_allowed_from_zero:
            diff_to_add = round_decimals_up(npts_to_ms(midpoint_of_window-minimum_distance_allowed_from_zero, sfreq),1)
            sum_half_range = [sum_half_width+diff_to_add, sum_half_width-diff_to_add]
        else:
            sum_half_range = [sum_half_width, sum_half_width]
        
        # sum_half_range = [sum_half_width, sum_half_width]
        area_to_be_trapzed = time_series_data[range(midpoint_of_window-ms_to_npts(sum_half_range[0], sfreq), midpoint_of_window+ms_to_npts(sum_half_range[1], sfreq))]

    area_within_window = area_to_be_trapzed[0] if len(area_to_be_trapzed) == 1 else np.trapz(area_to_be_trapzed)

    # import pdb; pdb.set_trace()
    if plot_corr:
        
        num_samples = len(time_series_data)
        total_duration_ms = num_samples / sfreq * 1000
        
        # Calculate half of the total duration
        half_duration_ms = total_duration_ms / 2
        
        # Create the x-axis values centered at 0 milliseconds
        time_ms = np.linspace(-half_duration_ms, half_duration_ms, num_samples)


        plt.figure(figsize=(12, 4))
        plt.plot(time_ms, time_series_data)
        plt.plot(time_ms[peaks_max_index], time_series_data[peaks_max_index], "x", color='red')
        plt.title(f'Cross Correlation - At {time_ms[peaks_max_index]:.2} ms [{counter_for_plot/2}s]')
        plt.axvspan(-window_distance_to, -window_distance_from, color='g', alpha=0.2, lw=0)
        plt.xlim(-10, 10)
        plt.ylim(-0.05, 0.05)
        plt.show()
        

        # plot_filename = Path(rf'D:/PINT/saveplot/cc_peak_{counter_for_plot}.png')
        # plt.savefig(plot_filename)
        # plt.close()
        
        counter_for_plot += 1
        
        # plt.pause(0.1)
        # plt.waitforbuttonpress()

    return peaks_max_index, peak_value, area_within_window, found_peak


def get_correlation(input_data, downsampled_pressure, data_type='', calculation_type='rolling_mean', rolling_window=10, cutoff_frequency=1, sf_downsampled=2, auto_flip=True, show_plot=True, print_output=True):
   
    # import pdb; pdb.set_trace()
    
    # Define the dictionary to map calculation types to functions
    calculation_functions = {
        'cumsum': lambda x: smoother.cumulative_sum(x),
        'ema': lambda x: smoother.exponential_moving_average(x, alpha=0.3),
        'sgf': lambda x: smoother.savitzky_golay_filter(x, window_size=5, order=2),
        'lowess': lambda x: smoother.lowess_smoothing(x, frac=0.3),
        'moving_median': lambda x: smoother.moving_median(x, window_size=smoother.make_odd(rolling_window)),
        'rolling_mean': lambda x: smoother.rolling_mean(x, window_size=rolling_window),
        'kalman_filter': lambda x: smoother.kalman_filter(x),
        
        'stl_seasonal': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[0],
        'stl_trend': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[1],
        'stl_residual': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[2],

        'stl_envelope_seasonal': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[3],
        'stl_envelope_trend': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[4],
        'stl_envelope_residual': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[5],

        'stl_envelope_seasonal_abs': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[6],
        'stl_envelope_trend_abs': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[7],
        'stl_envelope_residual_abs': lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[8],
        
        'stl_envelope_residual_rolling': lambda x: smoother.rolling_mean(smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[5], window_size=rolling_window),
        'stl_envelope_residual_abs_rolling': lambda x: smoother.rolling_mean(smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[8], window_size=rolling_window),

    }
    
    # Check if calculation_type is valid
    if calculation_type not in calculation_functions:
        allowed_calculation_types = list(calculation_functions.keys())
        raise ValueError(f"calculation_type is not valid! It should be one of: {allowed_calculation_types}")

    downsampled_var = pd.Series(calculation_functions[calculation_type](input_data))
    
    if auto_flip:
        if abs(downsampled_var.min()) > abs(downsampled_var.max()):
            downsampled_var *= -1    

    rho, pval = stats.pearsonr(downsampled_var, downsampled_pressure)
        
    # Output and Plotting
    if print_output:
        print()
        print('DataType:', data_type)
        print("DataLength:", len(downsampled_var))
        print("pval:", pval)
        if auto_flip:
            print("Pearson:", abs(rho))
        else:
            print("Pearson:", rho)
        
    
    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(downsampled_var, "b", label="Neural activity", alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(downsampled_pressure, "r" , label="Downsampled pressure", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neural activity (au)")
        ax2.set_ylabel("Downsampled Pressure")
        fig.legend(loc="upper left")
        
        # Add Pearson correlation coefficient to the plot
        correlation_text = f"{data_type} (Pearson: {rho:.2f})"
        ax.text(0.5, 0.95, correlation_text, transform=ax.transAxes, ha='center')
        plt.show()
        
    return downsampled_var, rho


def flip_flag_using_rolling_mean(input_signal, downsampled_pressure, 
                            medfilt_size, calculation_type, data_type, 
                            rolling_window, sf_downsampled, sf_downsampled_ratio, peak_window, 
                            show_plot=True, auto_flip=False, print_output=True):
    '''
    flip signal method
    '''
    signal_medfilt = signal.medfilt(input_signal, medfilt_size)    

    downsampled_var, rho = get_correlation(signal_medfilt, downsampled_pressure, 
                                            f'{data_type} - {calculation_type} {peak_window}\nrolling_window: {rolling_window}', 
                                            calculation_type=calculation_type, 
                                            rolling_window=rolling_window,
                                            cutoff_frequency=sf_downsampled*sf_downsampled_ratio,
                                            sf_downsampled=sf_downsampled,
                                            auto_flip=auto_flip,
                                            show_plot=show_plot, 
                                            print_output=print_output)

    signal_quality_ma = rho
    flip_signal = True if rho < 0 else False
    return signal_quality_ma, flip_signal




def check_rho_based_on_peak_detection_window(rho_list, title):
    # Extract keys (y-values) and values (x-values) from the dictionaries
    y_values = [list(d.keys())[0] for d in rho_list]
    x_values = [-list(d.values())[0][1] for d in rho_list]
    
    # Plot the data
    plt.figure()
    plt.plot(x_values, y_values, 'bo-')  # 'bo-' specifies blue color, round markers, and lines
    plt.xlabel('lag distance from 0 (ms)')
    plt.ylabel('rho')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
    
def movingxcor(x, y, pressure, sfreq=30000, box_s=0.5):
    '''
    box_s in s: This box is the amount of data to be cross-correlated
    
    Arbitrary value but bigger the value, larger the peak since it has more data, Otherwise, it will be too small to be detected
    It is possible that, if we overlap, James' cumulative sum may not work too well?
    
    corsize_s in s: cross correlation size (e.g. -1500 | 0 | +1500 if 0.1 sec (30,000 * 0.1))
    
    sum_all_width: integration of all points within the time window
    sum_half_width: find peak, go left 0.2 go right 0.2 (that is the window for integral)
    peak_type: 'abs' to get highest peak
    non_moving_peak: not very intuitive name. instead of 'moving' all the time, it finds a peak and applies that to all data
   
    Part of non_moving_peaks --
        select_random_peak: this is mostly for baseline (0.1) data, because when we get non_moving_peak, it just needs to find some random peaks
        grouped_peak: instead of finding peak for individual peaks, it finds overall (mean) and applies that to the subject/data
        individual_peak: this is finding each peak and applies the peak throughout
    
    Note that non_moving_peak is not viable in real-time data
    
    '''
    
    def compute_crosscorrelation(data):
        start, end, x, y = data
        cor, lags = xcor(x[start:end], y[start:end])
        return cor, lags
    
    # import pdb; pdb.set_trace()
    
    box = int(sfreq*box_s)  # 30000 = 1 sec -- boxsize is therefor 0.5*30000 = 15000
    
    start_range = box // 2
    step_range = box
    limit_range = len(x)
    stop_range = start_range + ((limit_range - start_range) // step_range) * step_range
    data_list = [(i - (box // 2), i + (box // 2), x, y) for i in range(start_range, stop_range, step_range)]

    all_cor_lags = list(map(compute_crosscorrelation, data_list))    
    all_cor = [tup[0] for tup in all_cor_lags]
    lags = [tup[1] for tup in all_cor_lags][0]
    
    return all_cor, lags
    
    

def get_peak_output(all_cor, pressure, sfreq=30000, corsize_s=0.1, 
                    window_distance_from_zero=[0.1, 0.65], get_left_peak_next_to_zero_corr=False,
                    sum_all_width=False, sum_half_width=0.2, peak_type='abs', 
                    non_moving_peak=False, select_random_peak=False, grouped_peak=False, individual_peak=False, 
                    plot_peaks_in_ts=True, plot_corr=False):
    
    def get_xcor_output(all_cor, pressure, window_distance_from_zero, get_left_peak_next_to_zero_corr, sum_all_width, sum_half_width, peak_type, plot_corr=plot_corr):

        output = list(map(lambda cor: find_left_peak_with_area(cor, 
                                                        sfreq, 
                                                        # title='hi', 
                                                        window_distance_from_zero=window_distance_from_zero,
                                                        get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                                                        sum_all_width=sum_all_width,
                                                        sum_half_width=sum_half_width,
                                                        plot_corr=plot_corr,
                                                        peak_type=peak_type,
                                                        pre_defined_peak=None), all_cor))
            
        downsampled_pressure = downsample(median_normalisation(pressure.copy()), len(output))
        peak_indices = np.array([tup[0] for tup in output])
        
        
        # import pdb; pdb.set_trace()
        
        
        if non_moving_peak:
            
            if select_random_peak:
                # =============================================================================
                # This is to get random peak within the defined range (a delta -0.65 to -0.1)
                # Seems to make sense for 'baseline' condition, although this is biased to make sure we don't get highest activity
                # =============================================================================
                corsize = int(sfreq*corsize_s)  # 30000 = 1 sec -- corsize is therefore 0.1*30000 = 3000

                random_npts_range = np.arange(corsize//2 - ms_to_npts(window_distance_from_zero[1], sfreq)+1, corsize//2 - ms_to_npts(window_distance_from_zero[0], sfreq))
                number_of_random_indices = len(peak_indices)
                np.random.seed(42)
                # Randomly select n samples from the array
                peak_indices_random = np.random.choice(random_npts_range, size=number_of_random_indices, replace=True)
                        
                output = list(map(lambda cor, peak_idx: find_left_peak_with_area(
                    cor, 
                    sfreq, 
                    window_distance_from_zero=window_distance_from_zero,
                    get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                    sum_all_width=sum_all_width,
                    sum_half_width=sum_half_width,
                    plot_corr=plot_corr,
                    peak_type=peak_type,
                    pre_defined_peak=peak_idx
                    ), 
                    all_cor, peak_indices_random))
    
            
            else:
                if grouped_peak:
                    # =============================================================================
                    #     ## hopefully get the mean peak across subject/file
                    # =============================================================================
                    meedian_sig_filtered = signal.medfilt(np.median(np.stack(all_cor), 0))
                    peak_indices_mean = find_left_peak_with_area(meedian_sig_filtered, 
                                                                sfreq, 
                                                                window_distance_from_zero=window_distance_from_zero,
                                                                get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                                                                sum_all_width=sum_all_width,
                                                                sum_half_width=sum_half_width,
                                                                plot_corr=False,
                                                                peak_type=peak_type)[0]
                    
                    output = list(map(lambda cor: find_left_peak_with_area(cor, 
                                                                    sfreq, 
                                                                    # title='hi', 
                                                                    window_distance_from_zero=window_distance_from_zero,
                                                                    get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                                                                    sum_all_width=sum_all_width,
                                                                    sum_half_width=sum_half_width,
                                                                    plot_corr=plot_corr,
                                                                    peak_type=peak_type,
                                                                    pre_defined_peak=peak_indices_mean), all_cor))
                
                # =============================================================================
                # hacky way to get peak correlation for each void? But isn't this the whole point?
                # =============================================================================
                elif individual_peak:
                    
                    def find_peaks_in_time_series(all_cor, downsampled_pressure, gaus_filt_sigma=10, show_plot=True):
                        
                        def get_threshold_and_peaks(var, std_threshold_multiplier=0.5, max_threshold_multiplier=0.1, prominence_multiplier=3, peak_distance=50):
                            var_std = np.std(var)
                            threshold_1 = var_std * std_threshold_multiplier
                            threshold_2 = np.max(var) * max_threshold_multiplier
                            threshold_var = threshold_1 if threshold_1>threshold_2 else threshold_2
                            peaks, _ = signal.find_peaks(var, distance=peak_distance, height=threshold_var, prominence=np.mean(var)*prominence_multiplier) 
                            return peaks

                        
                        ## gaus_filt_sigma somewhat arbitarily chosen
                        var_cor = np.var(np.stack(all_cor), 1)   
                        var_cor = median_normalisation(var_cor)

                        
                        ## SUCH A HACKY WAY!! but this seems to work.
                        thresholds = [20, 19, 18, 17, 16, 15, 14, 13, 12]

                        for threshold in thresholds:
                            var_cor = interpolate_outliers(pd.Series(var_cor), threshold=threshold, method='pchip', show_plot=False)[0].values

                        var_cor = signal.medfilt(var_cor, 7)
                        var_cor_gaus = gaussian_filter1d(var_cor, gaus_filt_sigma)

                        peaks1 = get_threshold_and_peaks(var=var_cor, std_threshold_multiplier=0.5, max_threshold_multiplier=0.1, prominence_multiplier=3, peak_distance=50)
                        peaks2 = get_threshold_and_peaks(var=var_cor_gaus, std_threshold_multiplier=0.5, max_threshold_multiplier=0.1, prominence_multiplier=3, peak_distance=50)

                        # getting similar peaks between regular peak and gaussian peak (regular peak as base)
                        tolerance = 10
                        peaks = np.array([value for value in peaks1 if any(np.abs(peaks2 - value) <= tolerance)])

                        if show_plot:
                            x_axis = np.arange(0, len(var_cor))
                            
                            min_val_gaus = np.min(var_cor)
                            max_val_gaus = np.max(var_cor)
                            
                            # Normalize downsampled_bladder to the range of var_cor_gaus
                            downsampled_pressure_normalised = (downsampled_pressure - np.min(downsampled_pressure)) / (np.max(downsampled_pressure) - np.min(downsampled_pressure)) * (max_val_gaus - min_val_gaus) + min_val_gaus

                            plt.figure(figsize=(16,6))
                            plt.plot(x_axis, downsampled_pressure_normalised, label='bladder pressure')
                            plt.plot(x_axis, var_cor, label='variance')
                            plt.plot(x_axis, var_cor_gaus, label='variance (Guassian filtered)')
                            plt.plot(x_axis[peaks], var_cor[peaks], "x", color='red', label='peak found')
                            plt.legend()

                        return peaks
                    
                    
                    def create_subarrays(cor_peaks, all_cor):
                        '''
                        This creates the range of indexes separated by the peaks found in the variance of correlations in time.
                        It is separated by the number of peaks in a sense, with +10 number of points (arbitrary) after the peak,
                        so that we still capture some left overs.
                        This is used to hopefully make NoneMovingPeak more focussed on individual void rather than all within file/animal.
                        Technically it should be the same, but it may not be.
                        '''
                        subarrays = []
                        start = 0
                        last_end = len(all_cor)
                    
                        for end in cor_peaks:
                            subarray_end = end + 10 if end != cor_peaks[-1] else last_end + 1
                            subarrays.append(list(range(start, subarray_end)))
                            start = end + 10
                    
                        return subarrays
                    
                    # import pdb; pdb.set_trace()
                    cor_peaks = find_peaks_in_time_series(all_cor, downsampled_pressure, gaus_filt_sigma=10, show_plot=plot_peaks_in_ts)
                    cor_peaks_adjusted = cor_peaks - 0 ## this is kind of arbitary and can be explored.
                    separated_indices_based_on_cor_peaks = create_subarrays(cor_peaks_adjusted, all_cor)
                    # cor_peak_indices = peak_indices[cor_peaks_adjusted]
                    
                    ## applying new method where we get the median of all the peaks, instead of picking the 'peak' of the highest point
                    cor_peak_indices = []
                    for i in separated_indices_based_on_cor_peaks:
                        cor_peak_indices.append(np.int(np.median(peak_indices[i[0]:i[-1]+1])))
                    
                    output = []
                    for cor_peak_index, sep_indices in zip(cor_peak_indices, separated_indices_based_on_cor_peaks):
                        temp_cor = all_cor[sep_indices[0]:sep_indices[-1]+1]
                        
                        temp_output = list(map(lambda cor: find_left_peak_with_area(cor, 
                                                                            sfreq, 
                                                                            # title='hi', 
                                                                            window_distance_from_zero=window_distance_from_zero,
                                                                            get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                                                                            sum_all_width=sum_all_width,
                                                                            sum_half_width=sum_half_width,
                                                                            plot_corr=plot_corr,
                                                                            peak_type=peak_type,
                                                                            pre_defined_peak=cor_peak_index), temp_cor))
                        
                        output.append(temp_output)
            
                    output = [item for sublist in output for item in sublist]
        
        return output, downsampled_pressure
    
    output, downsampled_pressure = get_xcor_output(all_cor, pressure, window_distance_from_zero, get_left_peak_next_to_zero_corr, sum_all_width, sum_half_width, peak_type, plot_corr)
    
    return output, downsampled_pressure
    


def peak_output_into_df(output, sfreq, corsize_s, data_type):
    ## peak information
    peak_indices = np.array([tup[0] for tup in output])
    corsize = int(sfreq*corsize_s) 
    peak_ms = npts_to_ms(peak_indices - corsize//2, sfreq)
    ## it becomes strange when med normed
    # neural_act = median_normalisation(np.array([tup[1] for tup in output]))
    # neural_area = median_normalisation(np.array([tup[2] for tup in output]))
    neural_act = np.array([tup[1] for tup in output])
    neural_area = np.array([tup[2] for tup in output])
    peak_found = np.array([tup[3] for tup in output])
    
    df_peak_info_temp = pd.DataFrame({f'peak_found-{data_type}': peak_found,
                                    f'peak_indices-{data_type}': peak_indices,
                                   f'peak_ms-{data_type}': peak_ms,
                                  f'neural_act-{data_type}': neural_act,
                                  f'neural_area-{data_type}': neural_area})
    return df_peak_info_temp
    





def flip_scoring(x, y, pressure, data_type, rolling_window, sf_downsampled_ratio, peak_window, get_left_peak_next_to_zero_corr, 
                  sum_all_width, sum_half_width, peak_type, 
                  non_moving_peak, select_random_peak, grouped_peak, individual_peak, sfreq=30000, box_s=0.5):
    
    def get_signal_corr_and_flip_flag(x, y, pressure, sfreq, box_s):
    
        all_cor, lags = movingxcor(x, y, pressure, sfreq, box_s)
        
        result = np.array(all_cor)
        samples = len(result)
        maxtime = len(x) / sfreq
        time = np.linspace(0, maxtime, samples)
        sf_downsampled = samples/(time[-1] - time[0]) 
        
        output, downsampled_pressure = get_peak_output(all_cor, pressure, sfreq, corsize_s=0.1, 
                                  window_distance_from_zero=peak_window, get_left_peak_next_to_zero_corr=get_left_peak_next_to_zero_corr,
                                  sum_all_width=sum_all_width, sum_half_width=sum_half_width, 
                                  peak_type=peak_type, non_moving_peak=non_moving_peak, select_random_peak=select_random_peak, 
                                  grouped_peak=grouped_peak, individual_peak=individual_peak, 
                                  plot_peaks_in_ts=False, plot_corr=False)
        
        # peak_indices = np.array([tup[0] for tup in output])
        neural_act = np.array([tup[1] for tup in output])
        neural_area = np.array([tup[2] for tup in output])
        
        signal_quality_ma_file_act, flip_signal_file_act = flip_flag_using_rolling_mean(input_signal = neural_act, 
                                                                                            downsampled_pressure=downsampled_pressure,
                                                                                            medfilt_size = 9, 
                                                                                            calculation_type = 'rolling_mean', # 'moving_median // rolling_mean
                                                                                            data_type = data_type,
                                                                                            rolling_window = rolling_window, 
                                                                                            sf_downsampled=sf_downsampled, 
                                                                                            sf_downsampled_ratio=sf_downsampled_ratio,
                                                                                            peak_window=peak_window,
                                                                                            show_plot=False,
                                                                                            auto_flip=False,
                                                                                            print_output=False)
                                                                
        signal_quality_ma_file_area, flip_signal_file_area =  flip_flag_using_rolling_mean(input_signal = neural_area, 
                                                                        downsampled_pressure=downsampled_pressure,
                                                                        medfilt_size = 9, 
                                                                        calculation_type = 'rolling_mean', # 'moving_median // rolling_mean
                                                                        data_type = data_type,
                                                                        rolling_window = rolling_window, 
                                                                        sf_downsampled=sf_downsampled, 
                                                                        sf_downsampled_ratio=sf_downsampled_ratio,
                                                                        peak_window=peak_window,
                                                                        show_plot=False,
                                                                        auto_flip=False,
                                                                        print_output=False)
        
        return signal_quality_ma_file_act, flip_signal_file_act, signal_quality_ma_file_area, flip_signal_file_area
    
    
    # =============================================================================
    # normal signal
    # =============================================================================
    signal_quality_ma_file_act_1, \
        flip_signal_file_act_1, \
            signal_quality_ma_file_area_1, \
                flip_signal_file_area_1 = get_signal_corr_and_flip_flag(x, y, pressure, sfreq, box_s)


    # =============================================================================
    # switch signal
    # =============================================================================
    signal_quality_ma_file_act_2, \
        flip_signal_file_act_2, \
            signal_quality_ma_file_area_2, \
                flip_signal_file_area_2 = get_signal_corr_and_flip_flag(y, x, pressure, sfreq, box_s)
    
    
    score_1 = (signal_quality_ma_file_act_1 + signal_quality_ma_file_area_1) / 2
    score_2 = (signal_quality_ma_file_act_2 + signal_quality_ma_file_area_2) / 2
    total_score = (score_1 + score_2) / 2
        
    
    # import pdb; pdb.set_trace()
    ## if total score is negative then we flip one of them, depending on which had the lowest score
    if total_score < 0:
        flip_signal = True
        
        if abs(score_1) > abs(score_2):
            x = x
            y = y * -1
            flip_score = score_1
            flipped = 'bladder'
        elif (abs(score_1) < abs(score_2)) | (abs(score_1) == abs(score_2)):
            x = x*-1
            y = y
            flip_score = score_2
            flipped = 'spine'   
            
    else:
        flip_signal = False
        flip_score = score_1
        flipped = 'none'
    
    return x, y, flip_signal, flip_score, flipped
    

    
def plot_xcor_info(all_cor, data_type, lags, time, peak_indices, peak_ms, maxtime, 
                   downsampled_neural, downsampled_neural_area, downsampled_pressure,
                   window_distance_from_zero, 
                   vmin=-0.05, vmax=0.05, cmap='viridis',
                   ylim_min=-2, ylim_max=2,
                   show_peak_latencies=True,
                   plot_3d=False, plot_main=False, plot_sns=False):
    
    y_point = np.mean(peak_ms)
    # import pdb; pdb.set_trace()
    if plot_3d:
        # =============================================================================
        # 3D plot
        # =============================================================================       
        # Create X and Y values based on your data shape
        result = np.array(all_cor)
        y_axis = np.arange(result.shape[0])
        x_lag = lags.copy()*1000
        y_time = time.copy()
        X, Y = np.meshgrid(x_lag, y_time)

        # Create a 3D figure
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the 3D surface
        surf = ax.plot_surface(X, Y, result, cmap=cmap, rstride=50, cstride=50, alpha=0.4)

        # ax.set_zlim3d(-.1,.1)

        
        # Create corresponding Y and Z values for the line
        z_line_values = result[y_axis, peak_indices]  # Z values from your data
        
        # Create a line to represent the peak in cross correlation
        ax.scatter(x_lag[peak_indices], y_time, z_line_values, color='red', label='Cross Correlation at Chosen Peak', alpha=0.3, s=5)
       
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Time')
        ax.set_zlabel('cross correlation (a.u)')
        ax.set_title(f'3D cross correlation martix: {data_type}\n(mean peak: {y_point:.2} ms)')
        
        # Add a color bar to indicate values
        fig.colorbar(surf)
        plt.legend()
        plt.show()
        
    

    # import pdb; pdb.set_trace()
    if plot_main:
        result = np.array(all_cor)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [5, 1, 1, 1, 1]}, sharex=True)
        ax1.imshow(np.rot90(result), extent=[0, maxtime, np.round(np.min(lags)*1000), np.round(np.max(lags)*1000)], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Add a colorbar to the ax1 subplot     
        if show_peak_latencies:
            x_values = np.linspace(0, maxtime, num=result.shape[0])
            ax1.plot(x_values, np.array(peak_ms), color='cyan', linestyle='--', linewidth=1, label=f'Y = {y_point}')

        ax1.set_ylim(ylim_min, ylim_max)

        
        ax2.plot(time, np.array(downsampled_neural))
        ax3.plot(time, np.array(downsampled_neural_area))
        ax4.plot(time, np.array(downsampled_pressure))
        ax5.plot(time, np.array(peak_ms))
        ax5.axhline(y=y_point, color='red', linestyle='--', linewidth=0.5, label=f'Y = {y_point}')

        ax1.set_ylabel("lags (ms)")
        ax5.set_xlabel("Time (s)")
        
        ax1.title.set_text('cross-correlation')
        ax2.title.set_text('neural (peak)')
        ax3.title.set_text('neural (area)')
        ax4.title.set_text('pressure')
        ax5.title.set_text(f'peak detected (mean: {round(y_point,4)} ms)')
        ax5.set_ylim([-i for i in window_distance_from_zero])
        ax5.set_yticks([-i for i in window_distance_from_zero])
        ax5.invert_yaxis()

        plt.subplots_adjust(hspace=0.05)
        plt.tight_layout()

    if plot_sns:
        import seaborn as sns
       
        result = np.array(all_cor)
        # Create a 1x2 grid of subplots, where the first subplot is for the heatmap and the second subplot is for the line plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [5, 1]}, sharey=True)
        
        x_time = pd.Series(time).map('{:.2f}'.format)
        y_lags = pd.Series(lags*1000).map('{:.2f}'.format)
        df_result = pd.DataFrame(result, index = x_time, columns = y_lags)
        # Plot the heatmap on the first subplot
        heatmap = sns.heatmap(df_result, cmap=cmap, ax=ax1, vmin=-0.01, vmax=0.01)
        ax1.set_xlabel("lags (ms)")
        ax1.set_ylabel("Time (s)")
    
        ax2.plot(downsampled_pressure, range(0, result.shape[0]))
        ax2.set_xlabel("Bladder pressure")
        plt.subplots_adjust(wspace=0.01)  # You can adjust the value as needed to set the spacing between the subplots
    
        plt.tight_layout()
        


def showdata(t_pressure,t_neural,Pressure,N1,N2,show_diff=True):
    plt.rcParams["figure.figsize"]=10,8

    if show_diff:
        fig, axs = plt.subplots(4, 1)
    else:
        fig, axs = plt.subplot(3, 1)

    axs[0].plot(t_pressure,Pressure, color='red')
    axs[0].set(ylabel='Pressure\n(mmHg)', xlabel='Time (s)')
    axs[0].set_title('Bladder pressure')
    
    axs[1].plot(t_neural,N1)
    axs[1].set(ylabel='Neural\nactivity (V)', xlabel='Time (s)')
    axs[1].set_title('Neural activity spine')
    
    axs[2].plot(t_neural,N2)
    axs[2].set(ylabel='Neural\nactivity (V)', xlabel='Time (s)')
    axs[2].set_title('Neural activity bladder')

    
    axs[3].plot(t_neural, (N1-N2))
    axs[3].set(ylabel='Subtracted\nNeural nactivity (V)', xlabel='Time (s)')
    axs[3].set_title('Spine - Bladder')
    
    plt.tight_layout()
    
    
    