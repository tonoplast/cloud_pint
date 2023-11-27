# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:37:34 2023

@author: sungw
"""
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
from src.preprocessing.curve_fit_clustering import simulate_x_and_y_using_peaks_and_troughs, check_corr_between_scaled_and_simulated
from src.preprocessing.curve_fit_clustering import min_max_scale_data_by_group
import matplotlib.pyplot as plt
from kneed import KneeLocator
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import concurrent.futures
import time
from pathlib import Path


def split_dataframe_by_percentage(df, first_split_df_percentage):
    """
    Split a DataFrame into two parts based on a given percentage.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        percentage (float): The percentage of rows to include in the first part.

    Returns:
        (pd.DataFrame, pd.DataFrame): Two DataFrames representing the first and last parts.
    """
    if first_split_df_percentage < 0 or first_split_df_percentage > 100:
        raise ValueError("Percentage must be between 0 and 100.")
    
    # Calculate the number of rows for the first part
    num_rows_first_part = int(len(df) * (first_split_df_percentage / 100))
    
    # Split the DataFrame into two parts
    first_part = df.head(num_rows_first_part)
    last_part = df.tail(len(df) - num_rows_first_part)
    
    return first_part, last_part
     
   

def interpolate_and_smooth(x1, x2, x3, y1, y2, y3, x2_min, x2_max, interpolation_type=''):
    """
    Interpolate and smooth a set of data points between x1, x2, and x3.
    
    Parameters:
        x1 (float): The x-coordinate of the first data point.
        x2 (float): The x-coordinate of the second data point.
        x3 (float): The x-coordinate of the third data point.
        y1 (float): The y-coordinate of the first data point.
        y2 (float): The y-coordinate of the second data point.
        y3 (float): The y-coordinate of the third data point.
        x2_min (float): The minimum x-coordinate for the interpolation range.
        x2_max (float): The maximum x-coordinate for the interpolation range.
        interpolation_type (str): The type of interpolation to apply ('cubic', 'pchip', 'akima').
    
    Returns:
        tuple: A tuple containing:
            - x_values (array): The x-coordinates of the interpolated and smoothed data.
            - y_values (array): The y-coordinates of the interpolated and smoothed data.
            - y_smooth (array or None): The y-coordinates of the smoothed data using the specified interpolation
              type, or None if no interpolation is specified.
            - y_linear_smooth (array or None): The y-coordinates of the linear then smoothed data, using the specified
              interpolation type, or None if no interpolation is specified.
            - y_smooth_linear (array or None): The y-coordinates of the smoothed then linear data using the specified
              interpolation type, or None if no interpolation is specified.
    """
    # Interpolate points between x1 and x2
    x_values_x1_x2 = np.linspace(x1, x2, x2 - x1 + 1)
    y_values_x1_x2 = np.interp(x_values_x1_x2, [x1, x2], [y1, y2])
    
    # Interpolate points between x2 and x3
    x_values_x2_x3 = np.linspace(x2, x3, x3 - x2 + 1)
    y_values_x2_x3 = np.interp(x_values_x2_x3, [x2, x3], [y2, y3])
    
    # Combine the interpolated points
    x_values = np.concatenate([x_values_x1_x2, x_values_x2_x3[1:]])
    y_values = np.concatenate([y_values_x1_x2, y_values_x2_x3[1:]])

    # Create the specified interpolation
    x_coord = [x2_min, x2, x2_max]
    y_coord = [y1, y2, y3]
    
    if (x2_min != x2 and x2 != x2_max) & (interpolation_type != ''):
        if interpolation_type == 'cubic':
            interpolation = CubicSpline(x_coord, y_coord)
        elif interpolation_type == 'pchip':
            interpolation = PchipInterpolator(x_coord, y_coord)
        elif interpolation_type == 'akima':
            interpolation = Akima1DInterpolator(x_coord, y_coord)
        
        y_smooth = interpolation(x_values)
        y_smooth_x1_x2 = interpolation(x_values_x1_x2)
        y_smooth_x2_x3 = interpolation(x_values_x2_x3)
        y_linear_smooth = np.concatenate([y_values_x1_x2, y_smooth_x2_x3[1:]])
        y_smooth_linear = np.concatenate([y_smooth_x1_x2, y_values_x2_x3[1:]])
        
        return x_values, y_values, y_smooth, y_linear_smooth, y_smooth_linear
    else:
        return x_values, y_values, None, None, None
       
         
def find_minimum_in_non_overlapping_window(x, y, window_size, show_plot=True):
    
    """
    Find the minimum values in non-overlapping windows of a time series and create a dot plot.
    
    Parameters:
        x (array-like): The time values.
        y (array-like): The corresponding values.
        window_size (int): The size of non-overlapping windows.
        show_plot (bool, optional): Whether to show the dot plot (default is True).
    
    Returns:
        tuple: A tuple containing:
            - min_indices (list): The indices of the minimum values in each non-overlapping window.
            - min_values (list): The minimum values in each non-overlapping window.
    
    The function calculates the minimum value within each non-overlapping window of the specified size and returns
    the indices and values of these minima. Optionally, it can also create a dot plot showing the minimum values
    in the time series.
    
    Example usage:
    min_indices, min_values = find_minimum_in_non_overlapping_window(x, y, window_size=10)
    """

    # import pdb; pdb.set_trace()
    # Calculate the number of non-overlapping windows
    num_windows = len(y) // window_size

    # Initialize lists to store minimum values and corresponding indices
    min_values = []
    min_indices = []

    # Include the first and last index and value
    min_values.append(y[0])
    min_indices.append(0)

    # Iterate through the non-overlapping windows
    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size

        window = y[window_start:window_end]
        min_value = np.min(window)
        min_index = window_start + np.argmin(window)

        # Avoid adding duplicates
        if min_index != min_indices[-1]:
            min_values.append(min_value)
            min_indices.append(min_index)

    # Include the last index and value
    min_values.append(y[-1])
    min_indices.append(len(y) - 1)
    
    if show_plot:
        # Create a dot plot showing the minimum values
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, label='Original Data', linewidth=1)
        plt.scatter(x[min_indices], min_values, color='red', marker='o', label='Minimum Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Dot Plot of Minimum Values in Non-Overlapping Windows')
        plt.grid(True)
        plt.show()
    return min_indices, min_values


def inflection_point_search(df: pd.DataFrame, 
                            target_signal_cleaned,
                            target_signal_cleaned_rescaled,
                            use_min_from_non_overlapping_window_func: bool = True,
                            desired_no_of_min_points: int = 10,
                            apply_gaussian_filter: bool = False,
                            gaus_sigma: int = 5, 
                            first_split_df_percentage: float = 0,
                            interpolation_types: list[str] = [''],
                            exponent_factors: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                            mean_corr_methods: bool = False,
                            search_step_size: int = 1,
                            ma_window: int = 10, 
                            tolerance: float = 0.2, 
                            show_plot: bool = True) -> list[pd.DataFrame, pd.DataFrame]:
    '''
    Search for inflection points in a dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    
    apply_gaussian_filter : bool, optional (default=False)
        Apply Gaussian filter to smooth the data.
        It doesn't seem necessary for the data tested, but I suspect if it's very noisy, then it could be applied.
        Make sure to have gaus_signma value.
    
    gaus_sigma : int, optional (default=5)
        Standard deviation of the Gaussian filter if `apply_gaussian_filter` is True.
        The higher the smoother.
    
    first_split_df_percentage : float, optional (default=0)
        Percentage of the initial data to exclude when splitting for correlation ONLY.
    
    interpolation_types : list of str, optional (default=[''])
        List of interpolation methods to apply for data smoothing.
        Supported methods: ['cubic', 'pchip', 'akima'].
        Empty list or None will skip interpolation.
    
    exponent_factors : list of int, optional (default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        List of exponent factors for exponential curve.
        This must not be empty, so choose at least one in a list.
    
    mean_corr_methods : bool, optional (default=False)
        Apply mean correction methods to the data. If False, will just use Pearsons
    
    search_step_size : int, optional (default=1)
        Step size for the inflection point search algorithm (x-axis).
    
    ma_window : int, optional (default=10)
        Window size for the moving average calculation to detect inflection points.
        This is really to prevent sudden spike to 'finish' the grid search, so by smoothing out, hopefully it won't just exit out of search loop.
    
    tolerance : float, optional (default=0.2)
        Tolerance level for exiting search loop.
        Each time, optimal correlation will be retained and tested vs moving average rho value, and if it exceeds 0.2, it will stop.
        This is implemented to 1) stop early rather than trying all data points and 2) in case some weird end place that might give better corr value.
    
    show_plot : bool, optional (default=True)
        Display a plot showing the detected inflection points.
    
    Returns:
    --------
    list of dataframes
        A list of dataframe, first containing all necessary data, and another that has inflection information
    
    '''
    df_merge_later = pd.DataFrame()
    df = df.sort_values(by=['source', 'time'])
    cols_to_be_used = [target_signal_cleaned, 'peaks', 'troughs', 'sf', 'source', 'time']
    df_merge_later = df.copy()[df.columns[~df.columns.isin(cols_to_be_used)]].rename(columns={'y_simulated': 'y_simulated_all',
                                                                                              'y_simulated_exponential': 'y_simulated_exponential_all'})
    df = df.copy()[cols_to_be_used]
    
    def get_optimal_line_fit_by_group(group_df, target_signal_cleaned_rescaled, first_split_df_percentage, interpolation_types, exponent_factors, mean_corr_methods, search_step_size, ma_window, tolerance):
        # import pdb; pdb.set_trace()
        
        def calculate_optimal_inflection_point(x1, x2, x3, y1, y3, 
                                                df_first_x_percent, 
                                                df_input, 
                                                target_signal_cleaned_rescaled, 
                                                peaks_group_bool, 
                                                troughs_group_bool, 
                                                search_step_size, 
                                                ma_window=ma_window, 
                                                tolerance=tolerance):
            
            exit_candidate_x2_values_loop = False

            # import pdb; pdb.set_trace()
            # Initialize variables for the optimal x2 and correlation
            optimal_x2 = x2
            optimal_corr = -1  # Initialize with a negative value
            # y2 = df_input[target_signal_cleaned_rescaled][optimal_x2]
            
            # Determine the minimum and maximum bounds for x2
            x2_min = df_input[troughs_group_bool].index.min()
            x2_max = df_input[troughs_group_bool].index.max()
            
            # Generate the first sequence from 313 to 98
            values_from_x2_to_x2_min = [x for x in range(x2 - 1, x2_min - 1, -1)]
            
            # Generate the second sequence from 313 to 416
            values_from_x2_to_x2_max = [x for x in range(x2 + 1, x2_max + 1)]
            
            # Determine the minimum length of the two sequences
            min_length = min(len(values_from_x2_to_x2_min), len(values_from_x2_to_x2_max))
            
            # Combine the two lists so that values alternate, and attach remaining values if any
            combined_values = [val for pair in zip(values_from_x2_to_x2_min, values_from_x2_to_x2_max) for val in pair]
            combined_values += values_from_x2_to_x2_min[min_length:] + values_from_x2_to_x2_max[min_length:]
                    
            # Define a list of candidate x2 values
            candidate_x2_values = np.concatenate(([x2], combined_values))
            
            
            loop_count = 0
            # for moving average thresholding for exiting the grid search
            correlation_history = [] 
            exit_interpolation_types_loop = False
            # for x2 in tqdm(candidate_x2_values):
            for x2 in candidate_x2_values:
                # import pdb; pdb.set_trace()
                y2 = df_input[target_signal_cleaned_rescaled][x2]
                
                # Check if x2 is within valid index bounds
                if x2 < x2_min or x2 > x2_max:
                    continue
                
                # interpolation_types = ['cubic', 'pchip', 'akima']
                
                exit_yval_loop = False    
                for interpolation_type in interpolation_types:
                    x_values, y_linear, y_smooth, y_linear_smooth, y_smooth_linear = interpolate_and_smooth(x1, x2, x3, y1, y2, y3, x2_min, x2_max, interpolation_type)               
                    all_y_values = [y_linear, y_smooth, y_linear_smooth, y_smooth_linear]
                                    
                    # import pdb; pdb.set_trace()
                    for i, y_values in enumerate(all_y_values):
                        
                        if y_values is not None:
                            two_lines_y = np.append(y_values, df_input[target_signal_cleaned_rescaled][peaks_group_bool].values)
                            df_input['y_two_lines'] = two_lines_y
                            
                            df_final = pd.concat([df_first_x_percent, df_input])
                            df_final['y_two_lines'] = np.where(df_final['y_two_lines'].isna(), df_final[target_signal_cleaned_rescaled], df_final['y_two_lines'])
                            pearson, spearman, kendall, _, _ = check_corr_between_scaled_and_simulated(df_final, target_signal_cleaned_rescaled,
                                                                                            simulated_signal='y_two_lines',
                                                                                            print_corr=False, show_plot=False)
                            
                            # plot_filename = Path(rf"D:/PINT/saveplot/find_inflection_{group_df['trough_to_trough']}_{time.time()}.png")
                            # plt.savefig(plot_filename)
                            # plt.close()
                            
                            if mean_corr_methods:
                                corr_included = np.mean([pearson, spearman, kendall])
                            else:
                                corr_included = pearson
                            
                            # Update the correlation history with the latest value
                            correlation_history.append(corr_included)
                            
                            # Ensure that the correlation history does not exceed the window size
                            if len(correlation_history) > ma_window:
                                correlation_history.pop(0)
                                
                            # Calculate the moving average of correlations
                            moving_average = sum(correlation_history) / len(correlation_history)
                                           
                            if corr_included > optimal_corr:
                                optimal_x2 = x2
                                optimal_corr = corr_included
                                optimal_y_two_lines = df_final['y_two_lines'].values
                                optimal_y_value = i # 0: y_linear, 1: y_smooth, 2: y_linear_smooth, 3: y_smooth_linear
                                optimal_interpolation_type = 'linear' if optimal_y_value == 0 else interpolation_type
                            
                            loop_count += 1
                            corr_difference_from_ma = optimal_corr - moving_average
                            if (corr_difference_from_ma > tolerance) & (len(correlation_history) == ma_window):
                                exit_interpolation_types_loop = True
                                exit_yval_loop = True
                                # print('break yval loop:', exit_yval_loop)
                                break
    
                    if exit_interpolation_types_loop:
                        exit_candidate_x2_values_loop = True
                        # print('break interpol loop:', exit_interpolation_types_loop)
                        break 
                    
                if exit_candidate_x2_values_loop:
                    # print('break x2 loop:', exit_candidate_x2_values_loop)
                    print()
                    print(f'\nBest corr: {optimal_corr}\nStopped trigger iteration at: {moving_average}\nCorr diff: {corr_difference_from_ma}\nTolerance: {tolerance}\nIteration halted: {corr_difference_from_ma > tolerance}')
                    print()
                    break 
                
            print(f'\nTotal loop iteration: {loop_count}\n')
            
            df_final['y_two_lines'] = optimal_y_two_lines
                                
            return df_final, optimal_x2, optimal_corr, optimal_interpolation_type
        
        
        def use_min_from_non_overlapping_window(df, target_signal_cleaned_rescaled, x1, x3, desired_no_of_min_points=10, show_plot=False):
            """
            Interpolate a signal within a specified range using PCHIP interpolation and update the DataFrame.
        
            Parameters:
                df (pd.DataFrame): The DataFrame containing the signal data.
                target_signal_cleaned_rescaled (str): The column name of the signal to be interpolated.
                x1 (int): The start of the interpolation range.
                x3 (int): The end of the interpolation range.
        
            Returns:
                None: The function updates the DataFrame with the interpolated signal.
            """
            x_temp = np.arange(x1, x3 + 1)
            y_temp = df[target_signal_cleaned_rescaled][x_temp].values
            window_size = len(x_temp) // desired_no_of_min_points
            min_indices, min_values = find_minimum_in_non_overlapping_window(x_temp, y_temp, window_size, show_plot=show_plot)
            interpolator = PchipInterpolator(min_indices, min_values)
            x_interp = np.linspace(min(min_indices), max(min_indices), len(x_temp))
            y_interp = interpolator(x_interp)
            df[f'{target_signal_cleaned_rescaled}_original'] = df[target_signal_cleaned_rescaled]
            df.loc[x_temp, target_signal_cleaned_rescaled] = y_interp
            
            if show_plot:
                plt.plot(x_temp, y_interp)
            return df
        
        # import pdb; pdb.set_trace()
        ## splitting the data, so that if we want to not include certain % for only correlation to improve the correlation (due to noise)
        df_first_x_percent, df_last_x_percent = split_dataframe_by_percentage(group_df, first_split_df_percentage)
        df_first_x_percent = df_first_x_percent.copy()
        # df_first_x_percent.loc[:, 'first_x_percent'] = True ## we will need this tag for later when clustering
        df_first_x_percent['first_x_percent'] = True ## we will need this tag for later when clustering

        df_input = df_last_x_percent.copy()
    
        peaks = df_input.index[df_input['peaks']].values
        troughs = np.unique(np.append(df_input.index[0], df_input.index[-1])) ## adding last value as trough so the curve will complete
        
        troughs_group_bool = df_input['troughs_group'].notna()
        peaks_group_bool = df_input['peaks_group'].notna()
        
        x1 = df_input[troughs_group_bool].index[0]
        x3 = df_input[troughs_group_bool].index[-1]
        x_end = peaks_group_bool.index[-1]
        
        # =============================================================================
        #         # average of 5/10 cycles back/forward (so that it doesn't start at some off place)
        # =============================================================================
        no_of_cycle_fw = int(round(10 * df_input['sf'].iloc[0]))
        no_of_cycle_bw = int(round(5 * df_input['sf'].iloc[0]))
        y1_alternative = df_input.loc[x1-no_of_cycle_bw:x1+no_of_cycle_fw][target_signal_cleaned_rescaled].median()
        
        # y1 = df_input[target_signal_cleaned_rescaled][x1]
        y1 = y1_alternative
        y3 = df_input[target_signal_cleaned_rescaled][x3]
        y_end = df_input[target_signal_cleaned_rescaled][x_end]
        
        # tried to do the best to match the first y point, but with exponential, it's not easy.
        y_value_user_defined = [y1, y3, y_end]
    
        curve_corrs = []
        keep_cols = ['y_simulated_exponential', target_signal_cleaned_rescaled]
        
        
        # =============================================================================
        #         ## using 'bottom' part of the data that is smoothed (if we want)
        # =============================================================================
        if use_min_from_non_overlapping_window_func:
            df_input = use_min_from_non_overlapping_window(df_input, 
                                                           target_signal_cleaned_rescaled, x1, x3, 
                                                           desired_no_of_min_points=desired_no_of_min_points, 
                                                           show_plot=False)
        # import pdb; pdb.set_trace()
        for exponent_factor in exponent_factors:

            # =============================================================================
            #         ## NOTE that we are replacing the 'drop' of exponential curve with actual rescaled pressure data. This is because
            #         ## later we will be comparing the correlation with two_line data, which also has actual rescaled pressure data at the 'drop'
            # =============================================================================
    
            df_temp = simulate_x_and_y_using_peaks_and_troughs(df_input[['troughs_group', target_signal_cleaned_rescaled]].copy(), peaks, troughs, y_value_user_defined, exponent_factor=exponent_factor)   
            df_temp['y_simulated_exponential'] = np.where(df_temp['troughs_group'].isna(), df_temp[target_signal_cleaned_rescaled], df_temp['y_simulated_exponential'])
            df_temp = pd.concat([df_first_x_percent[[target_signal_cleaned_rescaled]], df_temp[keep_cols]])
            df_temp['y_simulated_exponential'] = np.where(df_temp['y_simulated_exponential'].isna(), df_temp[target_signal_cleaned_rescaled], df_temp['y_simulated_exponential'])
    
            # df_temp['peaks'] = np.where(df_temp.index == peaks[0], True, False)
            pearson, spearman, kendall, _, _ = check_corr_between_scaled_and_simulated(df_temp, target_signal_cleaned_rescaled,
                                                                                       print_corr=False, show_plot=False)
            if mean_corr_methods:
                corr_included = np.mean([pearson, spearman, kendall])
            else:
                corr_included = pearson
                
                
            curve_corrs.append(corr_included)
        
        # finding the best curve fit first
        best_curve_fit_index = np.argmax(curve_corrs)
        best_curve_fit = exponent_factors[best_curve_fit_index]
        best_curve_fit_corrs = curve_corrs[best_curve_fit_index]
        df_input = simulate_x_and_y_using_peaks_and_troughs(df_input, peaks, troughs, y_value_user_defined, exponent_factor=best_curve_fit) 
        
        # note that we do this again since .. we re ran it    
        df_input['y_simulated_exponential'] = np.where(df_input['troughs_group'].isna(), df_input[target_signal_cleaned_rescaled], df_input['y_simulated_exponential'])

            
        # =============================================================================
        #         # elbow point for exponential curve (search starting point)
        # =============================================================================
        kneedle = KneeLocator(df_input[troughs_group_bool].index.values, df_input[troughs_group_bool]['y_simulated_exponential'], S=1, curve="convex")
        x2 = kneedle.knee
                
        df_final, optimal_x2, optimal_corr, optimal_interpolation_type = calculate_optimal_inflection_point(x1, x2, x3, y1, y3, 
                                                                                df_first_x_percent, 
                                                                                df_input, 
                                                                                target_signal_cleaned_rescaled, 
                                                                                peaks_group_bool, troughs_group_bool, 
                                                                                search_step_size, 
                                                                                ma_window=ma_window,
                                                                                tolerance=tolerance)
        
        df_final['y_simulated_exponential'] = np.where(df_final['y_simulated_exponential'].isna(), df_final[target_signal_cleaned_rescaled], df_final['y_simulated_exponential'])
        df_final['first_x_percent'] = df_final['first_x_percent'].fillna(False)
        
        results = pd.DataFrame({
            'optimal_exponent_factor': [best_curve_fit],
            'initial_inflection': [x2],
            'initial_corr': [best_curve_fit_corrs], 
            'optimal_inflection': [optimal_x2],
            'optimal_corr': [optimal_corr],
            'optimal_interpolation_type': [optimal_interpolation_type]
        })
        
        return [results, df_final]
    

    # Function to process groups in parallel
    def process_groups_in_parallel(groups, target_signal_cleaned_rescaled, first_split_df_percentage, interpolation_types, exponent_factors, mean_corr_methods, search_step_size, ma_window, tolerance):
        results_list = []
        df_final_list = []
        with tqdm(total=len(groups)) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(get_optimal_line_fit_by_group, group_df, target_signal_cleaned_rescaled, first_split_df_percentage, 
                                           interpolation_types, exponent_factors, mean_corr_methods, search_step_size, ma_window, tolerance) for _, group_df in groups}
                for future in concurrent.futures.as_completed(futures):
                    results, df_final = future.result()
                    results_list.append(results)
                    df_final_list.append(df_final)
                    
                    # Update the progress bar
                    pbar.update(1)
                    
            return pd.concat(results_list, ignore_index=True), pd.concat(df_final_list, ignore_index=True)
        
    
    def process_groups_sequentially(groups, target_signal_cleaned_rescaled, first_split_df_percentage, interpolation_types, exponent_factors, mean_corr_methods, search_step_size, ma_window, tolerance):
        results_list = []
        df_final_list = []
    
        with tqdm(total=len(groups)) as pbar:
            for _, group_df in groups:
                results, df_final = get_optimal_line_fit_by_group(group_df, target_signal_cleaned_rescaled, first_split_df_percentage, interpolation_types, exponent_factors, mean_corr_methods, search_step_size, ma_window, tolerance)
                results_list.append(results)
                df_final_list.append(df_final)
                pbar.update(1)
    
        return pd.concat(results_list, ignore_index=True), pd.concat(df_final_list, ignore_index=True)

    # =============================================================================
    # min-max scaling by peak and trough (not by group)
    # =============================================================================
    
    df['trough_to_trough'] = df['troughs'].cumsum()
    df['troughs_group'] = df['troughs'].cumsum()
    df['peaks_group'] = df['peaks'].cumsum()
    
    df['troughs_group'] = np.where(df['troughs_group'] == df['peaks_group'], np.nan, df['troughs_group'])
    df['peaks_group'] = np.where(df['troughs_group'].isna(), df['peaks_group'], np.nan)
    
    df['troughs_group'].values[-1] = df['troughs_group'].values[-2]
    df['peaks_group'].values[-1] = df['peaks_group'].values[-2]
    df['trough_to_trough'].values[-1] = df['trough_to_trough'].values[-2]
    
    if apply_gaussian_filter:
        df['f{target_signal_cleaned}_not_filtered'] = df[target_signal_cleaned]
        df[target_signal_cleaned] = gaussian_filter1d(df[target_signal_cleaned].values, int(round(gaus_sigma*df['sf'][0])))
    
    
    # =============================================================================
    # ## min-max scaling for pressure data. This is because they are quite differently scaled and I am not convinced that all the scaling factors are right..
    # =============================================================================
    df = min_max_scale_data_by_group(df.copy(), group_var='trough_to_trough', 
                                     in_var=target_signal_cleaned, 
                                     out_var=target_signal_cleaned_rescaled, 
                                     desired_min=0, desired_max=1)   
    

    # import pdb; pdb.set_trace()
    # Apply the function to groups
    ### CHANGE first_split_df_percentage so it will work better -- this is because the first trough point can be terrible for estimation
    grouped = df.groupby('trough_to_trough')
        
    # grouped_results, df_final_results = process_groups_in_parallel(grouped, target_signal_cleaned_rescaled,
    #                                                                first_split_df_percentage,
    #                                                                interpolation_types,
    #                                                                exponent_factors, 
    #                                                                mean_corr_methods,
    #                                                                search_step_size,
    #                                                                ma_window,
    #                                                                tolerance)
    
    grouped_results, df_final_results = process_groups_sequentially(grouped, target_signal_cleaned_rescaled,
                                                                   first_split_df_percentage,
                                                                   interpolation_types,
                                                                   exponent_factors, 
                                                                   mean_corr_methods,
                                                                   search_step_size,
                                                                   ma_window,
                                                                   tolerance)
    # import pdb; pdb.set_trace()
    # merging it back, hopefully it'll be faster this way
    df_final_results = df_final_results.sort_values(by=['source','time']).reset_index(drop=True).drop(columns='y_simulated', errors='ignore')
    df_final = pd.merge(df_final_results, df_merge_later, left_index=True, right_index=True)
  
    df_inflection_results = grouped_results.sort_values(by=['initial_inflection']).reset_index(drop=True)
    ## adding middle point between the two
    df_inflection_results['middle_inflection'] = (df_inflection_results['initial_inflection'] + df_inflection_results['optimal_inflection'])//2
    
    if show_plot:
        plt.figure(figsize=(16, 6))
        plt.plot(df_final.index.values, df_final[target_signal_cleaned_rescaled].values, label=target_signal_cleaned_rescaled)
        if use_min_from_non_overlapping_window_func:
            plt.plot(df_final.index.values, df_final[f'{target_signal_cleaned_rescaled}_original'].values, label=f'{target_signal_cleaned_rescaled}_original')
        plt.plot(df_final.index.values, df_final['y_two_lines'].values, label='y_two_lines')
        plt.plot(df_final.index.values, df_final['y_simulated_exponential'].values, label='y_simulated_exponential')
                
        for i, cp in enumerate(df_inflection_results['optimal_inflection']):
            plt.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            plt.text(cp, plt.ylim()[1], str(i), va='bottom', ha='center', color='red', fontsize=8, alpha=0.5)

        for cp in df_inflection_results['initial_inflection']:
            plt.axvline(x=cp, color='green', linestyle='--', alpha=0.5)
            
        for cp in df_inflection_results['middle_inflection']:
            plt.axvline(x=cp, color='grey', linestyle='--', alpha=0.5)
        
        # Add legends for the vertical lines
        plt.plot([], [], color='red', linestyle='--', label='inflection point')
        plt.plot([], [], color='green', linestyle='--', label='curve elbow')
        plt.plot([], [], color='grey', linestyle='--', label='middle point')
        
        ncols = 7 if use_min_from_non_overlapping_window_func else 6
        # Move the legend to the outside of the graph
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=ncols)
        plt.show()
    
    return df_final, df_inflection_results



def add_inflection_point_to_main_df(df, df_inflection_point, inflection_point_column):
    inflection_temp = df_inflection_point[[inflection_point_column]].set_index(inflection_point_column).rename(index={inflection_point_column: 'index'})
    inflection_temp[inflection_point_column] = True
    df = pd.concat([df, inflection_temp], axis=1)
    df[inflection_point_column] = df[inflection_point_column].fillna(False)
    return df


