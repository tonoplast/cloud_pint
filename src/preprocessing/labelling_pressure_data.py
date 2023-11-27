# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 09:28:26 2023

@author: sungw
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.data_loader import load_processed_data, load_void_info_data
from src.preprocessing.curve_fit_clustering import attach_hand_labelled_void_info, remove_edges_by_group_and_first_last_excess
from src.preprocessing.curve_fit_clustering import chop_and_interpolate_target_signal
from src.preprocessing.curve_fit_clustering import min_max_scale_data_by_group
from src.preprocessing.curve_fit_clustering import get_best_curve_fit
from src.preprocessing.curve_fit_clustering import exponential_clustering, comparison_between_prelabel_and_semiauto_clustering
from src.preprocessing.curve_fit_clustering import find_best_estimate_parameters_for_given_inflection_point, cluster_from_single_point
from src.preprocessing.find_inflection_point import inflection_point_search, add_inflection_point_to_main_df
# import matplotlib.pyplot as plt


data_drive = 'O'
base_dir = Path(rf'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')

file_exception_wildcard =['12_013.acute.saline.0.4']
file_extension = '.parquet'

processed_folder_name_cc = 'Neural' # StaticPeak_0.5 # MovingPeak
processed_folder_name_void = 'AutoCurate'
void_file_name = 'VoidInfo.csv'

bladder_pressure_dir = processed_dir.joinpath('BladderPressure')
bladder_pressure_dir.mkdir(parents=True, exist_ok=True)


# This is not very well coded. It is just wrapped into 'run_bp_processing' function at the momment
# so that it's easy to just run
## TODO: best to move it into a config file of some sort to clean it up

def run_bp_processing():
    
    def get_range(start, end, step):
        return np.arange(start, end + step, step)

    def check_label_distance_diff(df, hand_label, auto_label):
        # import pdb; pdb.set_trace()
        import matplotlib.pyplot as plt
        hand_label_index = df[df[hand_label]].index.values
        auto_inflection_label_index = df[df[auto_label]].index.values
        auto_inflection_label_index_without_unknown = df[df[auto_label] & (df['label_hand'] != 'Unknown')].index.values
        diff = (hand_label_index - auto_inflection_label_index_without_unknown)/(df['sf'].mean())
        
        common_values = np.intersect1d(auto_inflection_label_index, auto_inflection_label_index_without_unknown)
        indices_in_array1 = np.where(np.isin(auto_inflection_label_index, common_values))[0]

            
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        median_diff = np.median(diff)
        min_diff = np.min(diff)
        max_diff = np.max(diff)
        
        print()
        print(f'Mean diff: {mean_diff:.2f} s')
        print(f'Std diff: {std_diff:.2f} s')
        print(f'Median diff: {median_diff:.2f} s')
        print(f'Min diff: {min_diff:.2f} s')
        print(f'Max diff: {max_diff:.2f} s')
        
            # Create a bar chart
        plt.figure(figsize=(6, 12))  # Adjust the figure size as needed
        plt.barh(indices_in_array1, diff, color='b', alpha=0.7)
        
        # Add a horizontal line at y=0
        plt.axvline(0, color='r', linestyle='-', linewidth=1, alpha=0.5)
        
        # Invert the y-axis
        plt.gca().invert_yaxis()
        
        # Customize labels and title
        plt.yticks(indices_in_array1, fontsize=6)
        plt.xlabel('Difference (sec)')
        plt.ylabel('Voids')
        plt.title('Difference from hand-label')
        
        # Show the plot
        plt.tight_layout()
        plt.show()

    
    ATTACH_VOID_INFO = True
    
    GROUP_VAR_ALL = 'source'
    GROUP_VAR_INDIVIDUAL = 'trough_to_trough'
    TARGET_SIGNAL = 'bladder_pressure' # bladder_pressure
    TARGET_SIGNAL_CLEANED = f'{TARGET_SIGNAL}_cleaned'
    TARGET_SIGNAL_CLEANED_SCALED = f'{TARGET_SIGNAL_CLEANED}_scaled'
    TARGET_SIGNAL_CLEANED_SCALED_FILTERED = f'{TARGET_SIGNAL_CLEANED_SCALED}_filtered'
    TARGET_SIGNAL_CLEANED_RESCALED = f'{TARGET_SIGNAL_CLEANED}_rescaled'
    
    LABEL_VARNAME='label_auto'
    LABEL_HAND_VARNAME='label_hand'
    INFLECTION_POINT_VAR='middle_inflection' #  'middle_inflection' 'optimal_inflection' between_BP_and_TP
    FLOW_RATE='FLOWRATE(mL/hr)'
    SAMPLING_FREQUENCY = 'sf'
    RANK_METHODS= ['dense', 'average', 'min', 'max', 'ordinal']
    MODEL_SIGNALS = 'y_simulated_exponential' # y_two_lines // y_simulated_exponential


    ## for shifting comparison
    offset=100
    
    # this is done by just looking at the plot and checking the x-axis (index)
    unknown_void_index_list = []
    
    
    ## this is if we want to chop some bad looking data by hand (turned off)
    chop_bad_data = False
    chop_and_interpolate = [[7240, 7350], [10807,10881], [13102, 13204], [17473, 17612], [37985, 38046]]
    
    ## getting rid of some edges if wanted
    chop_edges_by_group_pct = [0, 0]
    chop_edges_combined_nrow = [0, 0] #[50, 300]
    
    save_data = True
    
    # =============================================================================
    # Parameters to find the best fit for curve fitting
    # =============================================================================
    # Define your parameters
    cutoff_freq_start = 0.002
    cutoff_freq_end = 0.05
    cutoff_freq_step = 0.002
    
    order_start = 1
    order_end = 5
    order_step = 1 
    
    exponent_factor_start = 1
    exponent_factor_end = 6
    exponent_factor_step = 1
    
    cutoff_freq_range = get_range(cutoff_freq_start, cutoff_freq_end, cutoff_freq_step)
    order_range = get_range(order_start, order_end, order_step)
    exponent_factor_range_for_curve_fit = get_range(exponent_factor_start, exponent_factor_end, exponent_factor_step)
    filtertypes = [False] # [False, True] False: filtfilt, True: lfilt
    
    
    best_fit_param = () ## leaving it blank because it should run for new set of data
    # best_fit_param = (False, 0.03, 1, 5) ## if unknown, use ()
    
    
    # =============================================================================
    # Parameters to find the best inflection point
    # =============================================================================
    exponent_factor_start = 1
    exponent_factor_end = 10
    exponent_factor_step = 1
    
    use_min_from_non_overlapping_window_func = False
    desired_no_of_points_for_non_overlapping_window = 10
    apply_gaussian_filter = False
    gaus_sigma = 5
    first_split_df_percentage = 0 #10
    interpolation_types = [''] # ['cubic', 'pchip', 'akima']
    exponent_factor_range_for_inflection_point = get_range(exponent_factor_start, exponent_factor_end, exponent_factor_step)
    mean_corr_methods = False
    search_step_size_for_inflection_point = 1
    moving_average_window_for_line_fit_cum_corr = 10
    tolerance_for_exiting_grid_search=0.2
    show_plot_for_inflection_point=True
    
    
    # =============================================================================
    # Parameters to find the best clustering to match inflection point
    # =============================================================================

    PREFRED_CLUSTER = 3 # target cluster
    
    num_of_cluster_start = 4 # 4 should be minimum because we should at least have 3 up 1 down
    num_of_cluster_end = 10
    num_of_cluster_step = 1
    
    exponent_degree_start = 0.1 # 0.1
    exponent_degree_end = 6
    exponent_degree_step = 0.1
    
    num_of_cluster_range = get_range(num_of_cluster_start, num_of_cluster_end, num_of_cluster_step)
    exponent_degree_range = get_range(exponent_degree_start, exponent_degree_end, exponent_degree_step)
    
    best_estimate_param = ()


    # =============================================================================
    # Load data and add void info
    # =============================================================================
    main_df = load_processed_data(data_drive, processed_dir, processed_folder_name_cc, file_extension, file_exception_wildcard)
    all_df_cols = [GROUP_VAR_ALL, TARGET_SIGNAL, 'time', SAMPLING_FREQUENCY]
    all_df = main_df[all_df_cols]
    # main_df = main_df.drop(columns=all_df_cols)
    
    if ATTACH_VOID_INFO:
        voidinfo_df = load_void_info_data(data_drive, base_dir, processed_folder_name=processed_folder_name_void, file_name=void_file_name)
        
        ## merging all data with void information (where exists)
        merged_df = attach_hand_labelled_void_info(all_df, voidinfo_df, GROUP_VAR_ALL, TARGET_SIGNAL, LABEL_HAND_VARNAME, show_plot=True, unknown_index_list=unknown_void_index_list)
        merged_df[FLOW_RATE] = merged_df[FLOW_RATE].fillna(method="ffill")
        
    else:
        merged_df = all_df.copy()
    
    contains_baseline = all_df['source'].str.contains('0.1')
    
    df_baseline = merged_df[contains_baseline].reset_index(drop=True)
    merged_df =  merged_df[~contains_baseline].reset_index(drop=True)
    
    # =============================================================================
    # chopping bad data out
    # =============================================================================
    if chop_bad_data:
        df = chop_and_interpolate_target_signal(merged_df.copy(), TARGET_SIGNAL, chop_and_interpolate, show_plot=True)
    else:
        df = merged_df.copy()
        df[TARGET_SIGNAL_CLEANED] = df[TARGET_SIGNAL]
    
    
    # =============================================================================
    # remove edges if wanted, and do min-max scaling (0-1)
    # =============================================================================
    
    ## removing some unnecessary data at beginning and end (edges)
    df = remove_edges_by_group_and_first_last_excess(df, GROUP_VAR_ALL, 
                                                     top_pct=chop_edges_by_group_pct[0], bot_pct=chop_edges_by_group_pct[1], 
                                                     n_first_rows_to_remove=chop_edges_combined_nrow[0], n_last_rows_to_remove=chop_edges_combined_nrow[1])
    
    ## min-max scaling for pressure data. This is because they are quite differently scaled and I am not convinced that all the scaling factors are right..
    df = min_max_scale_data_by_group(df.copy(), group_var=GROUP_VAR_ALL, 
                                     in_var=TARGET_SIGNAL_CLEANED, 
                                     out_var=TARGET_SIGNAL_CLEANED_SCALED, 
                                     desired_min=0, desired_max=1)
    
    
    
    
    # =============================================================================
    # Find the best curve fitting
    # This is used for peak/trough labels
    # =============================================================================
    
    
    if best_fit_param == ():
    
        # Create a list of parameter combinations
        fit_param_combinations = [(filtertype, freq, order, factor) for filtertype in filtertypes
                              for freq in cutoff_freq_range
                              for order in order_range
                              for factor in exponent_factor_range_for_curve_fit]
        
        print(f'\nNumber of parameter combinations for curve fitting: {len(fit_param_combinations)}\n')
        
        ## loop through to get the best params (slow)
        all_calculated_values_curve = []
        print('Searching for the best parameter for curve fitting..')
        for params in tqdm(fit_param_combinations):
            _, pearson, spearman, kendall = get_best_curve_fit(params, df, GROUP_VAR_ALL, TARGET_SIGNAL_CLEANED_SCALED, TARGET_SIGNAL_CLEANED_SCALED_FILTERED, SAMPLING_FREQUENCY)
            calculated_values = [pearson, spearman, kendall]
            all_calculated_values_curve.append(calculated_values)
        
        
        max_pos_per_corr = [np.argmax(i) for i in zip(*all_calculated_values_curve)]
        max_val_per_corr = [np.max(i) for i in zip(*all_calculated_values_curve)]
        
        # max_pos_per_corr = set(max_pos_per_corr)
        # best_fit_param = [param_combinations[i] for i in max_pos_per_corr]
        
        best_fit = max_pos_per_corr[0]
        best_fit_param = fit_param_combinations[best_fit]
        
    
    print(f'\nBest Parameters:\nFilter Type: {best_fit_param[0]}\nCutoff Freq: {best_fit_param[1]}\nFilter Order: {best_fit_param[2]}\nexponent factor:{best_fit_param[3]}\n')
    
    df, pearson, spearman, kendall = get_best_curve_fit(best_fit_param, df, GROUP_VAR_ALL, TARGET_SIGNAL_CLEANED_SCALED, TARGET_SIGNAL_CLEANED_SCALED_FILTERED, SAMPLING_FREQUENCY,
                                                    show_plot=True, print_peaks=True, print_corr=True)
    
    
    
    
    # =============================================================================
    # Finding inflection point - note that previous curve fit is not used here and
    # gets replaced by individual pressure curve fit
    # =============================================================================
    df_temp = df.copy()
    df_tempX = df.copy()
    # df_temp = df_tempX.copy()
    
    # target_signal_cleaned=TARGET_SIGNAL_CLEANED
    # target_signal_cleaned_rescaled=TARGET_SIGNAL_CLEANED_RESCALED
    # apply_gaussian_filter=False
    # gaus_sigma=5
    # first_split_df_percentage=10
    # interpolation_types=['']
    # exponent_factors=[5], #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # mean_corr_methods=False
    # search_step_size=1
    # ma_window=10
    # tolerance=0.0001 #tolerance=0.2, 
    # show_plot=True
    
    df, df_inflection_point = inflection_point_search(df_temp, 
                                                    target_signal_cleaned=TARGET_SIGNAL_CLEANED, 
                                                    target_signal_cleaned_rescaled=TARGET_SIGNAL_CLEANED_RESCALED,
                                                    use_min_from_non_overlapping_window_func=use_min_from_non_overlapping_window_func, #false
                                                    desired_no_of_min_points=desired_no_of_points_for_non_overlapping_window, #10
                                                    apply_gaussian_filter=apply_gaussian_filter, #False
                                                    gaus_sigma=gaus_sigma, #5
                                                    first_split_df_percentage=first_split_df_percentage, #10
                                                    interpolation_types=interpolation_types, #['']
                                                    exponent_factors=exponent_factor_range_for_inflection_point, # [1,2,3,4,5,6,7,8,9,10]
                                                    mean_corr_methods=mean_corr_methods, #False
                                                    search_step_size=search_step_size_for_inflection_point, #1
                                                    ma_window=moving_average_window_for_line_fit_cum_corr, #10
                                                    tolerance=tolerance_for_exiting_grid_search, #tolerance=0.2, 
                                                    show_plot=show_plot_for_inflection_point) #True
    
    
    
    
    df_temp_2 = df.copy()
    
    # df = df_temp_2.copy()
    # =============================================================================
    # Adding 'inflection points' to the main dataframe
    # =============================================================================
    df = add_inflection_point_to_main_df(df, df_inflection_point, 'initial_inflection')
    df = add_inflection_point_to_main_df(df, df_inflection_point, 'middle_inflection')
    df = add_inflection_point_to_main_df(df, df_inflection_point, 'optimal_inflection')
    
    
    # =============================================================================
    # Auto unknown detection
    # =============================================================================
    unique_label_counts = df.groupby('trough_to_trough')[LABEL_HAND_VARNAME].nunique()
    unknown_label = unique_label_counts[unique_label_counts == 1].reset_index()
    mask = df['trough_to_trough'].isin(unknown_label['trough_to_trough'])
    # unknown_indices = df.index[mask]
    df[LABEL_HAND_VARNAME] = np.where(mask, 'Unknown', df[LABEL_HAND_VARNAME])
    
    

    # =============================================================================
    # Clustering around pre-defined inflection point
    # =============================================================================    
    # best_estimate_param = (10, 1)
    
    ## Cluster from 'inflection' point to peak
    ## this will be 11 clusters (0 being bp, then 9 in )
    df = cluster_from_single_point(df.copy(),
                              best_estimate_param,
                              target_signal_for_clustering = TARGET_SIGNAL_CLEANED_RESCALED,
                              target_group_segment='troughs_group',
                              single_inflection_point=INFLECTION_POINT_VAR,
                              model_signal=MODEL_SIGNALS,
                              label_name=f'label_{INFLECTION_POINT_VAR}',
                              add_bp_cluster=True,
                              add_opcp_cluster=False,
                              show_plot=True)
    
    ## Cluster from 'bp' to inflection point
    ## note that above needs to be run with 'add_bp_cluster=True' to get bp information
    ## This is because bp information is dependent on the inflection point above
    df = cluster_from_single_point(df.copy(), 
                              best_estimate_param,
                              target_signal_for_clustering = TARGET_SIGNAL_CLEANED_RESCALED,
                              target_group_segment='bp_group',
                              single_inflection_point='bp_point',
                              model_signal=MODEL_SIGNALS,
                              label_name='label_bp',
                              add_bp_cluster=False,
                              add_opcp_cluster=True,
                              show_plot=True)
    
    
    
    # =============================================================================
    # Find best estimate param for inflection point 
    # =============================================================================
    
    
    
    num_of_cluster_range = get_range(num_of_cluster_start, num_of_cluster_end, num_of_cluster_step)
    exponent_degree_range = get_range(exponent_degree_start, exponent_degree_end, exponent_degree_step)
    
    best_estimate_param = ()

    
    if best_estimate_param == ():
        
        # Create a list of parameter combinations
        estimate_param_combinations = [(num_cluster, exponent_degree)
                                  for num_cluster in num_of_cluster_range
                                  for exponent_degree in exponent_degree_range]
        
        print(f'\nNumber of parameter combinations for inflection: {len(estimate_param_combinations)}\n')
        
        # estimate_param_combinations = [(4, 0.1)]
        # estimate_param_combinations = [(6, 3.9)]
        # estimate_param_combinations = [(6, 1.3)]
    
        best_estimate_param, df_ranking = find_best_estimate_parameters_for_given_inflection_point(df, 
                                                                                                    estimate_param_combinations, 
                                                                                                    offset, 
                                                                                                    target_signal_main=TARGET_SIGNAL_CLEANED, # target_signal_main
                                                                                                    target_signal_for_clustering=TARGET_SIGNAL_CLEANED_RESCALED, # target_signal_for_clustering
                                                                                                    target_signal_for_modelling=MODEL_SIGNALS, # y_simulated_exponential#'two_lines' target_signal_for_modelling
                                                                                                    label_varname=LABEL_VARNAME, 
                                                                                                    label_prelabel_varname=LABEL_HAND_VARNAME, 
                                                                                                    group_var=GROUP_VAR_INDIVIDUAL, 
                                                                                                    inflection_point_var=INFLECTION_POINT_VAR,
                                                                                                    rank_methods=RANK_METHODS,
                                                                                                    preferred_cluster=PREFRED_CLUSTER)
        
    
    # =============================================================================
    # Cluster bladder pressure data (based on inflection point)
    # =============================================================================
    
    df = exponential_clustering(df.copy(), 
                                  target_signal_for_clustering=TARGET_SIGNAL_CLEANED_SCALED, #TARGET_SIGNAL_CLEANED_RESCALED, 
                                  target_signal_for_modelling=MODEL_SIGNALS,
                                  out_var_label=LABEL_VARNAME, 
                                  num_of_clusters=best_estimate_param[0], exponent_degree=best_estimate_param[1], 
                                  add_bp_cluster=False,
                                  give_warnings=False,
                                  show_plot=True)
    
    
    
    
    
    ## check with the hand labels
    df_check, first_tp_indices_value_count, \
        inflection_diff, inflection_corr_rho, inflection_corr_p, \
        normal_dist_val, normal_dist_p, skewness, kurt, \
            edge_score, cv_score, latency_diff = comparison_between_prelabel_and_semiauto_clustering(df, 
                                                                                                    GROUP_VAR_INDIVIDUAL, 
                                                                                                    target_signal=TARGET_SIGNAL_CLEANED, 
                                                                                                    hand_label=LABEL_HAND_VARNAME,
                                                                                                    auto_label=LABEL_VARNAME,
                                                                                                    inflection_point_var=INFLECTION_POINT_VAR,
                                                                                                    # params=best_estimate_param,
                                                                                                    drop_cols=[], 
                                                                                                    offset=offset, 
                                                                                                    show_plot=True)
    
    
    
    first_max = np.argsort(first_tp_indices_value_count)[::-1].iloc[0]
    max_tp_val_index_on_first = first_tp_indices_value_count.index[first_max]
    
    print(f'Inflection point: Cluster #{max_tp_val_index_on_first} or less')
    df_temp_3 = df.copy()
    
    
    # =============================================================================
    # patching data back together with original data
    # =============================================================================
    
    ## for now, using -1 as label for baseline recordings
    df_baseline.loc[:, 'label_bp'] = -1
    df_baseline.loc[:, f'label_{INFLECTION_POINT_VAR}'] = -1
    df_baseline.loc[:, LABEL_VARNAME] = -1
    df_baseline.loc[:, TARGET_SIGNAL_CLEANED] = df_baseline[TARGET_SIGNAL]
    df_baseline.loc[:, 'bl_group'] = 1
    
    # import pdb; pdb.set_trace()
    if save_data:
        
        # Void
        grouped = df.groupby('source')
        
        for group_name, group_data in grouped:
            # Define the file name for this group (you can customize the naming)
            file_name = f'{group_name}.bp_void.parquet'
            
            # Save the group data to the file
            group_data.to_parquet(bladder_pressure_dir.joinpath(file_name), index=False)
         
        # baseline
        grouped = df_baseline.groupby('source')
        
        for group_name, group_data in grouped:
            # Define the file name for this group (you can customize the naming)
            file_name = f'{group_name}.bp_baseline.parquet'
            
            # Save the group data to the file
            group_data.to_parquet(bladder_pressure_dir.joinpath(file_name), index=False)

    check_label_distance_diff(df, 'between_BP_and_TP', 'optimal_inflection')
    check_label_distance_diff(df, 'between_BP_and_TP', 'middle_inflection')



def main():
    run_bp_processing()


if __name__ == "__main__":
    main()


