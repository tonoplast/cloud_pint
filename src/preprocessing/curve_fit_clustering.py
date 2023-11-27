# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:37:24 2023

@author: sungw
"""

import pandas as pd
import numpy as np
from numpy import concatenate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.preprocessing.preproc_tools import butter_lowpass_filter
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.interpolate import interp1d
from itertools import permutations
from scipy.stats import rankdata, skew, kurtosis, kstest, variation
from tqdm import tqdm
import concurrent.futures



def attach_hand_labelled_void_info(main_df, voidinfo_df, group_var, target_signal, label_hand_varname, show_plot=True, unknown_index_list=[]):
    # import pdb; pdb.set_trace()
    
    # print('Attaching void information to the data, please wait..')
    voidinfo_df[group_var] = voidinfo_df['ID'] + '.acute.saline.' + voidinfo_df['DAY'].str.split(expand=True)[1] + '.' + voidinfo_df['FILENO.'].astype(str)
    voidinfo_df = voidinfo_df.drop(columns=['ID','DAY','FILENO.'])
    
    merged_df = pd.merge(main_df, voidinfo_df, how='left', on=group_var)
    
    # reordering columns
    pressure_col_ordered = ['TP','OP','CP','BP']
    rest_of_cols = list(merged_df.columns[np.logical_not(merged_df.columns.isin(pressure_col_ordered))])
    merged_df = merged_df[rest_of_cols + pressure_col_ordered]
    
    # get the labelling in
    for col in pressure_col_ordered:
     merged_df[f'{col}_start'] = np.where(merged_df['time'] >= merged_df[col], 1, np.nan)

    merged_df['start_sum'] = merged_df[['TP_start','OP_start','CP_start','BP_start']].sum(axis=1)
    merged_df[label_hand_varname] = merged_df['start_sum'].replace({0:pd.NA, 1:'TP', 2:'OP', 3:'CP', 4:'BP'})
    
    merged_df['position'] = merged_df.groupby(group_var).cumcount()
    merged_df[label_hand_varname] = np.where(merged_df['position'] == 0, 'BP', merged_df[label_hand_varname])
    merged_df[label_hand_varname] = merged_df.groupby(group_var)[label_hand_varname].ffill()
    
    # Find indices of first 'TP's -- this is convoluted but much faster
    first_tp_indices = merged_df[merged_df['TP_start']==1].groupby([group_var, 'TP']).apply(lambda x: x.index[0])  
    merged_df['between_BP_and_TP'] = merged_df.index.isin(first_tp_indices)
    
    cols_to_drop = pressure_col_ordered + [f'{col}_start' for col in pressure_col_ordered] + ['start_sum', 'position']
    merged_df = merged_df.drop(columns=cols_to_drop).drop_duplicates()
    
    merged_df['between_BP_and_TP'] = np.where(merged_df['between_BP_and_TP'], 1, np.nan)
    merged_df['between_BP_and_TP'] = merged_df.groupby([group_var, 'time'])['between_BP_and_TP'].ffill()
    merged_df['between_BP_and_TP'] = np.where(merged_df['between_BP_and_TP']==1, True, False)
       
    merged_df = merged_df.groupby([group_var,'time']).last().reset_index()
    
    
# =============================================================================
#     TODO: TEST THIS!
# =============================================================================
    if unknown_index_list:
        for i in unknown_index_list:
            merged_df.loc[i[0]:i[1], label_hand_varname] = 'Unknown'    
        label_to_color = {'BP': 'cornflowerblue', 'TP': 'firebrick', 'OP': 'lightsalmon', 'CP': 'black', 'Unknown': 'grey'}
    else:
        label_to_color = {'BP': 'cornflowerblue', 'TP': 'firebrick', 'OP': 'lightsalmon', 'CP': 'black'}

    
    grouped = merged_df.groupby(label_hand_varname)
    # Create an array of indices as x-axis values
    indices = merged_df.index
    
    if show_plot:
        plt.figure(figsize=(15, 6))
        
        # Create an empty list to store scatter objects
        scatter_objects = []

        for label, color in label_to_color.items():
            group = grouped.get_group(label)  # Get the group by label
            scatter = plt.scatter(indices[group.index], group[target_signal], label=label, color=color, s=5)
            scatter_objects.append(scatter)
    
        # Set the legend in the desired order
        plt.legend(handles=scatter_objects, labels=label_to_color.keys())         
        plt.xlabel('Index')
        plt.ylabel('Bladder Pressure')
        plt.title('Pre-labelling (where exists)')
        plt.legend()
        plt.show()
    
    return merged_df


def chop_and_interpolate_target_signal(df, target_signal, chop_and_interpolate, show_plot=True):
    """
    Interpolate the 'bladder_pressure' column of a DataFrame by replacing specified ranges with NaN and using spline interpolation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - chop_and_interpolate (list of lists): List of index ranges to replace with NaN and interpolate.

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with 'bladder_pressure' column interpolated.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    # df_copy = df.copy()
    
    df[f'{target_signal}_cleaned'] = df[target_signal]

    # Replace the specified ranges with NaN in the 'bladder_pressure' column
    for start, end in chop_and_interpolate:
        df.loc[start:end, f'{target_signal}_cleaned'] = np.nan

    # Create a mask to identify NaN values in the 'bladder_pressure' column
    nan_mask = df[f'{target_signal}_cleaned'].isna()

    # Get the indices and values for the non-NaN values
    x = df.index[~nan_mask]
    y = df[f'{target_signal}_cleaned'][~nan_mask]

    # Create a spline interpolation function
    spline_interpolation = interp1d(x, y, kind='slinear', bounds_error=False)

    # Use the interpolation function to fill NaN values
    df[f'{target_signal}_cleaned'] = np.where(nan_mask, spline_interpolation(df.index), df[f'{target_signal}_cleaned'])
    
    if show_plot:
        # Plot 'bladder_pressure' column after cleaning
        plt.figure(figsize=(10, 5))
        plt.plot(df.index.values, df[f'{target_signal}'].values, label='Before Chopping', color='red')
        plt.plot(df.index.values, df[f'{target_signal}_cleaned'].values, label='After Chopping', color='green')
        plt.xlabel('Index')
        plt.ylabel(f'{target_signal}_cleaned')
        plt.title(f'{target_signal} After Chopping')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df


def remove_edges_by_group_and_first_last_excess(df, group_var, top_pct=0.001, bot_pct=0.001, n_first_rows_to_remove=100, n_last_rows_to_remove=100):
    # import pdb; pdb.set_trace()
    
    def remove_edges(group, top_pct, bot_pct):
        n = len(group)
        top_to_remove = int(n * top_pct)
        bottom_to_remove = int(n * bot_pct)
        
        if bot_pct == 0:
            output = group.sort_index()[top_to_remove:]
        else:
            output = group.sort_index()[top_to_remove:-bottom_to_remove]
        
        return output
    
    df = df.groupby(group_var).apply(remove_edges, top_pct=top_pct, bot_pct=bot_pct).reset_index(drop=True)
    
    if n_last_rows_to_remove == 0:
        df = df.iloc[n_first_rows_to_remove:]
    else:
        df = df.iloc[n_first_rows_to_remove:-n_last_rows_to_remove]
        
    df = df.reset_index(drop=True)
    return df



def min_max_scale_data_by_group(df, group_var, in_var, out_var, desired_min=0, desired_max=1):
    
    # Function to perform min-max scaling for each group
    def perform_scaling(group):
        min_value = group[in_var].min()
        max_value = group[in_var].max()
        group[out_var] = ((group[in_var] - min_value) / (max_value - min_value)) * (desired_max - desired_min) + desired_min
        return group
    
    # Apply scaling for each group using groupby and apply
    df = df.groupby(group_var).apply(perform_scaling).reset_index(drop=True)
    return df
    

def filter_to_get_peak_and_trough(df, group_var, scaled_signal, scaled_filtered_signal, cutoff_frequency, order, use_lfilter=False, show_plot=True):
    # =============================================================================
    # Filtering to get peak and troughs
    # =============================================================================
    # Low-pass filter parameters
    # cutoff_frequency = 0.02 # Cutoff frequency as a fraction of the Nyquist frequency (0.5)
    # order = 1  # Filter order
    
    # Group the DataFrame by.. whatever - source / trough to trough
    grouped = df.groupby(group_var)

    # Iterate through groups and apply filtering code because the fs is slightly different
    for group_name, group_df in grouped:
        fs = 1 / (group_df['time'].iloc[1] - group_df['time'].iloc[0])  # Calculate fs for each group
        filtered_signal = butter_lowpass_filter(group_df[scaled_signal], cutoff_frequency, fs, order, use_lfilter=False)
        df.loc[df[group_var] == group_name, scaled_filtered_signal] = filtered_signal
        
    if show_plot:    
        # Plot the original signal and the filtered signal
        plt.figure(figsize=(15, 6))
        plt.plot(df.index.to_numpy(), df[scaled_signal].values, label='Original Signal (scaled)')
        plt.plot(df.index.to_numpy(), df[scaled_filtered_signal].values, label=f'Filtered (Low-pass, cutoff={cutoff_frequency})', linestyle='dashed')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Signal Smoothing using Low-Pass Filtering')
        plt.legend()
        plt.show()
    
    return df


def find_peaks_and_troughs(df, scaled_signal, filtered_signal, var_sf, peak_trough_widths=[20, 20], peak_trough_heights=[0.2, -0.6], peak_trough_prominence=[0.1, 0.1], print_peaks=True, show_plot=True):
    # =============================================================================
    # ## Find peaks and toughs
    # =============================================================================
    
    mean_sf = int(round(df[var_sf].mean()))
    peak_trough_widths = [i*mean_sf for i in peak_trough_widths]
    
    peaks, _ = find_peaks(df[filtered_signal], height=peak_trough_heights[0], width=peak_trough_widths[0], prominence=peak_trough_prominence[0])  
    troughs, _ = find_peaks(-df[filtered_signal], height=peak_trough_heights[1], width=peak_trough_widths[1], prominence=peak_trough_prominence[1])
    
    # adding toughs if first peak comes before trough & if last tough comes before peak
    if troughs[0] > peaks[0]:
        troughs = np.append(df.index[0], troughs)
        
    if troughs[-1] < peaks[-1]:
        troughs = np.append(troughs, df.index[-1])
    
    
    df['peaks'] = False
    df.loc[peaks, 'peaks'] = True
    
    df['troughs'] = False
    df.loc[troughs, 'troughs'] = True
    
    if print_peaks:
        print(f'\nPeaks: {len(peaks)}\nTroughs: {len(troughs)}\n')    
    
    if show_plot:
        plt.figure(figsize=(15, 6))
        plt.plot(df.index.to_numpy(), df[scaled_signal].values, label='Original Data (scaled)', alpha=0.2)
        plt.plot(df.index.to_numpy(), df[filtered_signal].values, label='Filtered Original Data', alpha=0.5)
        
        plt.scatter(df.index[peaks], df[filtered_signal][peaks], color='red', label='Detected Peaks', alpha=0.3)
        plt.scatter(df.index[troughs], df[filtered_signal][troughs], color='green', label='Detected Troughs', alpha=0.3)

        plt.scatter(df.index[peaks], df[scaled_signal][peaks], color='red', label='Respective Peaks', s=5)
        plt.scatter(df.index[troughs], df[scaled_signal][troughs], color='green', label='Respective Troughs', s=5)
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Peak Detection in Time Series Data (Peaks: {len(peaks)}, Troughs: {len(troughs)})')
        plt.legend()
        plt.show()
    

    return df, peaks, troughs


# =============================================================================
# signal creation
# =============================================================================
def simulate_x_and_y_using_peaks_and_troughs(df, peaks, troughs, y_value_user_defined=None, exponent_factor=3):
    
    # def exponential(x, b):
    #     return 1 - np.power(1 - x, b)
    
    x_points = concatenate((peaks, troughs))
    x_points.sort(kind='mergesort')
    
    # 0 and 1 for trough and peak    
    y_values = np.zeros_like(x_points)
    # Set Y values for even indices (peaks)
    y_values[1::2] = 1
    
    # user defined y peak values
    if y_value_user_defined:
        y_values = np.array(y_value_user_defined)
    
    # Create an array of x values with 1 step increments
    x_all = df.index.to_numpy()
    
    # Interpolate y values between troughs and peaks to form diagonal lines
    y_all = np.interp(x_all, x_points, y_values)
    
    # Apply exponential function for interpolation
    y_all_exponential = np.exp(exponent_factor * y_all)
    # y_all_exponential = exponential(y_all, exponent_factor)
    
    if y_value_user_defined:
        ## Scaling to initial y_values because that should be the way, up to the peak of the signal
        scaling_factor = (np.max(y_values) - y_values[0]) / (np.max(y_all_exponential) - y_all_exponential[0])
        y_all_exponential = (y_all_exponential - y_all_exponential[0]) * scaling_factor + y_values[0]       
    
    else:
        # Scale the exponential_y_all to ensure values are between 0 and 1
        y_all_exponential = (y_all_exponential - y_all_exponential.min()) / (y_all_exponential.max() - y_all_exponential.min())
    
    df.loc[:, 'y_simulated'] = y_all
    df.loc[:, 'y_simulated_exponential'] = y_all_exponential

    return df



def check_corr_between_scaled_and_simulated(df, scaled_signal, simulated_signal='y_simulated_exponential', print_corr=True, show_plot=True):
    # import pdb; pdb.set_trace()
    signal1 = df[scaled_signal].values
    signal2 = df[simulated_signal].values
    
    # Pearson Correlation
    pearson_corr, _ = pearsonr(signal1, signal2)
    
    # Spearman Rank Correlation
    spearman_corr, _ = spearmanr(signal1, signal2)
    
    # Kendall Tau Correlation
    kendall_corr, _ = kendalltau(signal1, signal2)
    
    if print_corr:
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Rank Correlation: {spearman_corr:.4f}")
        print(f"Kendall Tau Correlation: {kendall_corr:.4f}")

    # Cross-Correlation
    cross_corr = np.correlate(signal1, signal2, mode='full')
    
    # Autocorrelation
    auto_corr = np.correlate(signal1, signal1, mode='full')

    x_all = df.index.to_numpy()

    if show_plot:
        plt.figure(figsize=(15, 6))
        plt.plot(x_all, signal1, alpha=0.8)
        plt.plot(x_all, signal2, alpha=0.8)
        plt.title(f'Scaled vs Simulated signal (Pearson: {pearson_corr})', pad=15)

        for i, cp in enumerate(df[df['peaks']].index):
            # plt.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            plt.text(cp, plt.ylim()[1], str(i), va='bottom', ha='center', color='red', fontsize=8, alpha=0.5)
        
        # plt.pause(0.1)
        # plt.waitforbuttonpress()
    
    return pearson_corr, spearman_corr, kendall_corr, cross_corr, auto_corr


def exponential_clustering(df, target_signal_for_clustering, target_signal_for_modelling, out_var_label, num_of_clusters, exponent_degree, add_bp_cluster=False, add_opcp_cluster=False, give_warnings=True, show_plot=True, print_trend=True, print_count=True):
    
    # import pdb; pdb.set_trace()
    
    def exponential(x, b):
        return 1 - np.power(1 - x, b)

    def sigmoid(x, k, x0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))
    
    
    # =============================================================================
    # ## UP DOWN LABEL
    # =============================================================================
    def label_up_and_down_signal(df, target_signal_for_clustering='bladder_pressure_cleaned_scaled', up_threshold=0.0001, down_threshold=-0.0001):
        df['diff'] = df['y_simulated_all'].diff()
            
        # Label trends as "up," "down," or "neutral"
        df['trend'] = 'neutral'

        df.loc[df['diff'] > up_threshold, 'trend'] = 'up'
        df.loc[df['diff'] < down_threshold, 'trend'] = 'down'
        
        if (df['trend'].iloc[0] == 'neutral') and (df['trend'].iloc[1] == 'up'):
            df.at[0, 'trend'] = 'up'
        elif (df['trend'].iloc[0] == 'neutral') and (df['trend'].iloc[1] == 'down'):
            df.at[0, 'trend'] = 'down'
        else:
            df.at[0, 'trend'] = 'neutral'

        df = df.drop(columns=['diff'])
        
        if print_trend:
            print()
            print(df['trend'].value_counts(dropna=False))
        
        return df
    
    def get_labels_using_simulated_signal(df, target_signal_for_clustering, target_signal_for_modelling, out_var_label, num_of_clusters, exponent_degree, add_bp_cluster=add_bp_cluster, add_opcp_cluster=add_opcp_cluster, show_plot=True):
        ## this is where the magic happens - stretching the 'data of interest' to fit 0 - 1 and clustering.
        ## Initially I thought that rescaling would have done the job, but with different height even after scaling (due to some noise or something)
        ## Now I only select the data within trough -> peak, and stretch it
        
        # import pdb; pdb.set_trace()
        ## adding one cluster because we'll combine 0 with 1
        x_values = np.linspace(0, 1, num_of_clusters)
        y_values = exponential(x_values, exponent_degree)
        
        # Add another data point at the end (for going down, last clustering)
        x_values = np.append(x_values, 1)
        y_values = np.append(y_values, 0)

        # # note that last y_value is not in the 'label' because it was artificially added for plotting purposes
        ## np.searchsorted is useful for finding the indices where elements of an array should be inserted to maintain order.
        ## because y_values are not evenly distributed, it may not be such a good idea if I want to take advantage of exponent calculation
        include_for_clustering = (df['troughs_group'].notna() & (~df['first_x_percent']))
        temp_df = df[include_for_clustering]
        temp_df_index = temp_df.index.values
        temp_df = min_max_scale_data_by_group(temp_df.copy(), 'troughs_group', target_signal_for_modelling, f'{target_signal_for_modelling}_temp')
        temp_df[f'{out_var_label}_temp'] = np.searchsorted(y_values[:-1], temp_df[f'{target_signal_for_modelling}_temp']) 
        
        # if looking at inflection point, we still want to capture bp
        ## THIS WILL INCREASE CLUSTER NUMBER!
        target_signal_for_modelling_index = temp_df[~temp_df[target_signal_for_modelling].isna()].index
        bp_index = np.setdiff1d(temp_df.index, target_signal_for_modelling_index)
        temp_df['bp_index'] = np.where(temp_df.index.isin(bp_index), True, False)
        
        # fig = plt.figure(figsize=(16, 6))
        # ax_scatter = fig.add_subplot()
        # # scatter = ax_scatter.scatter(temp_df.index, temp_df[f'{target_signal_for_modelling}_temp'], c=temp_df[f'{out_var_label}_temp'], cmap='tab20', s=5)
        # scatter = ax_scatter.scatter(temp_df.index, temp_df[target_signal_for_clustering], c=temp_df[f'{out_var_label}_temp'], cmap='tab20', s=5)
        # # for cp in target_signal_for_modelling_index:
        # #     plt.axvline(x=cp, color='yellow', linestyle='--', alpha=0.3)
        # for cp in bp_index:
        #     plt.axvline(x=cp, color='pink', linestyle='--', alpha=0.3)            
        # # for cp in temp_df_index:
        # #     plt.axvline(x=cp, color='green', linestyle='--', alpha=0.3)             

        
        # import pdb; pdb.set_trace()
        temp_df.index = temp_df_index
        temp_df = pd.concat([temp_df, df[~df.index.isin(temp_df_index)]], axis=0).sort_index()

        # since first label is really the same as the one next label (e.g. 0 --> 1 ), we'll combine them (e.g. 0 & 1)
        temp_df[f'{out_var_label}_temp'] = np.where(temp_df[f'{out_var_label}_temp'] == 0, 1, temp_df[f'{out_var_label}_temp'])
        
        # adding bp cluster / be group / bp point
        if add_bp_cluster:
            ## Adding 
            def first_index_of_bp_cluster(group):
                indices = group[group['bp_group']].index
                return indices[0] if not indices.empty else None
            
            temp_df['bp_group'] = temp_df['bp_index']
            temp_df['bp_group'] = temp_df['bp_group'].fillna(False)
            temp_df['bp_group_temp'] =  ((temp_df['bp_group'] != temp_df['bp_group'].shift()) & ~temp_df['bp_group']).cumsum()
            temp_df['bp_group_temp'] = np.where(temp_df['bp_group'], temp_df['bp_group'], np.nan)     

            ## adding baseline cluster (0)
            temp_df[f'{out_var_label}_temp'] = np.where(temp_df['bp_group'], 0, temp_df[f'{out_var_label}_temp'])
            
            bp_index_point = temp_df.groupby('troughs_group').apply(first_index_of_bp_cluster)    
            temp_df['bp_point'] = temp_df.index.isin(bp_index_point)
            temp_df['bp_group'] = temp_df['bp_group_temp']
            temp_df = temp_df.drop(columns=['bp_group_temp'])        
        
        
        # import pdb; pdb.set_trace()

        temp_df[f'{out_var_label}_temp'] = np.where(temp_df[f'{out_var_label}_temp'].isna(), num_of_clusters, temp_df[f'{out_var_label}_temp'])
        
        if add_opcp_cluster:
            max_cluster_number = np.max(temp_df[f'{out_var_label}_temp'])
            opcp_cluster = ~temp_df['peaks_group'].isna()
            tp_cluster = ~temp_df['troughs_group'].isna()
            
            tp_cluster_max = temp_df[f'{out_var_label}_temp'][tp_cluster].max()
            opcp_cluster_max = temp_df[f'{out_var_label}_temp'][opcp_cluster].max()
            
            if (tp_cluster_max == opcp_cluster_max) & (tp_cluster_max == max_cluster_number) & (opcp_cluster_max == max_cluster_number):
                temp_df.loc[opcp_cluster, f'{out_var_label}_temp'] = max_cluster_number + 1
        
        
        temp_df[f'{out_var_label}_temp'] = temp_df[f'{out_var_label}_temp'].astype(int)
        temp_df[out_var_label] = temp_df[f'{out_var_label}_temp']
        temp_df = temp_df.drop(columns=[f'{out_var_label}_temp', f'{target_signal_for_modelling}_temp'])
        temp_df = temp_df.drop(columns=['bp_index'])
        
        
        num_of_clusters_revised = temp_df[out_var_label].nunique()
        
        # import pdb; pdb.set_trace()
        df = temp_df.copy()
        if show_plot:
            # import pdb; pdb.set_trace()
            # Create the figure and gridspec
            fig = plt.figure(figsize=(15, 6))
            outer_gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.5)  # 2 rows, 1 column
            inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], width_ratios=[3, 1])
            
            # Add subplots to the first subplot (subgrid)
            ax1 = plt.Subplot(fig, inner_gs[0])
            ax2 = plt.Subplot(fig, inner_gs[1])
            
            # Plot the second subplot (Exp Clustering)
            scatter = ax1.scatter(df.index, df[target_signal_for_clustering], c=df[out_var_label].sort_index(), s=5, label='Clusters', cmap='tab20')
            # Add text labels to the scatter plot
            for i, cp in enumerate(df[df['peaks']].index):
                text_y = ax1.get_ylim()[1] - 0.05 # Get the y-axis limit
                ax1.text(cp, text_y, str(i), va='bottom', ha='center', color='red', fontsize=8, alpha=0.5, clip_on=True)
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Signal used for Clustering')
            ax1.set_title(f'{target_signal_for_modelling} Clustering (Clusters: {num_of_clusters_revised}, Degree: {round(exponent_degree, 1)})')
            
            # handles, labels = scatter.legend_elements(num=num_of_clusters-1)
            handles, labels = scatter.legend_elements()
            legend = ax1.legend(handles, labels, loc="upper right", title="Clusters")
            
            # Plot the first subplot (Exponential Data)
            ax2.plot(x_values, y_values, marker='o')
            ax2.set_xlabel('X (a.u)')
            ax2.set_ylabel('Y (0 to 1 model signal)')
            ax2.set_title('Exponential Cluster Range')
            
            for y in y_values:
                ax2.axhline(y=y, color='gray', linestyle='--', linewidth=0.8)
                           
            for i, num_of_cluster in zip(range(len(x_values) - 1), range(num_of_clusters)):
                x_mid = (x_values[i] + x_values[i + 1]) / 2
                y_mid = (y_values[i] + y_values[i + 1]) / 2
                label = f'{num_of_cluster+1}'
                ax2.annotate(label, (x_mid, y_mid), textcoords="offset points", xytext=(0,-10), ha='left')

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
                        
            # Create the second subplot
            ax3 = plt.Subplot(fig, outer_gs[1])
            
            # Plot the second subplot (up and down trends)
            ax3.plot(df.index.to_numpy(), df[target_signal_for_clustering].values, label='Original Data (scaled)', alpha=0.5)
            ax3.scatter(df.index[df['trend'] == 'up'], df['y_simulated_exponential'][df['trend'] == 'up'], color='green', label='up', s=3, alpha=0.3)
            ax3.scatter(df.index[df['trend'] == 'down'], df[target_signal_for_clustering][df['trend'] == 'down'], color='red', label='down', s=3, alpha=0.3)
            ax3.scatter(df.index[df['trend'] == 'neutral'], df[target_signal_for_clustering][df['trend'] == 'neutral'], color='grey', label='neutral', s=3, alpha=0.3)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Value')
            ax3.set_title("Labeling 'up' and 'down' trends in Time Series")
            ax3.legend()           
            fig.add_subplot(ax3)
            plt.show()
        
        return df
    
    df = label_up_and_down_signal(df, target_signal_for_clustering, up_threshold=0.0001, down_threshold=-0.0001)
    df = get_labels_using_simulated_signal(df, target_signal_for_clustering, target_signal_for_modelling, out_var_label, num_of_clusters, exponent_degree, add_bp_cluster=add_bp_cluster, add_opcp_cluster=add_opcp_cluster, show_plot=show_plot)
    
    counts = df[out_var_label].value_counts(dropna=False).sort_index()
    percentages = df[out_var_label].value_counts(normalize=True).sort_index().mul(100).round(1).astype(str) + '%'
    
    if print_count:    
        print()
        print(pd.concat({'Count': counts, 'Percentage': percentages}, axis=1))   
    
    # import pdb; pdb.set_trace()
    ## Warning so that user can change if they wish
    # Calculate the differences between consecutive elements (we will ignore last one because that is not part of exponent)
    differences = counts[:-1].diff()

    # Find indices where the difference is positive
    increasing_indices = differences[differences > 0].index
    
    if give_warnings:
        # If for some reason there are more clusters in the next value (apart from last), give warning
        for index in increasing_indices:
            value = counts[index]
            print()
            print('#########################################################################################')
            print(f"       Warning: Value at index {index} [{value}] is greater than the previous value.")
            print("       Please consider adjusting the 'num_of_clusters' and/or 'exponent_degree'.")
            print('#########################################################################################')
            print()
        
        # if there is not enough cluster for particular exponent degree, give warning
        if num_of_clusters > len(counts.index):
            print()
            print('#########################################################################################')
            print(f"    Warning: The number of clusters ({num_of_clusters}) is sup-optimal for the exponent degree ({exponent_degree}).")
            print("    Please consider adjusting the 'num_of_clusters' and/or 'exponent_degree'.")
            print('#########################################################################################')
            print()
        
        # if the number of last 'up' cluster is smaller than the total number of peaks, give warning
        # last_label_value = counts[-2:-1]
        # if (y_simulated_exponential == 1).sum() > last_label_value.iloc[0]:
        #     print()
        #     print('#########################################################################################')
        #     print(f"    Warning: The number of last 'up' cluster ('up' cluster: {num_of_clusters-1}), total cluster: {num_of_clusters})\n    is sup-optimal because it is less than the number of peaks found ({last_label_value.iloc[0]} < {(y_simulated == 1).sum()}).")
        #     print("    Please consider adjusting the 'num_of_clusters' and/or 'exponent_degree'.")
        #     print('#########################################################################################')
        #     print()
    
    return df


def mask_data_around_peak(df, peaks, input_signal, in_label, mask_window=[-10, 100]):
    # =============================================================================
    # Currently not used but leaving it here just in case
    # =============================================================================
    # Create a list to store the indices of peaks to be chopped
    peaks_to_chop = []
    
    # Loop through detected peaks and extract data within the window
    for peak_index in peaks:
        start_index = max(0, peak_index - mask_window[0] // 2)
        end_index = min(len(df), peak_index + mask_window[1] // 2) 
        indices_to_chop = list(range(start_index, end_index + 1))
        peaks_to_chop.extend(indices_to_chop)
    
    remaining_peak = np.where(df.index.isin(peaks_to_chop), np.nan, df[in_label])
    
    plt.figure()
    plt.plot(df[input_signal])
    # plt.scatter(df.index, df['scaled_bladder_pressure'],  c=chopped_peak, s=20)
    plt.scatter(df.index, df[input_signal],  c=remaining_peak, s=10)
    return df



def perform_kmeans_clustering(df, input_signal, out_var_label, num_of_clusters, show_plot=True):
    # Initialize k-means clustering
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=0, n_init=10)

    # Fit the model to the data
    kmeans.fit(pd.DataFrame(input_signal))

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Create a DataFrame to hold the results
    df[out_var_label] = cluster_labels
    
    if show_plot:
        # Plot the original data colored by cluster labels
        plt.figure(figsize=(15, 6))
        plt.scatter(df.index, input_signal, c=df[out_var_label], s=5)
        plt.xlabel('Time')
        plt.ylabel('Bladder Pressure (0 to 1 scaled')
        plt.title('K-Means Clustering')
        plt.show()

    return df



# =============================================================================
# ## WRAPPER for best curve fit for getting peak/trough
# =============================================================================
def filter_smooth_peaks_troughs(df, group_var, scaled_signal, scaled_filtered_signal, freq, order, filtertype, var_sf, show_plot, print_peaks):
    df_filtered = filter_to_get_peak_and_trough(df, 
                                                group_var, 
                                                scaled_signal, 
                                                scaled_filtered_signal, 
                                                cutoff_frequency=freq, 
                                                order=order, 
                                                use_lfilter=filtertype, 
                                                show_plot=show_plot)
    
    df_peaks_troughs, peaks, troughs = find_peaks_and_troughs(df_filtered, scaled_signal=scaled_signal, 
                                                               filtered_signal=scaled_filtered_signal, 
                                                               var_sf=var_sf,
                                                               peak_trough_widths=[10, 10], 
                                                               peak_trough_heights=[0.25, -0.6], 
                                                               peak_trough_prominence=[0.1, 0.1], 
                                                               print_peaks=print_peaks, show_plot=show_plot)
    return df_peaks_troughs, peaks, troughs



def get_best_curve_fit(params, df, group_var, scaled_signal, scaled_filtered_signal, var_sf, show_plot=False, print_peaks=False, print_corr=False):
    filtertype, freq, order, factor = params
    df_peaks_troughs, peaks, troughs = filter_smooth_peaks_troughs(df, group_var, scaled_signal, scaled_filtered_signal, freq, order, filtertype, var_sf, show_plot, print_peaks)
    df_peaks_troughs = simulate_x_and_y_using_peaks_and_troughs(df_peaks_troughs, peaks, troughs, exponent_factor=factor)   
    pearson, spearman, kendall, cross_corr, auto_corr = check_corr_between_scaled_and_simulated(df_peaks_troughs, scaled_signal, print_corr=print_corr, show_plot=show_plot)
    return df_peaks_troughs, pearson, spearman, kendall






# =============================================================================
# Get cluster using single point
# =============================================================================

def cluster_from_single_point(df, best_estimate_param, target_signal_for_clustering, target_group_segment, single_inflection_point, model_signal, label_name, add_bp_cluster=True, add_opcp_cluster=True, show_plot=True):
    # import pdb; pdb.set_trace()
    # Initialize 'new_column' as False
    df['signal_mask'] = False
    
    # Convert 'inflection' to a NumPy array
    inflection = df[single_inflection_point].to_numpy()
    
    # grouping for where to 'segment' clustering
    target_group = df[target_group_segment].to_numpy() # df['troughs_group'].to_numpy()
    
    # Use NumPy to efficiently calculate 'new_column'
    sequence_started = False
    for i in range(len(inflection)):
        if inflection[i]:
            sequence_started = True
            df.at[i, 'signal_mask'] = True
        elif np.isnan(target_group[i]):
            sequence_started = False
        elif sequence_started:
            df.at[i, 'signal_mask'] = True
        

    df['Model Signal'] = np.where(df['signal_mask'], df[model_signal], np.nan) # y_simulated_exponential # y_two_lines


    df = exponential_clustering(df.copy(), 
                                  target_signal_for_clustering=target_signal_for_clustering, #TARGET_SIGNAL_CLEANED_RESCALED, 
                                  target_signal_for_modelling='Model Signal',
                                  out_var_label=label_name, 
                                  num_of_clusters=best_estimate_param[0], exponent_degree=best_estimate_param[1], 
                                  add_bp_cluster=add_bp_cluster,
                                  add_opcp_cluster=add_opcp_cluster,
                                  give_warnings=False,
                                  show_plot=True)

    df = df.drop(columns=['Model Signal', 'signal_mask'])
    
    inflection_info = df[df[single_inflection_point]].groupby(single_inflection_point)[label_name].value_counts().sort_index()
    print(f'\n{single_inflection_point} falls under Cluster #{inflection_info.index[0][1]} in {inflection_info[0]} peaks.\n')
    return df



# =============================================================================
# AUTO CLUSTER GIVEN CLUSTER NUMBER
# =============================================================================
def comparison_between_prelabel_and_semiauto_clustering(df, group_var, target_signal, hand_label, auto_label, inflection_point_var, params=(10, 3), drop_cols=['BP'], offset=100, show_plot=True, print_mislabels=True, print_inflection_diff=True):
    
    # import pdb; pdb.set_trace()
    
    def get_op_cp_check(df_heatmap, df_count):
        tp_max_index = df_count[df_count['TP'] == df_count['TP'].max()].index
        op_max_index = df_count[df_count['OP'] == df_count['OP'].max()].index
        cp_max_index = df_count[df_count['CP'] == df_count['CP'].max()].index
    
        # these are for checking when OP and CP are not the last cluster
        # df_heatmap['op_check'] = np.where((df_heatmap[hand_label] == 'OP') & (df_heatmap[auto_label] < tp_max_index[0]), True, False)
        df_heatmap['op_check'] = np.where((df_heatmap[hand_label] == 'OP') & (df_heatmap[auto_label] != op_max_index[0]), True, False)
        df_heatmap['cp_check'] = np.where((df_heatmap[hand_label] == 'CP') & (df_heatmap[auto_label] != cp_max_index[0]), True, False)
        
        op_check = df_heatmap[df_heatmap['op_check']].index
        cp_check = df_heatmap[df_heatmap['cp_check']].index
        return op_check, cp_check, tp_max_index
    
    # Custom function to get the index of the first occurrence of 'label' as 10 within each group
    def get_first_label_index(group_df, label_index):
        index = group_df.index[group_df[auto_label] == label_index].min()
        return index
    
    def get_left_right_indices(auto_label_array, max_tp_val_index_on_first):
        left_indices = np.where(auto_label_array == max_tp_val_index_on_first)[0]
        right_indices = np.where(auto_label_array != max_tp_val_index_on_first)[0]
        return left_indices, right_indices
    
    def create_ranges_array(arr):
        ranges = []
        for i in range(0, len(arr), 2):
            if i + 1 < len(arr):
                start = arr[i]
                end = arr[i + 1]
                values_within_range = np.arange(start, end) # note that I didn't +1 here at the end because we don't want to include the last value
                ranges.append(values_within_range)
        return np.array(ranges, dtype=object)
      
    def concat_if_array_of_arrays(arr):
        if isinstance(arr, np.ndarray) and arr.dtype == np.object:
            return np.concatenate(arr)
        else:
            return arr
               
    def find_closest_matches(arr1, arr2, closest_type='any'):
        result_arr1 = []
        result_arr2 = []
        matched_indices = []

        for val1 in arr1:
            min_distance = float('inf')
            closest_val = None
            closest_idx = None

            for i, val2 in enumerate(arr2):
                if i not in matched_indices:
                    if closest_type == 'after':
                        if val2 >= val1:
                            distance = abs(val1 - val2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_val = val2
                                closest_idx = i
                    elif closest_type == 'before':
                        if val2 <= val1:
                            distance = abs(val1 - val2)
                            if distance < min_distance:
                                min_distance = distance
                                closest_val = val2
                                closest_idx = i
                    elif closest_type == 'any':
                        distance = abs(val1 - val2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_val = val2
                            closest_idx = i

            if closest_val is not None:
                result_arr1.append(val1)
                result_arr2.append(closest_val)
                matched_indices.append(closest_idx)

        return np.array(result_arr1), np.array(result_arr2), np.array(matched_indices)
    
    def get_discontinuity_index(arr, threshold_discontinuity):
        
        # Calculate the difference between adjacent elements
        unknown_discontinuity_differences = np.diff(arr)
        
        # Find the indices where differences exceed the threshold
        discontinuity_indices = np.where(np.abs(unknown_discontinuity_differences) > threshold_discontinuity)[0]
        
        first_discontinuity_indices = np.append(0, discontinuity_indices+1)
 
        unknown_first_discontinuity_indices = arr[first_discontinuity_indices]
        unknown_second_discontinuity_indices = np.append(arr[discontinuity_indices], arr[-1])
        
        return unknown_first_discontinuity_indices, unknown_second_discontinuity_indices

    
    def generate_normalised_bell_curve(data_length):
        # Create a range of x-values based on the data length
        x_values = np.arange(0, data_length)

        # Define the parameters for the normal distribution
        mean = (data_length - 1) / 2  # Set the mean to be at the middle of the data (adjusted for even length)
        std_deviation = data_length / 4  # Adjust the standard deviation as needed

        # Calculate the PDF (bell curve)
        pdf = (1 / (std_deviation * np.sqrt(2 * np.pi))) * np.exp(-(x_values - mean)**2 / (2 * std_deviation**2))

        # Normalize the PDF to have values between 0 and 100
        pdf_normalised = (pdf - min(pdf)) / (max(pdf) - min(pdf)) * 100

        return pdf_normalised
    
    
    
    # Select relevant columns
    df_heatmap = df[[auto_label, hand_label]]
    
    # Pivot table to count occurrences and calculate percentages
    df_count = df_heatmap.pivot_table(index=auto_label, columns=hand_label, aggfunc=len, fill_value=0, sort=False).sort_index()
    column_sums = df_count.sum()
    df_percentage_percentage = df_count / column_sums * 100

    # Create a new DataFrame with combined information
    df_combined = df_count.astype(str) + ' (' + df_percentage_percentage.round(2).astype(str) + '%)'
    df_combined_out = df_combined.copy()
    
    # drop column if want to get rid of some    
    if drop_cols != []:
        for col in drop_cols:
            if col in df_count.columns:
                df_count = df_count.drop(columns=[col])
                df_combined = df_combined.drop(columns=[col])
    
    # Get OP, CP checks and TP max index
    op_check, cp_check, tp_max_index = get_op_cp_check(df_heatmap.copy(), df_count)
    
    # Get first TP indices
    first_tp_indices = df[inflection_point_var]
    first_tp_indices = first_tp_indices[first_tp_indices].index
   
    # Get auto label in hand label for TP indices
    auto_label_in_hand_label = df.iloc[first_tp_indices][auto_label]
    
    # import pdb; pdb.set_trace()
    # fig = plt.figure(figsize=(16, 6))
    # ax_scatter = fig.add_subplot()
    # scatter = ax_scatter.scatter(df.index, df[target_signal], c=df[auto_label], cmap='tab20', s=5)
    
    # # Check if auto_label_in_hand_label is normally distributed
    # normal_dist_val, normal_dist_p = kstest(auto_label_in_hand_label, 'norm')
    
        
    last_label_index = df[auto_label].value_counts().sort_index().index[-1]
    auto_label_without_last_label = df[df[auto_label] != last_label_index][auto_label]

    normal_dist_val, normal_dist_p = kstest(auto_label_without_last_label, 'norm')
    skewness = skew(auto_label_without_last_label)
    kurt = kurtosis(auto_label_without_last_label)
    cv_score = variation(auto_label_without_last_label) # coefficient variation for even distribution score ## lower the better
    
    ## edge (1 or 10 for example) cluster penalty
    auto_label_counts = df[auto_label].value_counts().sort_index()
    edge_scores = generate_normalised_bell_curve(len(auto_label_counts) - 1)
    auto_label_in_hand_index = np.sort(auto_label_in_hand_label.unique())-1
    
    edge_scores_in_hand_index = edge_scores[auto_label_in_hand_index]
    label_counts_in_hand_index = auto_label_counts[auto_label_in_hand_index+1].values
    
    # e.g. [0 , 30] , [2, 5] --> [0,0,  30,30,30,30,30]
    edge_score = np.mean([value for value, count in zip(edge_scores_in_hand_index, label_counts_in_hand_index) for _ in range(count)])
    
    
    
        
    # =============================================================================
    #     ## giving it indicies between a cluster (start / end)
    # =============================================================================
    # import pdb; pdb.set_trace()
    # Calculate indices between clusters (start / end)
    auto_label_array = df[auto_label].values
    first_tp_indices_value_count = auto_label_in_hand_label.value_counts().sort_index()  
    first_max = np.argsort(first_tp_indices_value_count)[::-1].iloc[0]
    max_tp_val_index_on_first = first_tp_indices_value_count.index[first_max]    
    
    # Get left and right indices for the max count cluster
    left_indices_for_max_count, right_indices_for_max_count = get_left_right_indices(auto_label_array, max_tp_val_index_on_first)
    
    # Calculate indices for the start and end of the cluster of interest
    auto_label_indices_first = left_indices_for_max_count[np.isin(left_indices_for_max_count, right_indices_for_max_count + 1)] ## first of max index
    auto_label_indices_second = right_indices_for_max_count[np.isin(right_indices_for_max_count, left_indices_for_max_count + 1)] ## first of next index after max index
     
        
    # If the selected max cluster is the first or last in cluster range
    if len(auto_label_indices_first) != len(auto_label_indices_second):
        if first_max == 0:
            auto_label_indices_first = np.sort(np.append(df.index[0], auto_label_indices_first))
        elif max_tp_val_index_on_first == np.max(auto_label_array):
            auto_label_indices_second = np.sort(np.append(auto_label_indices_second, df.index[-1]))

    # import pdb; pdb.set_trace()
    # Calculate indices for the middle of clusters
    auto_label_indicies_middle = np.add(auto_label_indices_first, auto_label_indices_second) // 2
                
    # =============================================================================
    #     ## we need to remove unknown from auto label so that we can compare directly (but keep it for actual use)
    # =============================================================================
    ### UNKNOWN has to change based on the 'auto-label' because any remaining 'marker' (first / middle / second) should be removed, and it may be outside of the 
    ### pre-set unknown range
    
    unknown_index_all = df[hand_label]=='Unknown'
    unknown_index_all = unknown_index_all[unknown_index_all].index.values
    
    ## knowing peaks/troughs in unknown should help with finding the nearest unknown data
    
    peaks_index = df['peaks']
    peaks_index = peaks_index[peaks_index].index.values
    
    troughs_index = df['troughs']
    troughs_index = troughs_index[troughs_index].index.values
    
    peaks_in_unknown = ((df['peaks']) & (df[hand_label] == 'Unknown'))
    peaks_in_unknown_index = peaks_in_unknown[peaks_in_unknown].index.values   

    ## so we capture from previous trough
    troughs_after_peaks_in_unknown_index = find_closest_matches(peaks_in_unknown_index, troughs_index, closest_type='after')[1]
    troughs_after_peaks_in_unknown_index_and_one_before = troughs_index[find_closest_matches(troughs_after_peaks_in_unknown_index, troughs_index, closest_type='any')[2]-1]
    toughs_for_unknown_index = np.sort(np.unique(np.concatenate((troughs_after_peaks_in_unknown_index, troughs_after_peaks_in_unknown_index_and_one_before)), 0))

    ## hacky way to get indices for different 'blocks' of unknown data
    unknown_first_discontinuity_indices, unknown_second_discontinuity_indices = get_discontinuity_index(unknown_index_all, 5)        
    first_auto_label_in_unknown = find_closest_matches(unknown_first_discontinuity_indices, auto_label_indices_first, closest_type='any')[1]
    second_auto_label_in_unknown = find_closest_matches(unknown_second_discontinuity_indices, auto_label_indices_second, closest_type='any')[1]
    auto_label_in_unknown = np.sort(np.append(first_auto_label_in_unknown, second_auto_label_in_unknown))
    
    troughs_for_unknown_block_index = find_closest_matches(auto_label_in_unknown, toughs_for_unknown_index, closest_type='any')[1]
    
    
    # every 'end' of unknown should not be included since that is the start of the next cluster
    # first_auto_label_in_unknown[1::2] -= 1
    ranges_of_unknown_arrays = create_ranges_array(auto_label_in_unknown) 
    unknown_index_all_new = concat_if_array_of_arrays(ranges_of_unknown_arrays)
    
    ranges_of_toughs_for_unknown_block_arrays = create_ranges_array(troughs_for_unknown_block_index)
    troughs_for_unknown_block_all = concat_if_array_of_arrays(ranges_of_toughs_for_unknown_block_arrays)
    
    ## removing all new unknowns that are not part of a series of troughs
    unknown_index_all_new = np.intersect1d(troughs_for_unknown_block_all, unknown_index_all_new)
    auto_label_in_unknown_new = np.intersect1d(auto_label_indices_first, unknown_index_all_new)    
    
    unknown_index_first, unknown_index_second, _ = find_closest_matches(auto_label_in_unknown_new, auto_label_indices_second, closest_type='after')
    unknown_index_middle = np.add(unknown_index_first, unknown_index_second) // 2
    unknown_index = np.concatenate((unknown_index_first, unknown_index_second, unknown_index_middle))
    
    
    ## removing unknown if prelablled has it
    line_x_auto = df.index[auto_label_indicies_middle].values
    line_y_auto = df[target_signal][auto_label_indicies_middle].values
    
    if len(first_tp_indices) != len(auto_label_indicies_middle):
        auto_indices = np.array([x for x in auto_label_indicies_middle if x not in unknown_index])
        auto_values = df[target_signal][auto_indices].values
    else:
        auto_indices = line_x_auto
        auto_values = line_y_auto
     
    prelabel_indices = df.index[first_tp_indices].values
    prelabel_values = df[target_signal][first_tp_indices].values    
     
    
    # line_y_auto_first = df[target_signal][auto_label_indices_first].values
    # =============================================================================
    #         ### LEAVE THIS FOR CHECKING
    # =============================================================================
    # fig = plt.figure(figsize=(16, 6))
    # ax_scatter = fig.add_subplot()

    # # Plot the scatter points in the second subplot (Lower subplot)
    # scatter = ax_scatter.scatter(df.index, df[target_signal], c=df[auto_label], cmap='tab20', s=5)
    
    # # Create custom legend handles and labels for the scatter points in the second subplot (Lower subplot)
    # # unique_labels = np.unique(df[auto_label])
    # handles_original, labels_original = scatter.legend_elements()  # Specify num as length of unique_labels
    
    # # Combine the original legend handles and labels with the star and X legend handles and labels
    # all_handles = handles_original
    
    # all_labels = labels_original
    
    # # Add the combined legend to the plot for the scatter points
    # legend_scatter = ax_scatter.legend(all_handles, all_labels, loc='upper left', title='Semi-Auto Labels and Markers')  # Use ncol parameter
    # plt.show()
    
    # ymin = df[target_signal].min()
    # ymax = df[target_signal].max()
    
    # # x_auto = df.index[unknown_index_all].values
    # # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='pink', ls='--', lw=1)
    
    # # x_auto = df.index[unknown_index_all_new].values
    # # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='yellow', ls='--', lw=1)  
    
        
    # x_auto = df.index[auto_label_indices_first].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='blue', ls='--', lw=1)

    # x_auto = df.index[auto_label_indices_second].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='red', ls='--', lw=1)
    
    # x_auto = df.index[auto_label_indicies_middle].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='green', ls='--', lw=1)
    
    # x_auto = df.index[unknown_index_first].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='cyan', ls='--', lw=1)
    
    # x_auto = df.index[unknown_index_second].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='magenta', ls='--', lw=1)
    
    # x_auto = df.index[unknown_index_middle].values
    # plt.vlines(x=x_auto, ymin=ymin, ymax=ymax, colors='black', ls='--', lw=1)  
    
    # # plt.pause(0.1)
    # # plt.waitforbuttonpress()
    
    
    
   
    # =============================================================================
    # Sometimes the clustering may not use all the number of clusters defined.
    # This could mean 'not-a-good-fit', so we should skip this param
    # Example would be when param said 5th cluster is great but the 5th cluster is missing from a cycle
    # =============================================================================
    if len(auto_values) == len(prelabel_values):
    
        # # hacky way to get the same length of data
        # if len(line_y_hand) != len(y_auto_without_unknown):
        #     y_auto_without_unknown = y_auto_without_unknown[:len(line_y_hand)]
        
        # import pdb; pdb.set_trace()
            
        inflection_diff_between_prelabel_and_auto = np.subtract(prelabel_values, auto_values)
        inflection_corr_between_prelabel_and_auto_rho, inflection_corr_between_prelabel_and_auto_p = pearsonr(prelabel_values, auto_values)
        
        ## check the latency difference (because this is what matters!)
        latency_diff_between_between_prelabel_and_auto = np.subtract(prelabel_indices, auto_indices)        
    
        if show_plot:
            # import pdb; pdb.set_trace()
            # Create the figure and subplots
            fig = plt.figure(figsize=(16, 8))
            grid = fig.add_gridspec(3, 2, height_ratios=[0.4, 0.4, 0.2])
    
            
            # Define a dictionary to map hand labels to colors
            label_to_color = {'BP': 'cornflowerblue', 'TP': 'firebrick', 'OP': 'lightsalmon', 'CP': 'black', 'Unknown': 'grey'}
            
            # Create the heatmap in the first subplot (Upper subplot)
            ax_heatmap = fig.add_subplot(grid[0, 0])
    
            sns.heatmap(df_count, annot=df_combined, fmt='', cmap='YlGnBu', cbar=False, annot_kws={"size": 10}, ax=ax_heatmap)
            
            # Add axis labels and title for the first subplot (Upper subplot)
            ax_heatmap.set_xlabel('Pre-labelling')
            ax_heatmap.set_ylabel('Exponential Curve Labelling')
            ax_heatmap.set_title('Comparison between Prelabelled and Semi-auto clustering')
            
            # bar plot
            ax_bar = fig.add_subplot(grid[0, 1])
            first_tp_indices_value_count.plot(kind='bar', ax=ax_bar)
            ax_bar.set_xlabel('Label')
            ax_bar.set_ylabel('Count')
            ax_bar.set_title('Count of Labels at inflection points')
    
    
            ax_scatter = fig.add_subplot(grid[1:, :])
    
            # Plot the scatter points in the second subplot (Lower subplot)
            scatter = ax_scatter.scatter(df.index, df[target_signal], c=df[auto_label], cmap='tab20', s=5)
            
            # Create custom legend handles and labels for the scatter points in the second subplot (Lower subplot)
            handles_original, labels_original = scatter.legend_elements()
    
            # Combine the original legend handles and labels with the star and X legend handles and labels
            star_legend_handles = [plt.Line2D([0], [0], marker='*', color='w', markersize=10, markerfacecolor='red', label='OP')]
            x_legend_handles = [plt.Line2D([0], [0], marker='*', color='w', markersize=10, markerfacecolor='blue', label='CP')]
            all_handles = handles_original + star_legend_handles + x_legend_handles
            
            all_labels = labels_original + ['OP mis-label', 'CP mis-label']
            
            # Add the combined legend to the plot for the scatter points
            legend_scatter = ax_scatter.legend(all_handles, all_labels, loc='upper left', title='Semi-Auto Labels and Markers')  # Use ncol parameter
        
            
            # Plot the shadow scatter plot with an offset in the second subplot
            shadow_legend_handles = []
            for label, color in label_to_color.items():
                subset = df[df[hand_label] == label]
                shadow_legend = ax_scatter.scatter(subset.index + offset, subset[target_signal], color=color, s=5, alpha=1, label=f'{label}')
                shadow_legend_handles.append(shadow_legend)
            
    
            # Add line plot for first_tp_indices (hand)
            first_tp_legend_hand = Line2D([], [], color='red', marker='o', linestyle='dashed', markersize=6, label=f'Inflection Point (Pre-labelled: {len(prelabel_values)})')
            ax_scatter.plot(prelabel_indices + offset, prelabel_values, marker='o', color='red', linestyle='dashed', markersize=6)
            shadow_legend_handles.append(first_tp_legend_hand)
            
            # Add line plot for first_tp_indices (auto - middle)
            first_tp_legend_auto = Line2D([], [], color='blue', marker='o', linestyle='dashed', markersize=6, label=f'Inflection Point (Auto: (middle) {len(auto_values)})')
            ax_scatter.plot(line_x_auto, line_y_auto, marker='o', color='blue', linestyle='dashed', markersize=6)
            shadow_legend_handles.append(first_tp_legend_auto)    
            
            
            # Add the shadow legend to the plot
            ax_scatter.legend(handles=shadow_legend_handles, loc='upper right', title='Hand Labels')
            
            # Add the first legend back to the axes
            ax_scatter.add_artist(legend_scatter)
            
            
            for index in op_check:
                ax_scatter.plot(index, df[target_signal][index], marker='*', markersize=10, color='red', markerfacecolor='none', alpha=0.8)
                ax_scatter.annotate('', xy=(index, df[target_signal][index]), textcoords='offset points', xytext=(0,10), ha='center', fontsize=12, color='red', weight='bold')
            
            for index in cp_check:
                ax_scatter.plot(index, df[target_signal][index], marker='*', markersize=10, color='blue', markerfacecolor='none', alpha=0.8)
                ax_scatter.annotate('', xy=(index, df[target_signal][index]), textcoords='offset points', xytext=(0,10), ha='center', fontsize=12, color='blue', weight='bold')
            
            # Add labels to the axes and title for the second subplot (Lower subplot)
            ax_scatter.set_xlabel('Index')
            ax_scatter.set_ylabel('Bladder Pressure')
            ax_scatter.set_title('Superimposed Comparison Plots (Semi-auto vs Pre-label)')
            
            plt.tight_layout()
            plt.show()
        
    else:
        first_tp_indices_value_count = pd.Series(-999)
        inflection_diff_between_prelabel_and_auto = 999
        inflection_corr_between_prelabel_and_auto_rho = -999
        inflection_corr_between_prelabel_and_auto_p = 999
        latency_diff_between_between_prelabel_and_auto = 999
        print(f'\n## Missing cluster, best not to use param: {(params[0], round(params[1], 1))}. ##')
        
    
    # recommended_op_labelling = list(range(tp_max_index[0], df_heatmap[auto_label].max()+1))
    # print()
    # print('##################################################')
    # print(f'  Recommended OP labelling: {recommended_op_labelling}')
    # print('##################################################')
    
    if print_mislabels:
        print()
        print('##################################')
        print(f'  OP (potential) mis-labels: {len(op_check)}')
        print(f'  CP (potential) mis-labels: {len(cp_check)}')
        print('##################################')
        
    if print_inflection_diff:
        print()
        print('## Inflection difference between Pre label and Auto label ##')
        print(f'Median difference Y: {np.median(inflection_diff_between_prelabel_and_auto)}')
        print(f'Mean difference Y: {np.mean(inflection_diff_between_prelabel_and_auto)}')
        print()
        print(f'Median difference X: {np.median(latency_diff_between_between_prelabel_and_auto)}')
        print(f'Mean difference X: {np.mean(latency_diff_between_between_prelabel_and_auto)}')
        print()
    
    return df_combined_out, first_tp_indices_value_count, inflection_diff_between_prelabel_and_auto,\
            inflection_corr_between_prelabel_and_auto_rho, inflection_corr_between_prelabel_and_auto_p,\
            normal_dist_val, normal_dist_p, skewness, kurt, edge_score, cv_score, latency_diff_between_between_prelabel_and_auto









def find_best_estimate_parameters_for_given_inflection_point(df, 
                                                             estimate_param_combinations, 
                                                             offset, 
                                                             target_signal_main, 
                                                             target_signal_for_clustering, 
                                                             target_signal_for_modelling, 
                                                             label_varname, 
                                                             label_prelabel_varname, 
                                                             group_var, 
                                                             inflection_point_var, 
                                                             rank_methods=['dense','average'],
                                                             preferred_cluster=None):
    '''
    This will try and find the best estimate parameters for 'labelled' inflection point.
    Why? In case we need to find the best cluster parameter for inflection point.
    This is more for showing that curve fitting + exponent adjustment 'could' find the inflection point
    '''
    def calculate_weight(input_array):
        max_value = np.max(input_array)
        total_elements = len(input_array)
        weight = 0 if max_value == 0 else total_elements / max_value
        return weight, total_elements

    def get_ranking_weight_based_on_order(stackedrankings):
        ranking_weight = np.linspace(1, 0, stackedrankings.shape[0]+1)
        ranking_weight = (ranking_weight - ranking_weight.min()) / (ranking_weight.max() - ranking_weight.min())
        ranking_weight = ranking_weight / ranking_weight.sum()
        ranking_weight = ranking_weight[:-1].reshape(-1, 1)
        return ranking_weight

    def calculate_measurement_values(df, params, offset, target_signal_for_clustering, target_signal_for_modelling, label_varname, label_prelabel_varname, group_var):
        
        df_temp = exponential_clustering(df.copy(), target_signal_for_clustering, target_signal_for_modelling,
                                          label_varname, 
                                          num_of_clusters=params[0], 
                                          exponent_degree=params[1], 
                                          add_bp_cluster=False,
                                          give_warnings=False,
                                          show_plot=False,
                                          print_trend=False, 
                                          print_count=False)
        
        check_df, first_tp_indices_value_count, inflection_diff, inflection_corr_rho, inflection_corr_p, normal_dist_val, normal_dist_p, skewness, kurt, edge_score, cv_score, latency_diff = comparison_between_prelabel_and_semiauto_clustering(df_temp, group_var, target_signal_main, 
                                                                                                label_prelabel_varname, 
                                                                                                label_varname, 
                                                                                                inflection_point_var,
                                                                                                params,
                                                                                                drop_cols=[], 
                                                                                                offset=offset, 
                                                                                                show_plot=False,
                                                                                                print_mislabels=False,
                                                                                                print_inflection_diff=False)
        # import pdb; pdb.set_trace()
        cluster_number_fullness, element_number = calculate_weight(first_tp_indices_value_count.index)
        target_cluster = first_tp_indices_value_count.sort_index(ascending=False).sort_values(ascending=False).index[0]
        
        calculated_measurement_values = [cluster_number_fullness,
                              element_number,
                              first_tp_indices_value_count.max(), 
                              np.mean(inflection_diff), 
                              np.median(inflection_diff), 
                              inflection_corr_rho,
                              normal_dist_val,
                              skewness,
                              kurt,
                              edge_score,
                              cv_score,
                              np.mean(latency_diff),
                              np.median(latency_diff),
                              target_cluster
                              ]

        return calculated_measurement_values

    def rank_values(all_calculated_values_inflection, rank_method):
        # import pdb; pdb.set_trace()
        (cluster_number_fullnesses, 
          element_numbers, 
          max_counts,
          inflection_diffs_mean,
          inflection_diffs_median,
          inflection_corr_rhos, 
          normal_dist_vals,
          skewness,
          kurt,
          edge_score,
          cv_score,
          latency_diffs_mean,
          latency_diffs_median,
          target_cluster,
          
          ) = zip(*all_calculated_values_inflection)

        cluster_number_fullnesses_ranking = rankdata(cluster_number_fullnesses, method=rank_method)
        element_numbers_ranking = rankdata(element_numbers, method=rank_method)
        max_counts_ranking = rankdata(max_counts, method=rank_method)
        
        ## note absolute value and negative all, since we want the value closest to 0
        inflection_diffs_mean_ranking = rankdata([i*-1 for i in np.abs(inflection_diffs_mean)], method=rank_method)
        
        ## not using median
        inflection_diffs_median_ranking = rankdata([i*-1 for i in np.abs(inflection_diffs_median)], method=rank_method)
        inflection_corr_rhos_ranking = rankdata(inflection_corr_rhos, method=rank_method)
        
        # normal distributeness (smaller the better)
        normal_dist_vals_ranking = rankdata([i*-1 for i in normal_dist_vals], method=rank_method)

        skewness_ranking = rankdata([i*-1 for i in np.abs(skewness)], method=rank_method)
        kurtosis_ranking = rankdata([i*-1 for i in np.abs(kurt)], method=rank_method) # normal dist
        # kurtosis_ranking = rankdata(kurt, method=rank_method) # normal dist -- this is to make it 'peak' but this doesn't seem good.
        
        edge_score_ranking = rankdata(edge_score, method=rank_method)
        cv_score_ranking = rankdata([i*-1 for i in cv_score], method=rank_method) # even dist
        
        ## not using median
        latency_diffs_mean_ranking = rankdata([i*-1 for i in np.abs(latency_diffs_mean)], method=rank_method)
        latency_diffs_median_ranking = rankdata([i*-1 for i in np.abs(latency_diffs_median)], method=rank_method)

        rankings = [
            cluster_number_fullnesses_ranking,    # 0
            element_numbers_ranking,              # 1
            max_counts_ranking,                 # 2 -- probably too biased
            inflection_diffs_mean_ranking,        # 3 
            # inflection_diffs_median_ranking,     # 4 # same corr as median    
            # inflection_corr_rhos_ranking,       # 5 # many overlaps
            # normal_dist_vals_ranking,             # 6
            # skewness_ranking,                     # 7 # same corr as edge_score / inflection_corr_rhos
            kurtosis_ranking,                   # 8 # same corr as edge_score / inflection_corr_rhos
            edge_score_ranking,                   # 9 -- This helps with cluster not falling on endge (e.g. 1 or 10)
            # cv_score_ranking,                     # 10
            latency_diffs_mean_ranking,             # 11
            # latency_diffs_median_ranking,           # 12
        ]
        
        return rankings
    
    # function to map param_index to estimate_param_combinations
    def get_param_combination(index):
        combination = estimate_param_combinations[int(index)]
        formatted_combination = (combination[0], round(combination[1], 1))
        return formatted_combination
    
    
    def get_top_picks(values, rank_method, num_picks=5):
        rankings = rank_values(values, rank_method)
        all_ranking_permutations = list(permutations(rankings))
        
        # import seaborn as sns; plt.figure(); sns.heatmap(pd.DataFrame(rankings).T.corr(), annot=True, cmap='coolwarm')
        
        print(f'\nGetting Top Picks for {rank_method}')
        all_top_picks = []
        
        for perm in tqdm(all_ranking_permutations):
            # Create the stacked rankings for this permutation
            stacked_rankings_permutation = np.vstack(perm)
            ranking_weight = get_ranking_weight_based_on_order(stacked_rankings_permutation)
            weighted_ranking = perm * ranking_weight
            median_ranking = np.median(weighted_ranking, axis=0)
            top_picks = np.argsort(median_ranking)[::-1][:num_picks]
            all_top_picks.append(top_picks)
        
        return np.concatenate(all_top_picks)
    
    
    # import pdb; pdb.set_trace()
    ## calculating all measurement, such as cluster numbers, element number, max counts and such and giving them ordered ranking
    ## then we use all the possible permutations and get the top 5 ranking, and get the most frequent one out of all
    all_calculated_values_inflection = []
    print('Searching for the best parameter..')
    for params in tqdm(estimate_param_combinations):
        calculated_measurement_values = calculate_measurement_values(df, params, offset, target_signal_for_clustering, target_signal_for_modelling, label_varname, label_prelabel_varname, group_var)
        all_calculated_values_inflection.append(calculated_measurement_values)
    
    # import pdb; pdb.set_trace()
    # =============================================================================
    #     ## Correlation matrix to see how related these features are
    # =============================================================================
    # import seaborn as sns; plt.figure(); sns.heatmap(pd.DataFrame(all_calculated_values_inflection).corr(), annot=True, cmap='coolwarm')

    
    # =============================================================================
    #     ## PCA to reduce dimension but can't turn it back, so not sure which is good ## but this can be automated
    # =============================================================================
    
    # data = all_calculated_values_inflection.copy()
    # from sklearn.decomposition import PCA
    
    # mean = np.mean(data, axis=0)
    # std_dev = np.std(data, axis=0)
    # standardized_data = (data - mean) / std_dev

    # # Create a PCA instance
    # pca = PCA()
    
    # # Fit the PCA model to your data
    # pca.fit(standardized_data)
    
    # # Explained variance ratio
    # explained_variance_ratio = pca.explained_variance_ratio_
    
    # # Cumulative explained variance
    # cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # # Determine the number of components to retain (e.g., 95% of variance)
    # n_components = np.argmax(cumulative_variance >= 0.95) + 1
    
    # # Transform the data using the selected number of components
    # data_pca = pca.transform(standardized_data)[:, :n_components]
        
    # import pdb; pdb.set_trace()

    flat_arrays = []    
    for rank_method in rank_methods:
        flat_array = get_top_picks(all_calculated_values_inflection, rank_method)
        flat_arrays.append(flat_array)
    
    flat_arrays = np.concatenate(flat_arrays)
            
    unique_values, counts = np.unique(flat_arrays, return_counts=True) 
    target_clusters = np.array([i[-1] for i in all_calculated_values_inflection])
    df_ranking = pd.DataFrame({'param_index': unique_values, 'counts': counts, 'target_cluster': target_clusters[unique_values]}).sort_values('counts', ascending=False)
    df_ranking['estimate_param'] = df_ranking['param_index'].map(get_param_combination)
    
    if preferred_cluster:
        df_ranking['cluster_ranking'] = np.where(df_ranking['target_cluster'] == int(preferred_cluster), 10, 0)
        df_ranking = df_ranking.sort_values(by=['cluster_ranking', 'counts'], ascending=False)
        df_ranking = df_ranking.drop(columns='cluster_ranking')
    
    most_frequent_value = df_ranking.iloc[0].param_index
    print("\n\nMost frequent parameter index:", most_frequent_value) 

    best_estimate_param = estimate_param_combinations[most_frequent_value]
    print(f'\nBest Parameters:\nNumber of Clusters: {best_estimate_param[0]}\nExponent Degree: {round(best_estimate_param[1], 1)}\nTarget Cluster: #{df_ranking.iloc[0].target_cluster}\n')
    
    return best_estimate_param, df_ranking