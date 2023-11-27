# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:59:32 2023

@author: SChung
"""

"""
Created on Thu Sep  7 20:56:24 2023
@author: sungw
"""
import pandas as pd
from pathlib import Path
import numpy as np
# from scipy import signal
from src.utils.feature_extractor import get_features_from_timeseries
from src.utils.data_loader import load_processed_data
from tqdm import tqdm
# import matplotlib.pyplot as plt
from src.preprocessing.preproc_tools import remove_spikes, median_normalisation


def assign_labels(group):
    # Check if there are NaN values in 'trough_to_trough'
    if group['trough_to_trough'].isna().any():
        unique_source_values = group['source'].unique()
        label_mapping = {source: -(i + 1) for i, source in enumerate(unique_source_values)}
        group['trough_to_trough'] = group['source'].map(label_mapping)
    return group


def get_squeezed_labels(X, window_data, in_var):
    # import pdb; pdb.set_trace()
    bp_label_middle = window_data[in_var].iloc[middle_iloc].values
    bp_label_mean = round(window_data[in_var].mean())
    bp_label_end = window_data[in_var].iloc[-1]

    X[f'bp_{in_var}_middle'] = bp_label_middle
    X[f'bp_{in_var}_mean'] = bp_label_mean
    X[f'bp_{in_var}_end'] = bp_label_end
    return X

# Define a custom sorting key function
def custom_sort_key(val):
    if val >= 0:
        return (0, val)
    else:
        return(abs(val), val)



data_drive = 'O'
base_dir = Path(fr'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')
input_dir = base_dir.joinpath('AutoCurate')
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')
feature_classification_dir = processed_dir.joinpath('Features').joinpath('Classification')
feature_classification_dir.mkdir(parents=True, exist_ok=True)
processed_folder_name_cc = 'Neural'
neural_dir = processed_dir.joinpath(processed_folder_name_cc)
processed_folder_name_bp = 'BladderPressure'

file_exception_wildcard = []
file_extension = '.parquet' 
exclude_xcor_matrix = True

GROUP_VAR_ALL = 'source'
BP_SIGNAL = 'bladder_pressure'


clean_data = False
normalise_windowed_data = False
clean_method = 'titration_thresholding' # titration_thresholding // medfilt // titration_thresholding_then_medfilt // medfilt_then_titration_thresholding
medfilt_size = 3
titration_thresholds = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]


# =============================================================================
# USING JUST ONE FOR NOW!!!!
# =============================================================================
neural_cols = ['neural_act-NonMovingPeakIndividual']

# Specify the overlap as a fraction of the window size (e.g., 50% overlap)
# note that it overlaps more with smaller value (0.2 meaning 80% overlap)
overlap_fraction = 0.0000001  # Adjust this value as needed
window_size_in_sec = 50
folder_suffix= f'{window_size_in_sec}s_win'


# load data
df_neural = load_processed_data(data_drive, processed_dir, processed_folder_name_cc, file_extension, file_exception_wildcard, exclude_xcor_matrix)
df_bp = load_processed_data(data_drive, processed_dir, processed_folder_name_bp, file_extension, file_exception_wildcard, exclude_xcor_matrix)
shared_cols = list(set(df_neural.columns).intersection(set(df_bp.columns)))

df = pd.concat([df_neural, df_bp.drop(columns = shared_cols)], axis=1).reset_index(drop=True)

# =============================================================================
# fill missing trough grouping for bl data
# =============================================================================

unique_sources = df[df['source'].str.contains('0.1')]['source'].unique()
label_mapping = {source: -(i + 1) for i, source in enumerate(unique_sources)}

df['trough_to_trough'] = np.where(df['trough_to_trough'].isna(), df['source'].map(label_mapping), df['trough_to_trough'])

print(df.groupby('source')['trough_to_trough'].apply(lambda x: x.isna().sum()))
print(df.groupby('source')['trough_to_trough'].value_counts(dropna=False))

check = df[df['source'].str.contains('0.1')][['source','trough_to_trough']]

# re-labeling hand label into integer
df['label_hand_int'] = df['label_hand'].replace({'BP': 0, 'TP': 1, 'CP': 2, 'OP': 3, 'Unknown': 99})
neural_cols = [i for i in df.columns if 'neural' in i]
label_cols = [i for i in df.columns if 'label_' in i and i != 'label_hand']




for neural_col in tqdm(neural_cols):

    df_simple = df[[neural_col, BP_SIGNAL, 'sf', 'source', 'trough_to_trough'] + label_cols]

    neural_input_signal = neural_col
    bp_input_signal = BP_SIGNAL

    # =============================================================================
    # Feature extraction
    # =============================================================================

    window_size = window_size_in_sec * round(df_simple['sf']).iloc[0]
    overlap_size = int(window_size * overlap_fraction)
    overlap_size = overlap_size if overlap_size >= 1 else 1
    print(f'\nSliding distance: {overlap_size}s\n')

    X_features = []

    # Process the data with overlapping windows for each group
    for group_name, group_data in tqdm(df_simple.groupby('trough_to_trough')):
        group_data = group_data.reset_index(drop=True).sort_index()  # Sort by time if necessary

        # Calculate the start and end times for overlapping windows
        start_times = np.arange(group_data.index.values.min(), group_data.index.values.max() - window_size, overlap_size)
        end_times = start_times + window_size

        for index, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            window_data = group_data[(group_data.index >= start_time) & (group_data.index <= end_time)]

            if clean_data:
                window_data.loc[:, neural_col] = remove_spikes(df=window_data.copy(), 
                              input_signal=neural_col, 
                              clean_method=clean_method, 
                              medfilt_size=medfilt_size, 
                              titration_thresholds=titration_thresholds,
                              show_plot=False)

            if normalise_windowed_data:
                window_data.loc[:, neural_col] = median_normalisation(window_data[neural_col].copy().values)

            middle_iloc = [len(window_data) // 2]
            file_detail = window_data['source'].iloc[middle_iloc].values

            X, _ = get_features_from_timeseries(x_signal=window_data[neural_input_signal].values, fs=window_data['sf'].iloc[0])        

            # doing across all different labels so we have them        

            for label_col in label_cols:
                X = get_squeezed_labels(X, window_data, label_col)

            X['source'] = file_detail
            X['group'] = group_name
            X['start_index'] = start_time
            X['end_index'] = end_time
            X['index'] = index

            X_features.append(X)


    df_features = pd.concat(X_features)
    df_features = df_features.sort_values(by=['group', 'index'], key=lambda x: x.map(custom_sort_key)).reset_index(drop=True)


    feat_classification_neural_col_dir = feature_classification_dir.joinpath(f'{neural_col}_{folder_suffix}')
    feat_classification_neural_col_dir.mkdir(parents=True, exist_ok=True)


    grouped_df = df_features.groupby('source')   

    # Iterate through the groups and save each group to a separate file
    for group_name, group_data in grouped_df:
        # Define the file name using the group name
        file_name = f"{group_name}_xcor_neural_features.parquet"  # You can change the file format if needed (e.g., .xlsx)

        # Save the group to a file
        group_data.to_parquet(feat_classification_neural_col_dir.joinpath(file_name), index=False)  # Adjust the save format and options as needed


