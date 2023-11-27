# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 06:59:51 2023

@author: sungw
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.feature_extractor import get_features_from_timeseries
from tqdm import tqdm


data_drive = 'O'
base_dir = Path(fr'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')

input_dir = base_dir.joinpath('AutoCurate')
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')
feature_regression_dir = processed_dir.joinpath('Features').joinpath('Regression')
feature_regression_dir.mkdir(parents=True, exist_ok=True)

processed_folder_name_cc = 'Neural' 
neural_dir = processed_dir.joinpath(processed_folder_name_cc)

file_extension = '.parquet' 
load_files = list(neural_dir.glob(f"*{file_extension}"))
file_exception_wildcard = []
exclude_xcor_matrix = False

# Define the range you want to remove
min_range = -0.2
max_range = 0.2

min_range_text = f'{min_range}' if min_range<0 else f'+{min_range}'
max_range_text = f'{max_range}' if max_range<0 else f'+{max_range}'

min_max_text = f'[{min_range_text} {max_range_text}]' 


def run_feature_extraction_from_xcor_matrix(load_files):

    for load_file in tqdm(load_files):
        filename_stem = load_file.stem
        
        df_neural = pd.read_parquet(load_file)
        sf_downsampled = df_neural['sf'].iloc[0]
        
        float_columns = [col for col in df_neural.columns if (col.replace(".", "", 1).replace("-", "", 1).isdigit())]
        xcor_matrix = df_neural[float_columns]
        lag_ms = np.array([np.float64(i) for i in float_columns])
        
        # Create a boolean mask to filter the data within the desired range
        mask = (lag_ms >= min_range) & (lag_ms <= max_range)
        
        # Apply the mask to both the data and labels
        xcor_matrix_filtered = np.array(xcor_matrix)[:, ~mask]
        # lag_ms_filtered = lag_ms[~mask]
        
        # plt.plot(xcor_matrix_filtered[0,:])
        df_xcor_features, correlated_features = get_features_from_timeseries(xcor_matrix_filtered, fs=sf_downsampled)
        
        # df_xcor_features['source'] = load_file.stem
        df_correlated_features = pd.DataFrame({load_file.stem: correlated_features})
        
        df_xcor_features.to_parquet(feature_regression_dir.joinpath(f'{filename_stem}_xcore_features-{min_max_text}.parquet'), index=False)
        df_correlated_features.to_parquet(feature_regression_dir.joinpath(f'{filename_stem}_xcore_features-{min_max_text}-highly_correlated.parquet'), index=False)
        

def main():
    run_feature_extraction_from_xcor_matrix(load_files)


if __name__ == "__main__":
    main() 



    