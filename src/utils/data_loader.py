# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:15:44 2023

@author: sungw
"""
import pandas as pd


def load_processed_data(data_drive, base_dir, processed_folder_name, file_extension='.parquet', 
                        file_exception_wildcard=[], exclude_xcor_matrix=False):
    # import pdb; pdb.set_trace()
        
    def remove_xcor_matrix(df):
        float_columns = [col for col in df.columns if (col.replace(".", "", 1).replace("-", "", 1).isdigit())]
        df = df.drop(columns=float_columns)
        return df

    cross_corr_dir = base_dir.joinpath(processed_folder_name)
    load_files = list(cross_corr_dir.glob(f"*{file_extension}"))
    filtered_files = [file for file in load_files if not any(exception in file.stem for exception in file_exception_wildcard)]
    
    all_df = []
    for file in filtered_files:
        if '.csv' in file_extension:
            df = pd.read_csv(file, low_memory=False)
            if exclude_xcor_matrix:
                df = remove_xcor_matrix(df)
        elif '.parquet' in file_extension:
            df = pd.read_parquet(file)
            if exclude_xcor_matrix:
                df = remove_xcor_matrix(df)
        all_df.append(df)
    all_df = pd.concat(all_df, axis=0).reset_index(drop=True)
    
    return all_df


def load_void_info_data(data_drive, base_dir, processed_folder_name, file_name):
    void_info_dir = base_dir.joinpath(processed_folder_name)
    void_info_file_path = void_info_dir.joinpath(file_name)
    voidinfo_df = pd.read_csv(void_info_file_path, low_memory=False)
    return voidinfo_df