# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:59:42 2023

NOTE THAT THIS IS NOT USED BUT THERE FOR YOUR REFERENCE ONLY

@author: WookS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.smoothing_methods import SmoothingMethods
from src.utils.data_loader import load_processed_data
from src.preprocessing.preproc_tools import remove_spikes
from scipy import stats
import matplotlib.pyplot as plt
from src.preprocessing.curve_fit_clustering import min_max_scale_data_by_group
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


### This is not quite a part of analysis flow
### I was trying to see what normal integration might look like
### This can be ignored but leaving it here just in case
### It was a terrible way to do anyway, since we were trying to bend the narrative around James' method that was just not right 
### (not the method itself, the 'bending' part - flipping the signal (again) based on correlation coefficient)


smoother = SmoothingMethods()


def calculate_correlations_and_plot(group, bp_signal, neural_signal, show_plot=True):
    # Initialize lists to store correlations and group names
    correlations = []
    group_names = []


    # Iterate over groups and calculate correlations
    for group_name, grouped in group:
        bladder_pressure = grouped[bp_signal].values
        cross_correlation = grouped[neural_signal].values
        correlation, pval = stats.pearsonr(bladder_pressure, cross_correlation)
        correlations.append(correlation)
        group_names.append(group_name)
        
        if show_plot:
            # Create plots
            fig, ax = plt.subplots()
            ax.scatter(grouped.index, bladder_pressure, color='r', label=f'{bladder_pressure}', s=5, alpha=0.3)
            ax2 = ax.twinx()
            ax2.scatter(grouped.index, cross_correlation,  color='b', label=f'{cross_correlation}', s=5, alpha=0.3)
            ax.annotate(f'Correlation: {correlation:.2f}; P-value: {pval:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
            plt.title(f'Group {group_name}')


    # Create a new DataFrame to store results
    result_df = pd.DataFrame({'Group': group_names, 'Correlation': correlations})

    # Return the DataFrame
    return result_df


def calculate_correlations_and_plot_all_together(df, sig_method_col, bp_col, min_max_scale=False, show_plot=False, void_or_bl='void'):
    # import pdb; pdb.set_trace()
    # Select the relevant columns without NaN values
    df_nonna = df[~df[sig_method_col].isna()]
    if void_or_bl == 'void':
        df_nonna = df_nonna[~df_nonna['optimal_inflection'].isna()]
    
    bp_sig = df_nonna[bp_col].values
    method_sig = df_nonna[sig_method_col].values
    
    # Calculate and print the Pearson correlation coefficient
    correlation, pval = stats.pearsonr(method_sig, bp_sig)
    print(f'\n{sig_method_col} vs {bp_col}')
    print(f"Pearson Corr Coeff: {correlation:.2f}")
    print(f"Pearson Corr P: {pval:.4f}")
    
    df_nonna_temp = min_max_scale_data_by_group(df_nonna.copy(), group_var='troughs_group', 
                                      in_var=bp_col, 
                                      out_var=f'{bp_col}_scaled', 
                                      desired_min=0, desired_max=1)
    
    df_nonna_temp = min_max_scale_data_by_group(df_nonna_temp.copy(), group_var='troughs_group', 
                                      in_var=sig_method_col, 
                                      out_var=f'{sig_method_col}_scaled', 
                                      desired_min=0, desired_max=1)
        
    bp_sig_scaled = df_nonna_temp[f'{bp_col}_scaled'].values
    method_sig_scaled = df_nonna_temp[f'{sig_method_col}_scaled'].values
    
    
    if show_plot:
        if min_max_scale:
            fig, ax = plt.subplots(figsize=(16,6))
            ax.scatter(df_nonna_temp.index, method_sig_scaled, color='r', s=5, alpha=0.5, label='sig_method_col')
            ax2 = ax.twinx()
            ax2.scatter(df_nonna_temp.index, bp_sig_scaled, color='b', s=5, alpha=0.3, label='bp_col')
            plt.title(f'{sig_method_col} vs {bp_col}')


            if void_or_bl == 'void':
                for i, cp in enumerate(df_nonna_temp[df_nonna_temp['optimal_inflection']].index):
                    plt.text(cp, plt.ylim()[1], str(i), va='bottom', ha='center', color='black', fontsize=8, alpha=0.5)
            
        else:

            fig, ax = plt.subplots(figsize=(16,6))
            ax.scatter(df_nonna.index, method_sig, color='r', s=5, alpha=0.5, label='sig_method_col')
            ax2 = ax.twinx()
            ax2.scatter(df_nonna.index, bp_sig, color='b', s=5, alpha=0.3, label='bp_col')
            plt.title(f'{sig_method_col} vs {bp_col}')
            
            if void_or_bl == 'void':
                for i, cp in enumerate(df_nonna[df_nonna['optimal_inflection']].index):
                    plt.text(cp, plt.ylim()[1], str(i), va='bottom', ha='center', color='black', fontsize=8, alpha=0.5)
            
        
        # Combine the legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # Add the correlation value as a text annotation on the plot
        ax.annotate(f'Correlation: {correlation:.2f}; P-value: {pval:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
       
    return correlation, pval
  

def load_and_patch_data(data_drive, processed_dir, processed_folder_name_cc, processed_folder_name_void, processed_folder_name_bp, file_extension, file_exception_wildcard, exclude_xcor_matrix):
    # Load neural data
    df_neural = load_processed_data(data_drive, processed_dir, processed_folder_name_cc, file_extension, file_exception_wildcard, exclude_xcor_matrix)
    all_df_cols = [GROUP_VAR_ALL, BP_SIGNAL, 'time', 'sf']
    df_bp = load_processed_data(data_drive, processed_dir, processed_folder_name_bp, file_extension, file_exception_wildcard, exclude_xcor_matrix)    
    df = pd.merge(df_neural, df_bp, on = all_df_cols, how='inner')
    df_void = df[~df[GROUP_VAR_ALL].str.contains('0.1')].reset_index(drop=True)
    df_bl = df[df[GROUP_VAR_ALL].str.contains('0.1')].reset_index(drop=True)

    return df_void, df_bl, df



def run_james_method(df, 
                     neural_signal, 
                     bp_signal, 
                     void_or_bl, 
                     void_grouping, 
                     file_grouping,
                     clean_signal=True,
                     clean_method='titration_thresholding', 
                     medfilt_size=3, 
                     titration_thresholds=[20, 19, 18, 17, 16, 15, 14, 13],
                     rollingmean_window=30):
    

    if clean_signal:
        # Apply the function to each group
        def remove_spike_by_group(group):
            cleaned_signal = remove_spikes(group.copy(), 
                                           input_signal=neural_signal, 
                                           clean_method=clean_method, 
                                           medfilt_size=medfilt_size,
                                           titration_thresholds=titration_thresholds,
                                           show_plot=False)
            
            # return pd.DataFrame({'cleaned_signal': cleaned_signal})
            return cleaned_signal
        
        df[f'{neural_signal}_cleaned'] = np.concatenate(df.groupby(file_grouping).apply(remove_spike_by_group))
    else:
        df[f'{neural_signal}_cleaned'] = df[neural_signal].copy()
    
    # import pdb; pdb.set_trace()

    plt.figure()
    plt.plot(df[neural_signal], label='original signal')
    neural_signal = f'{neural_signal}_cleaned'
    plt.plot(df[neural_signal], label='cleaned signal')    
    bp_scaled = np.mean(df[neural_signal]) + (df[bp_signal] - np.min(df[bp_signal])) * (np.max(df[neural_signal]) - np.min(df[neural_signal])) / (np.max(df[bp_signal]) - np.min(df[bp_signal]))
    plt.plot(bp_scaled, label='bladder pressure (scaled)')
    plt.legend()

    # =============================================================================
    # signal methods
    # =============================================================================
    cumsum = lambda x: smoother.cumulative_sum(x)
    rollingmean = lambda x: smoother.rolling_mean(x, rollingmean_window)
    
    if void_or_bl == 'void':
        df['cumsum'] = df.groupby(void_grouping)[neural_signal].transform(cumsum)
        df['rollingmean'] = df.groupby(void_grouping)[neural_signal].transform(rollingmean)
        
        troughs_void = df.groupby(void_grouping)
        
        # =============================================================================
        # correlation (by void) -> trough-to-peak
        # =============================================================================
        df_cumsum = calculate_correlations_and_plot(troughs_void, bp_signal, 'cumsum', show_plot=False)
        df_rollingmean_void = calculate_correlations_and_plot(troughs_void, bp_signal, 'rollingmean', show_plot=False)
    
    
        # =============================================================================
        # correlation (by all) -> trough-to-peak 
        # =============================================================================
        cumsum_all = calculate_correlations_and_plot_all_together(df, 'cumsum', bp_signal, show_plot=True, min_max_scale=False, void_or_bl=void_or_bl)
        rollingmean_all = calculate_correlations_and_plot_all_together(df, 'rollingmean', bp_signal, show_plot=True, min_max_scale=False, void_or_bl=void_or_bl)
    
    
    # =============================================================================
    # correlation (by all) -> trough-to-trough
    # =============================================================================
    
    df['rollingmean'] = df.groupby(file_grouping)[neural_signal].transform(rollingmean)
    source_void = df.groupby(file_grouping)
    
    if void_or_bl == 'bl':
        df['cumsum'] = df.groupby(file_grouping)[neural_signal].transform(cumsum)
        df_cumsum = calculate_correlations_and_plot(source_void, bp_signal, 'cumsum', show_plot=False)

    df_rollingmean_source = calculate_correlations_and_plot(source_void, bp_signal, 'rollingmean', show_plot=False)
    df_rollingmean_source_all = calculate_correlations_and_plot_all_together(df, 'rollingmean', bp_signal, show_plot=True, min_max_scale=False, void_or_bl=void_or_bl)
    
    if void_or_bl == 'void':
        return df, df_cumsum, cumsum_all, df_rollingmean_void, rollingmean_all, df_rollingmean_source, df_rollingmean_source_all
    else:
        return df, df_cumsum, df_rollingmean_source, df_rollingmean_source_all



def encode_if_necessary_and_get_data_split(df, X_cols_main, cols_to_encode, y_cols_main,
                                           train_ratio=0.7, test_ratio_of_rest_of_data=0.5, shuffle=False, show_plot=True):
    
    def split_data_by_percentage(df, train_ratio, test_ratio_of_rest_of_data, shuffle, show_plot):
        # Data split
        split = {
            'train': train_ratio, 
            'test': test_ratio_of_rest_of_data,
        }
        
        # Split data into train/test data
        train_df, temp_df = train_test_split(df, test_size=1-split['train'], shuffle=shuffle) 
        val_df, test_df = train_test_split(temp_df, test_size=split['test'], shuffle=shuffle) 
        
        if show_plot:
            # Visualise
            plt.figure(figsize=(16,4))
            plt.scatter(train_df.index, train_df['bladder_pressure'], label="Training Data", s=5, alpha=0.5)
            plt.scatter(val_df.index, val_df['bladder_pressure'], label="Validation Data", s=5, alpha=0.5)
            plt.scatter(test_df.index, test_df['bladder_pressure'], label="Testing Data", s=5, alpha=0.5)
            plt.title("Bladder Pressure Test/Train Split")
            plt.legend()
        return train_df, val_df, test_df

    label_encoder = LabelEncoder()
    
    for cols in cols_to_encode:
        if cols:
            df.loc[:, f'{cols}_encoded'] = label_encoder.fit_transform(df[cols])
    
    X_cols_combined = X_cols_main + cols_to_encode
    X_cols = X_cols_combined if any(isinstance(item, list) for item in X_cols_combined) else X_cols_combined
    X_cols = [i for i in X_cols if i]
    
    y_cols = y_cols_main[0] if any(isinstance(item, list) for item in y_cols_main[0]) else y_cols_main
    
    # actual split
    train_df, val_df, test_df = split_data_by_percentage(df, train_ratio=train_ratio, test_ratio_of_rest_of_data=test_ratio_of_rest_of_data, shuffle=shuffle, show_plot=show_plot)

    
    # =============================================================================
    # ## input data
    # =============================================================================

    X_train = train_df[X_cols].values.reshape(-1, 1) if len(X_cols) == 1 else train_df[X_cols].values
    X_test = test_df[X_cols].values.reshape(-1, 1) if len(X_cols) == 1 else test_df[X_cols].values
    X_val = val_df[X_cols].values.reshape(-1, 1) if len(X_cols) == 1 else val_df[X_cols].values

    y_train = train_df[y_cols].values
    y_test = test_df[y_cols].values
    y_val = val_df[y_cols].values

    return train_df, val_df, test_df, X_cols, y_cols, X_train, X_test, X_val, y_train, y_test, y_val


def evaluate_model(test_val_df, y_test, y_pred, test_df, model_type, print_output=True, show_plot=True):
    # import pdb; pdb.set_trace()
    # Evaluate the SVR model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    if print_output:
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared: {r2}')

    if show_plot:
        plt.figure(figsize=(16,4))
        # plt.plot(test_val_df.index.values, y_test,  label='bladder pressure', alpha=0.7)
        # plt.plot(test_val_df.index.values, y_pred, label='predicted pressure', alpha=1)
        plt.scatter(test_val_df.index, y_test,  label='bladder pressure', s=5, alpha=0.5)
        plt.scatter(test_val_df.index, y_pred, label='predicted pressure', s=5, alpha=0.5)
        plt.title('{}\nMSE: {:.2f}, RMSE: {:.2f}, MAE: {:.2f}, R-squared: {:.2f}'.format(model_type, mse, rmse, mae, r2))
        plt.legend()
    return mse, rmse, mae, r2


def linear_regression(test_val_df, X_train, y_train, X_test, y_test, model_type = 'Linear Regression', print_output=True, show_plot=True):
    # import pdb; pdb.set_trace()

    model = LinearRegression()    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, rmse, mae, r2 = evaluate_model(test_val_df, y_test, y_pred, test_df, model_type, print_output, show_plot)
    return mse, rmse, mae, r2, model, model_type
    

def support_vector_regression(test_val_df, X_train, y_train, X_test, y_test,
                              kernel='rbf', C=1.0, epsilon=0.2, model_type='Support Vector Regression', print_output=True, show_plot=True):
    # Standardize the features (important for SVR)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    X_test_scaled = scaler_x.transform(X_test)
    
    # Create and train the SVR model
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions on the test set
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mse, rmse, mae, r2 = evaluate_model(test_val_df, y_test, y_pred, test_df, model_type, print_output, show_plot)
    return mse, rmse, mae, r2, model, model_type


def xgboost_regression(test_val_df, X_train, y_train, X_test, y_test, objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, model_type='XGBoost Regression', print_output=True, show_plot=True):

    # Create an XGBoost regressor
    model = xgb.XGBRegressor(objective=objective, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    
    # Train the XGBoost model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the XGBoost model
    mse, rmse, mae, r2 = evaluate_model(test_val_df, y_test, y_pred, test_df, model_type, print_output, show_plot)
    return mse, rmse, mae, r2, model, model_type



# =============================================================================
# settings
# =============================================================================

data_drive = 'O'
base_dir = Path(rf'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')
file_exception_wildcard =['12_013.acute.saline.0.4']
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')

processed_folder_name_cc = 'Neural' 
processed_folder_name_void = 'AutoCurate'
processed_folder_name_bp = 'BladderPressure'
file_extension='.parquet'

GROUP_VAR_ALL = 'source'
BP_SIGNAL = 'bladder_pressure' # bladder_pressure
# NEURAL_SIGNAL = 'neural_act' # 'neural_area' neural_act # 'StaticPeak_0.5' # MovingPeak # NonMovingPeak #NonMovingPeakIndividual


exclude_xcor_matrix = True
save_data = False

df_1, df_2, df = load_and_patch_data(data_drive, processed_dir, processed_folder_name_cc, 
                                     processed_folder_name_void, processed_folder_name_bp, 
                                     file_extension, file_exception_wildcard,
                                     exclude_xcor_matrix)

# df = df[df['flip_signal'].notna()]

NEURAL_SIGNALS = [i for i in df.columns if 'neural_' in i]
PEAK_FOUNDS = [i for i in df.columns if 'peak_found' in i]

# input_index = 2
NEURAL_SIGNAL = NEURAL_SIGNALS[2]
PEAK_FOUND = PEAK_FOUNDS[1]

clean_signal = True
clean_method = 'titration_thresholding_then_medfilt' # titration_thresholding // medfilt // titration_thresholding_then_medfilt // medfilt_then_titration_thresholding
medfilt_size = 3
titration_thresholds=[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

# =============================================================================
# James' model (correlation)
# =============================================================================
# void
# =============================================================================

df_void = df_1.copy()
df_bl = df_2.copy()
df_bl['troughs_group'] = df_bl['source']



# df_void[NEURAL_SIGNAL] = df_void.groupby('trough_to_trough')[NEURAL_SIGNAL].transform(median_normalisation)
# df_bl[NEURAL_SIGNAL] = df_bl.groupby('source')[NEURAL_SIGNAL].transform(median_normalisation)

# =============================================================================
# Model
# =============================================================================
df_void_output,\
    df_cumsum_void, cumsum_all_void, \
    df_rollingmean_void, rollingmean_all_void, \
    df_rollingmean_source_void, rollingmean_source_all_void = run_james_method(df_void.copy(), 
                                                                       NEURAL_SIGNAL, 
                                                                       BP_SIGNAL, 
                                                                       void_or_bl='void', 
                                                                        void_grouping='troughs_group', 
                                                                        # void_grouping='trough_to_trough', 
                                                                       file_grouping='source',
                                                                       clean_signal=clean_signal,
                                                                       clean_method=clean_method,
                                                                       medfilt_size=medfilt_size,
                                                                       titration_thresholds=titration_thresholds,
                                                                       rollingmean_window=20
                                                                       )



# df_cumsum_flipped_void // df_rollingmean_void

target_df = df_cumsum_void.copy()
print('\ncumsum')
print('Mean Corr: ', f"{target_df['Correlation'].mean():0.2}")
print('Median Corr: ', f"{target_df['Correlation'].median():0.2}")
print('Max Corr: ', f"{target_df['Correlation'].max():0.2}")
print('Min Corr: ', f"{target_df['Correlation'].min():0.2}")
more_than_06 = (target_df['Correlation']>0.6).sum()
total_number = len(target_df)
print('r2 higher than 0.6: ', f'{more_than_06}/{total_number}', f'{more_than_06/total_number:0.2%}')



target_df = df_rollingmean_void.copy()
print('\nrollingmean')
print('Mean Corr: ', f"{target_df['Correlation'].mean():0.2}")
print('Median Corr: ', f"{target_df['Correlation'].median():0.2}")
print('Max Corr: ', f"{target_df['Correlation'].max():0.2}")
print('Min Corr: ', f"{target_df['Correlation'].min():0.2}")
more_than_06 = (target_df['Correlation']>0.6).sum()
total_number = len(target_df)
print('r2 higher than 0.6: ', f'{more_than_06}/{total_number}', f'{more_than_06/total_number:0.2%}')





# =============================================================================
# ## baseline
# =============================================================================
df_bl_output, \
    df_cumsum_bl, \
    df_rollingmean_source_bl, rollingmean_source_all_bl, \
    df_cumsum_flipped_bl, cumsum_flipped_all_bl = run_james_method(df_bl.copy(), 
                                                                   NEURAL_SIGNAL, 
                                                                   BP_SIGNAL, 
                                                                   void_or_bl='bl', 
                                                                   void_grouping='troughs_group', 
                                                                   file_grouping='source',
                                                                   clean_signal=clean_signal,
                                                                   clean_method=clean_method,
                                                                   medfilt_size=medfilt_size,
                                                                   titration_thresholds=titration_thresholds,
                                                                    )


# =============================================================================
# saving
# =============================================================================

if save_data:
    compiled_dir = processed_dir.joinpath('Compiled')
    compiled_dir.mkdir(parents=True, exist_ok=True)
    
    save_prefix = f"{df_void_output['source'][0].split('.')[1]}.{df_void_output['source'][0].split('.')[2]}"
    df_void_output.to_parquet(compiled_dir.joinpath(f'{save_prefix}.{processed_folder_name_cc}_with_BP_void_compiled.parquet'), index=False)
    df_bl_output.to_parquet(compiled_dir.joinpath(f'{save_prefix}.{processed_folder_name_cc}_with_BP_bl_compiled.parquet'), index=False)




# =============================================================================
# ## using scikit-learn
# =============================================================================

in_var = 'cumsum_flipped' # cumsum_flipped // rollingmean

df_model = df_void_output[df_void_output[in_var].notna()]
cols_to_encode = ['']
X_cols_main = [in_var]
y_cols_main = ['bladder_pressure']



train_df, val_df, test_df, \
    X_cols, y_cols, \
    X_train, X_test, X_val, \
    y_train, y_test, y_val = encode_if_necessary_and_get_data_split(df_model, X_cols_main, cols_to_encode, y_cols_main,
                                                                    train_ratio=0.7, test_ratio_of_rest_of_data=0.5, # test_ratio_of_rest_of_data=0.5, 
                                                                    shuffle=False, show_plot=True)
    





# =============================================================================
# Linear regression
# =============================================================================
mse_lr, rmse_lr, mae_lr, r2_lr, model_lr, model_type_lr = linear_regression(test_df, X_train, y_train, X_test, y_test, model_type = 'Linear Regression', print_output=True, show_plot=True)

# =============================================================================
# SVR
# =============================================================================
mse_svr, rmse_svr, mae_svr, r2_svr, model_svr, model_type_svr = support_vector_regression(test_df, X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, epsilon=0.2, model_type='Support Vector Regression', print_output=True, show_plot=True)


# =============================================================================
# XGboost 
# =============================================================================
mse_xgbr, rmse_xgbr, mae_xgbr, r2_xgbr, model_xgbr, model_type_xgbr = xgboost_regression(test_df, X_train, y_train, X_test, y_test, objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, max_depth=3, model_type='XGBoost Regression', print_output=True, show_plot=True)

