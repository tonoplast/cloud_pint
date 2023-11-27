# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:48:37 2023

Added a bunch of models. Just wanted to see how they perform.

For RF - parameter tuning
For Polynomial - different degree? (2,3,4 etc)
I think 2 worked the best.

"""

from pathlib import Path
import scipy.stats as stats
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import tsfel
from src.utils.data_loader import load_processed_data
from src.preprocessing.preproc_tools import interpolate_outliers, remove_spikes, median_normalisation, detrend_signal
from src.utils.feature_extractor import remove_same_valued_columns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import VarianceThreshold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils.modelling_tools import run_pca_to_remove_correlated_features, select_features, compute_vif, plot_actual_vs_predicted_value, plot_feature_importance
from src.utils.modelling_tools import logged_box_plot, get_corr, eval_regression_models

from src.utils.smoothing_methods import SmoothingMethods
smoother = SmoothingMethods()



SEED = 9

## you can make this 'True' but for now, I turned it off
## You'll need to have a play and tune it
RF_HYPERPARAM_TUNING = False

# import seaborn as sns; plt.figure(); sns.heatmap(df_features.drop(columns=['source','time']).corr(), cmap='coolwarm')



# =============================================================================
# # loading data, check for nulls and defining dv and ivs
# =============================================================================

data_drive = 'O'
base_dir = Path(rf'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')
file_exception_wildcard =['0.1']
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')
# processed_dir = base_dir.joinpath('Processed_raw_cleaned')
# processed_dir = base_dir.joinpath('Processed_raw_uncleaned')

processed_folder_name_cc = 'Neural' 
processed_folder_name_void = 'AutoCurate'
processed_folder_name_bp = 'BladderPressure'
processed_folder_name_regression = 'Regression'
file_extension='.parquet'

GROUP_VAR_ALL = 'source'
BP_SIGNAL = 'bladder_pressure' # bladder_pressure
# NEURAL_SIGNAL = 'neural_act' # 'neural_area' neural_act # 'StaticPeak_0.5' # MovingPeak # NonMovingPeak #NonMovingPeakIndividual
features_dir = processed_dir.joinpath('Features')
# features_regression_dir = processed_dir.joinpath('Features', 'Regression')
automl_regression_input_dir = processed_dir.joinpath('AutoML', 'Regression', 'Input')
automl_regression_input_dir.mkdir(parents=True, exist_ok=True)

exclude_xcor_matrix = True

# Calculate the percentile cutoff for the top X %
percentile_cutoff = 30
run_pca = False
run_regression = False

plot_sorted_scores = False
plot_ori_and_mod_fignals = True
plot_feature_importance_bool = False
plot_actual_vs_predicted = False
plot_top_X_perfecnt_features = False


df_neural = load_processed_data(data_drive, processed_dir, processed_folder_name_cc, 
                                file_extension, file_exception_wildcard, exclude_xcor_matrix)

df_bp = load_processed_data(data_drive, processed_dir, processed_folder_name_bp, 
                                file_extension, file_exception_wildcard, exclude_xcor_matrix)

common_cols = list(set(df_neural.columns).intersection(set(df_bp.columns)))
df_merged = pd.merge(df_neural, df_bp, how='left', on=common_cols)



# all_df_cols = [GROUP_VAR_ALL, BP_SIGNAL, 'time', 'sf']
file_exception_wildcard=['highly_correlated', '0.1']
file_extension='.parquet'
df_features = load_processed_data(data_drive, features_dir, processed_folder_name_regression, 
                                file_extension, file_exception_wildcard, exclude_xcor_matrix)

file_exception_wildcard=['0.1']
file_extension='highly_correlated.parquet'
df_correlated_features = load_processed_data(data_drive, features_dir, processed_folder_name_regression, 
                                file_extension, file_exception_wildcard, exclude_xcor_matrix)


feature_cols =  [i for i in df_features.columns if '0_' in i]
highly_correlated_features = tsfel.utils.signal_processing.correlated_features(df_features[feature_cols])
df_features = df_features.drop(columns=highly_correlated_features)

# =============================================================================
# Add some smoothings?
# =============================================================================

rolling_window = 30
cutoff_frequency = 0.1
lowcut = 0.1
highcut = 0.8
sf_downsampled=2
filt_order = 3
lowess_frac = 0.3

# cumsum = lambda x: smoother.cumulative_sum(x)
# rollingmean = lambda x: smoother.rolling_mean(x, rolling_window)
# ema = lambda x: smoother.exponential_moving_average(x, alpha=0.3)
# sgf = lambda x: smoother.savitzky_golay_filter(x, window_size=rolling_window, order=3)
# kalman_filter = lambda x: smoother.kalman_filter(x)
# stl_seasonal = lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[0]
# stl_trend: lambda x: smoother.seasonal_decomposition(x, cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)[1]

# chosen_func = stl_seasonal


calculation_functions = {
    'cumsum': lambda x: smoother.cumulative_sum(x),
    # 'ema': lambda x: smoother.exponential_moving_average(x, alpha=0.3),
    'sgf': lambda x: smoother.savitzky_golay_filter(x, window_size=rolling_window, order=2),
    'gaus_filt': lambda x: smoother.gaussian_filt(x, sigma=rolling_window),
    'lowess': lambda x: smoother.lowess_smoothing(x, frac=lowess_frac),
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
    
    'bandpass_filt': lambda x: smoother.bandpass_filt(x, lowcut=lowcut, highcut=highcut, fs=sf_downsampled, order=filt_order, use_lfilter=False),
    
    'lowpass_filt': lambda x: smoother.lowpass_filt(x, cutoff=cutoff_frequency, fs=sf_downsampled, order=filt_order, use_lfilter=False),
    
    
    'mix_bag_1': lambda x: smoother.max_norm(
            smoother.gaussian_filt(x, sigma=rolling_window),
            ),
    
    'mix_bag_2': lambda x: smoother.cumulative_sum(
            smoother.gaussian_filt(x, sigma=rolling_window),
            ),    
    
    "adaptive_filt": lambda x, y, t: smoother.adaptive_filter_predict(x, y, t, filter_order=42, step_size=4, show_plot=False)
    
    # 'mix_bag_1': lambda x: smoother.cumulative_sum(
    #         smoother.smooth_envelope(
    #             smoother.bandpass_filt(x, lowcut=lowcut, highcut=highcut, fs=sf_downsampled, order=filt_order, use_lfilter=False), 
    #             cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)
    #         ),
        

    # 'mix_bag_2': lambda x: smoother.gaussian_filt(
    #     smoother.cumulative_sum(
    #         smoother.smooth_envelope(
    #             smoother.bandpass_filt(x, lowcut=lowcut, highcut=highcut, fs=sf_downsampled, order=filt_order, use_lfilter=False), 
    #             cutoff_frequency=cutoff_frequency, sample_rate=sf_downsampled)
    #         ), 
    #     sigma=rolling_window),

}




# =============================================================================
# Select signals
# =============================================================================

neural_signals = [i for i in df_neural.columns if 'neural' in i]
# neural_signals = ['neural_act-StaticPeak_0.5']
neural_signals = ['neural_act-NonMovingPeakIndividual']

id_cols = ['source', 'time']

dv = 'bladder_pressure'

chosen_func_name = 'gaus_filt'
chosen_func = calculation_functions.get(chosen_func_name)


print(f'\nUsing {chosen_func_name}\n')
    

df_features_same_val_removed = remove_same_valued_columns(df_features.copy())
df = remove_same_valued_columns(df_merged.copy())

# df = df[df['trough_to_trough'].isin([120, 121, 122, 123, 124, 125, 126, 127])]
# df = df[df['trough_to_trough'].isin([99, 100, 101, 102, 103])].reset_index()

df_temp = df.copy()
seed = 42
suffle_data = False

df_models = pd.DataFrame(columns=['signal_type', 'model', 'run_time', 'r-squared', 'mae', 'rmse', 'rmse_cv', 'pearsonr2'])

for neural_signal in neural_signals:
    df = df_temp.copy()
    signal_mod = f'{neural_signal}_{chosen_func_name}_{rolling_window}'
    
    print(f'\nUsing {neural_signal}\n')   
    
    # =============================================================================
    # Clean signal..
    # =============================================================================
        
    clean_signal = True
    clean_method = 'titration_thresholding_then_medfilt' # titration_thresholding // medfilt // titration_thresholding_then_medfilt // medfilt_then_titration_thresholding
    medfilt_size = 3
    titration_thresholds=[20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    file_grouping = 'source'
    
    if clean_signal:
        # Apply the function to each group
        def remove_spike_by_group(group):
            cleaned_signal = remove_spikes(group.copy(), 
                                           input_signal=signal_mod, 
                                           clean_method=clean_method, 
                                           medfilt_size=medfilt_size,
                                           titration_thresholds=titration_thresholds,
                                           show_plot=False)
            
            # return pd.DataFrame({'cleaned_signal': cleaned_signal})
            return cleaned_signal
        
        df[signal_mod] = df[neural_signal].copy()
        df[signal_mod] = np.concatenate(df.groupby(file_grouping).apply(remove_spike_by_group))
    else:
        df[signal_mod] = df[neural_signal].copy()
    
    
    # =============================================================================
    # modifying signal -- detrend + normalisation - this is required for cumulative sum especially
    # =============================================================================
    # Check if there's only one group
    if len(df['source'].unique()) == 1:
        df[signal_mod] = detrend_signal(df.index, df[signal_mod])
    else:
        df[signal_mod] = df.groupby('source').apply(lambda group: detrend_signal(group.index, group[signal_mod])).values
    
    df[signal_mod] = df.groupby('trough_to_trough')[signal_mod].transform(median_normalisation)
    
    ## masking nan of the troughs group, and overwrite the nan with regular signal
    ## This looked pretty terrible, so I think we don't do this.
    # mask = df['troughs_group'].isna()
    # temp_signal_mod = df.groupby('trough_to_trough')[signal_mod].transform(lambda x: chosen_func(x))
    # df[signal_mod] = np.where(mask, df[signal_mod], temp_signal_mod)

    
    if chosen_func_name == 'adaptive_filt':
        df[signal_mod] = chosen_func(df[signal_mod].values, df[dv].values, df.index.values)
    else:
        df[signal_mod] = df.groupby('trough_to_trough')[signal_mod].transform(lambda x: chosen_func(x))


    # =============================================================================
    #     # Shuffle the unique group values
    # =============================================================================
    if suffle_data:
        unique_groups = df['trough_to_trough'].unique()
        np.random.seed(seed)
        np.random.shuffle(unique_groups)
        group_mapping = {group: shuffled_group for group, shuffled_group in zip(df['trough_to_trough'].unique(), unique_groups)}
        df['trough_to_trough'] = df['trough_to_trough'].map(group_mapping)
        df = df.sort_values(by=['trough_to_trough', 'time']).reset_index(drop=True)
    
    
    # =============================================================================
    # removing 'down' signal
    # =============================================================================
    # df = df[df['troughs_group'].notna()]
    # =============================================================================
    
    if plot_ori_and_mod_fignals:
        # Create a plot with separate y-axes for each signal
        fig, ax1 = plt.subplots(figsize=(16, 6))
        ax1.plot(df.index.values, df[neural_signal].values, label='Original X-Corr', color='blue', alpha=0.8)
        # ax1.scatter(df.index, df[neural_signal].values, label='Neural', color='blue', alpha=0.8, s=5)
    
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Amplitudes (a.u.)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(df.index.values, df[signal_mod].values, label=f'Modified X-Corr ({chosen_func_name})', color='red', alpha=0.8)
        # ax2.scatter(df.index, df[signal_mod].values, label='Neural (modified)', color='red', alpha=0.8, s=5)
    
        ax2.set_ylabel('Amplitudes (a.u.)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Adjust position for the third axis
        ax3.plot(df.index.values, df[dv].values, label='Bladder Pressure (rescaled)', color='green', alpha=0.8)
        ax3.set_ylabel('Bladder Pressure', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        # Create separate legends for each axis
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        
        # Combine legends for all axes
        lines = lines1 + lines2 + lines3
        labels = labels1 + labels2 + labels3
        
        # lines = lines1 + lines3
        # labels = labels1 + labels3
        
        ax1.legend(lines, labels, loc='upper right')

                
        # Adjust the plot layout and legend
        plt.title(f'Plots using - {neural_signal}')
        plt.tight_layout()
        plt.show()
        
        # plt.xlim(49_838, 60_000)
        # plt.xlim(54_200, 55_600)
    
    

    
    df_temp2 = df.copy()
    # =============================================================================
    # Split two signal processings
    # =============================================================================
    for sig in [neural_signal, signal_mod]:
        func_name = '' if sig == neural_signal else f'_{chosen_func_name}'
        df = df_temp2.copy()
        
        
        def calculate_pearsonr(group):
            corr_coefficient, _ = stats.pearsonr(group[sig], group[dv])
            return corr_coefficient
            
        correlation_by_group = df.groupby('trough_to_trough').apply(calculate_pearsonr)
        
        print(f'\n{sig}')
        print('Mean Corr: ', f"{correlation_by_group.mean():0.2}")
        print('Median Corr: ', f"{correlation_by_group.median():0.2}")
        print('Max Corr: ', f"{correlation_by_group.max():0.2}")
        print('Min Corr: ', f"{correlation_by_group.min():0.2}")
        more_than_06 = (correlation_by_group>0.6).sum()
        total_number = len(correlation_by_group)
        print('r2 higher than 0.6: ', f'{more_than_06}/{total_number}', f'{more_than_06/total_number:0.2%}')

        
        # =============================================================================
        #     Save for automl
        # =============================================================================
        filename_to_use =  f'{sig}_automl_{percentile_cutoff}_pc'
    
        df_regression_automl = df[[sig, dv]].copy()
        df_regression_automl.columns = [i.replace('-', '_').replace(' ', '_').lower() for i in df_regression_automl.columns]
        filename_automl = f'{filename_to_use}_sfeat.csv'
        filename_automl = filename_automl.lower().replace('-','_').replace(' ', '_')
        df_regression_automl.to_csv(automl_regression_input_dir.joinpath(filename_automl), index=False)

        # =============================================================================
        # data split
        # =============================================================================
        ivs = [sig] + list(df_features_same_val_removed.columns)
        df = pd.concat([df_neural[id_cols], df[[dv, sig]],  df_features_same_val_removed], axis=1)
    
        X = df[ivs]
        y = df[dv]
        X_source_time = df[id_cols]
        
        
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=SEED)
        X_train_source_time, X_test_source_time, _, _ = train_test_split(X_source_time, y, test_size=0.3, shuffle=False, random_state=SEED)
        
        # # Remove low variance features
        # selector = VarianceThreshold()
        # X_train = selector.fit_transform(X_train)
        # X_test = selector.transform(X_test)
                       
        
        if len(ivs) > 1:
        
            # =============================================================================
            # Feature selection - using mutual info regression
            # =============================================================================
            
            X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression)
            
            # Create a list of feature names and their corresponding scores
            feature_scores = [(X.columns[i], fs.scores_[i]) for i in range(len(fs.scores_))]
            
            # Sort the list by scores in descending order
            sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
            
            # Print and plot the sorted feature scores
            # for feature, score in sorted_feature_scores:
            #     print('Feature - %s: %f' % (feature, score))
            
            # Plot the sorted scores
            if plot_sorted_scores:
                sorted_features, sorted_scores = zip(*sorted_feature_scores)
                plt.figure(figsize=(7, 6))
                plt.bar(sorted_features, sorted_scores)
                plt.xticks(rotation=90)
                plt.title('Feature selection scores (sorted)')
                plt.tight_layout()
                plt.show()
            
            
            # =============================================================================
            # top X % features
            # =============================================================================

            
            # Calculate the number of features to keep (10% of total features)
            num_features_to_keep = int(len(sorted_feature_scores) * (percentile_cutoff / 100))
            
            # Get the top X % features
            top_X_percent_features = sorted_feature_scores[:num_features_to_keep]
            
            
            # Plot the top 10% features and their scores
            top_X_percent_features, top_X_percent_scores = zip(*top_X_percent_features)
            
            if plot_top_X_perfecnt_features:
                plt.figure(figsize=(7, 6))
                plt.bar(top_X_percent_features, top_X_percent_scores)
                plt.xticks(rotation=90)
                plt.title(f'Top {percentile_cutoff}% of Feature selection scores')
                plt.tight_layout()
                plt.show()
                            
            X = X[list(top_X_percent_features)]
            
            
            if run_pca:
            
                # =============================================================================
                # ## PCA    
                # =============================================================================
                filename_to_use = f'{filename_to_use}_w_pcafeat'
                
                df_pca, \
                    df_top_features_and_exp_variance, \
                        loadings_df = run_pca_to_remove_correlated_features(X, 
                                                                            excluded_columns=[], 
                                                                            variance_threshold=0.95, 
                                                                            top_n_component_per_pc=10, 
                                                                            show_plot=False) 
                
                ## saving so that we know what PCA consists of
                filename_automl_top_features_pca = f'{filename_to_use}_expvar.csv'
                filename_automl_top_features_pca = filename_automl_top_features_pca.lower().replace('-', '_').replace(' ', '_')
                df_top_features_and_exp_variance.to_csv(automl_regression_input_dir.joinpath(filename_automl_top_features_pca), index=False)
                
                
                pca_comps = [i for i in df_pca.columns if 'PC' in i]
    
                            
                # =============================================================================
                # Simple linear regression (visualisation purposes)
                # =============================================================================
                ## can be useful for understanding the shape of the data
                ## usually with many features, all features are unlikely to have linear relationship
                
                # get_corr(df, dv, ivs)
                
                # =============================================================================
                # Check for (multi)collinearity
                # =============================================================================
    
                # seems acceptable..
                considered_features = pca_comps.copy()
                check_vif = compute_vif(df_pca, considered_features).sort_values('VIF', ascending=False)
                features_to_keep = list(df_pca.columns)
                
    
                # =============================================================================
                # Split data - training / test
                # =============================================================================
                
                X = df_pca[features_to_keep]
                # X = pd.concat([df_neural[ivs[0]], df_pca[features_to_keep]], axis=1)
                
            else:
                filename_to_use = f'{filename_to_use}'
                
          
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=SEED)
            
            df_regression_automl_signal_with_features = pd.concat([X, y], axis=1).copy()
            df_regression_automl_signal_with_features.columns = [i.replace('-', '_').replace(' ', '_').lower() for i in df_regression_automl_signal_with_features.columns]
            filename_automl_with_features = f'{filename_to_use}.csv'
            filename_automl_with_features = filename_automl_with_features.lower().replace('-','_').replace(' ', '_')
            df_regression_automl_signal_with_features.to_csv(automl_regression_input_dir.joinpath(filename_automl_with_features), index=False)

        



        if run_regression:
            train_index = X_train.index.values
            test_index = X_test.index.values
            
            # =============================================================================
            # Scaling - mainly because we are comparing different models below
            # but also, the range are quite different. I think for RF, it may not be
            # necessary, but I like scaling. Will leave this up to you.
            # =============================================================================
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            
            # =============================================================================
            # Model selection
            # =============================================================================
            
            # polynomial is a bit special and need to pipe it through first
            # degree=2
            # polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
            polyreg_degree_2 = make_pipeline(PolynomialFeatures(2),LinearRegression())
            polyreg_degree_3 = make_pipeline(PolynomialFeatures(3),LinearRegression())
            polyreg_degree_4 = make_pipeline(PolynomialFeatures(4),LinearRegression())
            
            
            # you can 'uncomment' these and test it out. They run pretty quickly.
            
            regressors = {
                "XGBRegressor":  XGBRegressor(
                                                learning_rate=0.1,          # Adjust as needed
                                                n_estimators=100,          # Adjust as needed
                                                max_depth=5,               # Adjust as needed
                                                min_child_weight=1,        # Adjust as needed
                                                subsample=0.8,             # Adjust as needed
                                                colsample_bytree=0.8,      # Adjust as needed
                                                reg_alpha=0.1,             # Adjust as needed
                                                reg_lambda=0.1,            # Adjust as needed
                                                random_state=SEED            # Set a random seed for reproducibility
                                            ),
                # "RandomForestRegressor": RandomForestRegressor(),
                # "DecisionTreeRegressor": DecisionTreeRegressor(),
                # "LinearRegression": LinearRegression(),
                # "RANSAC": RANSACRegressor(),
                # "LASSO": Lasso(),
                # "Polynomial": polyreg,
                
                # "GaussianProcessRegressor": GaussianProcesssRegressor(), ## somehow it keeps crashing
                # "SVR": SVR(),
                # "NuSVR": NuSVR(),
                # "LinearSVR": LinearSVR(),
                # "KernelRidge": KernelRidge(),
                # "Ridge":Ridge(),
                # "TheilSenRegressor": TheilSenRegressor(),
                # "HuberRegressor": HuberRegressor(),
                # "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
                # "ARDRegression": ARDRegression(),
                # "BayesianRidge": BayesianRidge(),
                # "ElasticNet": ElasticNet(),
            
                # "Polynomial_2": polyreg_degree_2,
                # "Polynomial_3": polyreg_degree_3,
                # "Polynomial_4": polyreg_degree_4,
            }
            
            
            # =============================================================================
            # loop and ranking based on rmse_cv (10-fold cross-validation)
            # =============================================================================
            
            
            for key in regressors:
            
                print('*',key)
            
                start_time = time.time()
                
                regressor = regressors[key]
                model = regressor.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # plot check
                if plot_actual_vs_predicted:
                    plot_actual_vs_predicted_value(test_index, y_pred, y_test.values, f'Actual vs Predicted: {key} - {sig}')    
                
                if key in ['XGBRegressor','RandomForestRegressor','DecisionTreeRegressor']:
                    if plot_feature_importance_bool:
        
                        plot_feature_importance(X, regressor, key)
                        # importance_model = pd.Series(regressor.feature_importances_, index=X.columns).sort_values()
                
                scores = cross_val_score(model, 
                                         X_train, 
                                         y_train,
                                         scoring="neg_mean_squared_error", 
                                         cv=10)
                
                _r2_score, _mae, _mse, _rmse, _pearsonr2 = eval_regression_models(model, y_test, y_pred)
                        
                row = {'signal_type': sig,
                       'model': key,
                       'run_time': format(round((time.time() - start_time)/60, 3)),
                       'r-squared': round(_r2_score, 3),
                       'mae': round(_mae, 3),
                       'mse': round(_mse, 3),
                       'rmse': round(_rmse, 3),
                       'rmse_cv': round(np.mean(np.sqrt(-scores)), 3),
                       'pearsonr2': round(_pearsonr2[0], 3)
                }
                
                df_dict = pd.DataFrame([row])
                df_models = pd.concat([df_models, df_dict], ignore_index=True)
            
            
            # see scoring
            print(f'{sig}')
            print(df_models.sort_values(by='rmse_cv', ascending=True))
            
            
            # =============================================================================
            # plot (top performing one)
            # =============================================================================
            # regressor = XGBRegressor(random_state=SEED)
            # model = regressor.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            
            # eval_models(model, y_test, y_pred, True)
            # plot_actual_vs_predicted_value(test_index, y_pred, y_test, 'Actual vs Predicted: XGBoostRegressor')    
            # plot_feature_importance(regressor, 'XGBoost')
            
            
            if RF_HYPERPARAM_TUNING:
                # =============================================================================
                # Hyperparameter tuning -- THIS TAKES VERY LONG
                # Try out whatever you think is best
                # =============================================================================
                model.get_params()
                
                def eval_best_rf_model(best_model, X_test, y_test):
                    y_pred = best_model.predict(X_test)
                    eval_regression_models(best_model, y_test, y_pred, True)
                    plot_actual_vs_predicted_value(test_index, y_pred, y_test, 'Actual vs Predicted: RandomForestRegressor')    
                    plot_feature_importance(best_model.best_estimator_, 'RandomForest (Hyperparameter-tuned)')
                    
                   
                    predicted = best_model.best_estimator_.predict(X_train)
                    residuals = y_train-predicted
                    
                    fig, ax = plt.subplots()
                    ax.scatter(y_train, residuals)
                    ax.axhline(lw=2,color='black')
                    ax.set_xlabel('Observed')
                    ax.set_ylabel('Residual')
                    plt.show()
                    
                    df_gs = pd.DataFrame(data=best_model.cv_results_)
                    df_gs_plot = df_gs[['mean_test_score',
                                        'param_max_leaf_nodes',
                                        'param_max_depth']].sort_values(by=['param_max_depth',
                                                                            'param_max_leaf_nodes'])
                    
                    
                    
                    fig,ax = plt.subplots()
                    sns.pointplot(data=df_gs_plot,
                                  y='mean_test_score',
                                  x='param_max_depth',
                                  hue='param_max_leaf_nodes',
                                  ax=ax)
                    ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
                
                
                # =============================================================================
                #     # random search -- quick
                # =============================================================================
                random_grid  = dict(
                                bootstrap = [True, False],
                                criterion = ['squared_error'],
                                max_depth = [None] + [int(x) for x in np.linspace(10, 100, num = 10)],
                                min_samples_leaf = [1, 2, 3, 4],
                                min_samples_split = [2, 4, 6, 8, 10],
                                max_leaf_nodes = [None, 2, 4, 8, 10, 12],
                                max_features = [None, 'sqrt', 'log2'], 
                                n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 50)],
                                )  
                
                rf = RandomForestRegressor(random_state=SEED)
                rf_random = RandomizedSearchCV(estimator = rf, 
                                               param_distributions = random_grid, 
                                               n_iter = 200, 
                                               cv = 3, 
                                               verbose = 2, 
                                               random_state = SEED, 
                                               n_jobs = -1)
                
                best_model_rs = rf_random.fit(X_train, y_train)   
                print('Optimum parameters:')
                pprint(best_model_rs.best_params_)
                
                eval_best_rf_model(best_model_rs, X_test, y_test)
            
            
                # # =============================================================================
                # #     # grid search -- takes long
                # # =============================================================================
                # param_grid = dict(
                #                 criterion=['squared_error'],
                #                 max_depth=[None, 3, 5, 7, 9], 
                #                 min_samples_leaf=np.linspace(0.1, 0.5, 5, endpoint=True),
                #                 min_samples_split=[0.01, 0.03, 0.05, 0.1, 0.2],
                #                 max_leaf_nodes=[None, 2, 4, 8, 10, 12],
                #                 # max_features=list(range(1, X.shape[1])), 
                #                 max_features=[None, 'sqrt', 'log2'], 
                #                 # n_estimators=[1, 2, 4, 6, 16, 32, 64, 100, 120, 140, 160],
                #                 n_estimators=[20, 50, 75, 100, 200],
                
                #                 )
            
                # model = RandomForestRegressor(random_state=SEED)
                
                # grid_search = GridSearchCV(estimator = model,
                #                            param_grid = param_grid,
                #                            scoring = 'neg_root_mean_squared_error',
                #                            n_jobs = -1,
                #                            verbose = 2,
                #                            cv = 3
                #                            )
                
                
                # best_model_gs = grid_search.fit(X_train, y_train)
                # print('Optimum parameters:', best_model_gs.best_params_)
                
             
                # # Optimum parameters: {'criterion': 'squared_error', 
                # #                      'max_features': 'sqrt', 
                # #                      'max_leaf_nodes': 8, 
                # #                      'min_samples_leaf': 0.1, 
                # #                      'min_samples_split': 0.01, 
                # #                      'n_estimators': 50}
                
                # eval_best_rf_model(best_model_gs, X_test, y_test)
        

    