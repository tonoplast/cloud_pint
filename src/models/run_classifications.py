# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:03:20 2023

@author: WookS
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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_predict
from src.utils.modelling_tools import run_pca_to_remove_correlated_features, select_features, compute_vif, plot_actual_vs_predicted_value, plot_feature_importance, assign_labels_by_percentage, add_void_number_to_classification_data
import json


SEED = 9

data_drive = 'O'
base_dir = Path(rf'{data_drive}:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data')
file_exception_wildcard =['0.1']
processed_dir = base_dir.joinpath('Processed_raw_cleaned_xcor_cleaned')
# processed_dir = base_dir.joinpath('Processed_raw_cleaned')
# processed_dir = base_dir.joinpath('Processed_raw_uncleaned')

processed_folder_name_cc = 'Neural' 
processed_folder_name_void = 'AutoCurate'
processed_folder_name_bp = 'BladderPressure'
processed_folder_name_classification = 'Classification'
file_extension='.parquet'
GROUP_VAR_ALL = 'source'
BP_SIGNAL = 'bladder_pressure' # bladder_pressure

peaktype = 'neural_act-NonMovingPeakIndividual_50s_win'


# NEURAL_SIGNAL = 'neural_act' # 'neural_area' neural_act # 'StaticPeak_0.5' # MovingPeak # NonMovingPeak #NonMovingPeakIndividual
features_dir = processed_dir.joinpath('Features')
automl_classification_input_dir = processed_dir.joinpath('AutoML', 'Classification', 'Input')
automl_classification_input_dir.mkdir(parents=True, exist_ok=True)

exclude_xcor_matrix = True
# save_data = False


# all_df_cols = [GROUP_VAR_ALL, BP_SIGNAL, 'time', 'sf']
file_exception_wildcard=['highly_correlated', '0.1']
file_extension='.parquet'
df_features = load_processed_data(data_drive, features_dir, f'{processed_folder_name_classification}/{peaktype}', 
                                file_extension, file_exception_wildcard, exclude_xcor_matrix)

df_classification = remove_same_valued_columns(df_features.copy())

feature_cols = [i for i in df_classification.columns if '0_' in i]
highly_correlated_features = tsfel.utils.signal_processing.correlated_features(df_classification[feature_cols])
df_classification = df_classification.drop(columns=highly_correlated_features)

ivs = [i for i in df_classification.columns if '0_' in i]
exclude_substrings = ['_mean', '_hand_', 'bp_end', 'middle_inflection', '_middle']
# exclude_substrings = ['_mean', '_hand_', 'bp_middle', 'middle_inflection', '_end']

dvs = [i for i in df_classification.columns if '_label_' in i and all(exclude not in i for exclude in exclude_substrings)]
print(dvs)


leave_out_subject = ["12_225"]
fn_suffix = '225'
## Save data form automl
automl_filename_suffix = f'_leftout_{len(leave_out_subject)}_{fn_suffix}'
# data_split_for_automl = [70, 10, 10, 10] ## train / validate / test / leave out / unassigned (100 - rest)
# data_split_for_automl = [50, 10, 10, 30] ## train / validate / test / leave out / unassigned (100 - rest)
# automl_filename_suffix=f'_{data_split_for_automl[0]}_{data_split_for_automl[1]}_{data_split_for_automl[2]}_{data_split_for_automl[3]}'


# Calculate the percentile cutoff for the top X %
percentile_cutoff = 40
    
plot_sorted_scores = False
plot_top_X_perfecnt_features = False
plot_feature_importance_bool = False

run_pca = False

remove_after_peak = False
do_group_labelling = False
# group_labels_auto = {1: [1,2], 2:[3,4], 3:[5,6], 4:[7,8,9], 5:[10]}
# group_labels_middle_inflection =  {1:[0], 2:[1,2], 3:[3,4], 4:[5,6], 5:[7,8,9], 6:[10]}

group_labels_auto = {1: [1, 2], 3:[3,4], 4:[5,6], 5:[7,8], 6:[9,10]}
group_labels_middle_inflection =  {1:[0], 2:[1,2], 3:[3,4], 4:[5,6], 5:[7,8], 6:[9,10]}
agg = '_agg' if do_group_labelling else ''

# group_labels_auto = {1: [1, 2], 2:[3, 4, 5], 3:[6,7,8,9], 4:[10]}
# group_labels_middle_inflection =  {1:[0], 2:[1,2,3], 3:[4,5,6], 4:[7,8,9], 5:[10]}
# agg = '_agg_more' if do_group_labelling else ''

# group_labels_auto = {1: [1, 2, 6, 7, 8, 9, 10], 2:[3, 4, 5]}
# group_labels_middle_inflection =  {1:[0,4,5,6,7,8,9,10], 2:[1,2,3]}
# agg = '_agg_bi' if do_group_labelling else ''




all_results_list = []

for dv in dvs:
    
    df = df_classification[ivs + [dv]]
        
    if do_group_labelling:
        if 'middle_inflection' in dv:
            df.loc[:,dv] = df[dv].map({value: key for key, values in group_labels_middle_inflection.items() for value in values})
        elif 'auto' in dv:
            df.loc[:,dv] = df[dv].map({value: key for key, values in group_labels_auto.items() for value in values})
    
    if remove_after_peak:
        df = df[~(df[dv] == df[dv].max())].reset_index(drop=True)
    
    X = df[ivs]
    y = df[dv]
    
    # dv_short = dv.split('bp_label_')[1].split('_middle')[0]
    dv_short = dv.split('bp_label_')[1]
    
    filename_to_use = f"{peaktype}{agg}_{dv_short}_automl_{percentile_cutoff}_pc"


    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=SEED)
    
    
    # split into train and test sets       
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression)
    
    # Create a list of feature names and their corresponding scores
    feature_scores = [(X.columns[i], fs.scores_[i]) for i in range(len(fs.scores_))]
    
    # Sort the list by scores in descending order
    sorted_feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
    
    # Print and plot the sorted feature scores
    # for feature, score in sorted_feature_scores:
    #     print('Feature - %s: %f' % (feature, score))
    
    
    if plot_sorted_scores:
        # Plot the sorted scores
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
    
    # Print and plot the top 10% features and their scores
    # for feature, score in top_X_percent_features:
    #     print('Feature - %s: %f' % (feature, score))
    
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
        filename_to_use = f'{filename_to_use}_w_pcafeat'
        df_pca, \
            df_top_features_and_exp_variance, \
                loadings_df = run_pca_to_remove_correlated_features(X, 
                                                                    excluded_columns=[], 
                                                                    variance_threshold=0.95, 
                                                                    top_n_component_per_pc=10, 
                                                                    show_plot=True) 
                    
        
        
        filename_automl_top_features_pca = f'{filename_to_use}_expvar.csv'
        filename_automl_top_features_pca = filename_automl_top_features_pca.lower().replace('-', '_').replace(' ', '_')
        df_top_features_and_exp_variance.to_csv(automl_classification_input_dir.joinpath(filename_automl_top_features_pca), index=False)

        pca_comps = [i for i in df_pca.columns if 'PC' in i]
        
        considered_features = pca_comps.copy()
        check_vif = compute_vif(df_pca, considered_features).sort_values('VIF', ascending=False)

        features_to_keep = list(df_pca.columns)
        
        X = df_pca[features_to_keep]
    
    
    else:
        filename_to_use = f'{filename_to_use}'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=SEED)
    

    
    # json_x_test_string = json.dumps(json_x_test, indent=2)
    
    
    # =============================================================================
    # Encoding because classifiers expect consecutive integers - consider changing otuput in the future
    # =============================================================================
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit the LabelEncoder on the unique class labels in y
    label_encoder.fit(y_train)
    
    # Transform the class labels in y_train and y_test
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    df_classification_automl_with_pca_features = pd.concat([X, y], axis=1).copy()
    df_classification_automl_with_pca_features.columns = [i.replace('-', '_').replace(' ', '_').lower() for i in df_classification_automl_with_pca_features.columns]
    # df_classification_automl_with_pca_features = assign_labels_by_percentage(df_classification_automl_with_pca_features, data_split_for_automl[0], data_split_for_automl[1], data_split_for_automl[2], data_split_for_automl[3])
    
    # left_out_subject = np.random.choice(df_classification.source.unique(), leave_out_subject, replace=False)
    # left_out_subject_bool = df_classification['source'].isin(left_out_subject)
    pattern = '|'.join(leave_out_subject)
    left_out_subject_bool = df_classification['source'].str.contains(pattern, regex=True)
    
    df_classification_automl_with_pca_features_unassigned = df_classification_automl_with_pca_features[left_out_subject_bool]
    df_classification_automl_with_pca_features_unassigned.loc[:, ['source','group']] = df_classification[left_out_subject_bool][['source','group']]
    df_classification_automl_with_pca_features = df_classification_automl_with_pca_features[~left_out_subject_bool]
    
    # leaveout_label = df_classification_automl_with_pca_features['split_label'] == 'LEAVEOUT'
    # df_classification_automl_with_pca_features_unassigned = df_classification_automl_with_pca_features[leaveout_label]
    # df_classification_automl_with_pca_features = df_classification_automl_with_pca_features[~leaveout_label]    
    

    filename_automl_with_pca_features = f'{filename_to_use}{automl_filename_suffix}.csv'
    filename_automl_with_pca_features = filename_automl_with_pca_features.lower().replace('-','_').replace(' ', '_')
    
    filename_automl_with_pca_features_unassigned =  f'{filename_to_use}{automl_filename_suffix}_leftover.csv'
    filename_automl_with_pca_features_unassigned = filename_automl_with_pca_features_unassigned.lower().replace('-','_').replace(' ', '_')
    
    df_classification_automl_with_pca_features.to_csv(automl_classification_input_dir.joinpath(filename_automl_with_pca_features), index=False)
    df_classification_automl_with_pca_features_unassigned.to_csv(automl_classification_input_dir.joinpath(filename_automl_with_pca_features_unassigned), index=False)

    
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
    
    
    
    # for google automl downloaded model prediction
    jsonx_test_data = X_test.to_dict(orient='records')
    json_x_test = {"instances": jsonx_test_data}
    json_save_path = Path(rf"{automl_classification_input_dir}/x_test_{filename_to_use}.json")
    with open(json_save_path, 'w') as json_file:
        json.dump(json_x_test, json_file, indent=2)
      
    
    
    classifiers = {
        "XGBClassifier": XGBClassifier(
                learning_rate=0.1, 
                n_estimators=100,  
                max_depth=5,       
                min_child_weight=1,
                subsample=0.8,     
                colsample_bytree=0.8,
                reg_alpha=0.1,       
                reg_lambda=0.1,      
                random_state=SEED
                ),
        
        # "LogisticRegression": LogisticRegression(
        #       C=1.0,
        #       max_iter=1000,
        #       random_state=SEED
        #   ),
      
        # "RandomForestClassifier": RandomForestClassifier(
        #     n_estimators=100,
        #     max_depth=None,  # You can set a value or None for no limit
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     random_state=SEED
        # ),
       
        # "DecisionTreeClassifier": DecisionTreeClassifier(
        #     max_depth=None,  # You can set a value or None for no limit
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     random_state=SEED
        # ),
       
        # "AdaBoostClassifier": AdaBoostClassifier(
        #     n_estimators=50,
        #     learning_rate=1.0,
        #     random_state=SEED
        # ),
       
        # "GradientBoostingClassifier": GradientBoostingClassifier(
        #     n_estimators=100,
        #     learning_rate=0.1,
        #     max_depth=3,
        #     random_state=SEED
        # ),
   
        # "SVC": SVC(
        #     C=1.0,
        #     kernel='rbf',  # You can try different kernels (e.g., 'linear', 'poly')
        #     probability=True,
        #     random_state=SEED
        # ),
       
        # "KNeighborsClassifier": KNeighborsClassifier(
        #     n_neighbors=5,  # You can experiment with different values of K
        #     weights='uniform',
        # ),
       
        # "GaussianNB": GaussianNB(),
       
        # "MLPClassifier": MLPClassifier(
        #     hidden_layer_sizes=(100, 100),  # You can adjust the number of neurons and layers
        #     max_iter=1000,
        #     early_stopping=True,
        #     validation_fraction=0.2,
        #     random_state=SEED
        # ),
    }
    
    
    results_list = []
    
    for key in classifiers:
    
        print('*',key)
    
        start_time = time.time()
        
        classifier = classifiers[key]
        # model = classifier.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
    
        model = classifier.fit(X_train, y_train_encoded)
        y_pred_encoded = classifier.predict(X_test)
        
        y_pred = label_encoder.inverse_transform(y_pred_encoded) # turning it back to normal coding
        y_test = label_encoder.inverse_transform(y_test_encoded) # turning it back to normal coding
        
        # plot check
        # plot_actual_vs_predicted_value(test_index, y_pred, y_test, f'Actual vs Predicted: {key} - {peaktype}')    
        
        if key in ['XGBClassifier','RandomForestClassifier','DecisionTreeClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier']:
            if plot_feature_importance_bool:
                plot_feature_importance(X, classifier, key)
    
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc_roc = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovr')
        
        # Generate a classification report
        class_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        
            
        # Generate a confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Cross-validation predictions for ROC and AUC
        y_probas = cross_val_predict(classifier, X_train, y_train, cv=10, method="predict_proba")
        
        # Add your code to calculate ROC and AUC here if needed
        
        # Calculate runtime
        runtime = round((time.time() - start_time) / 60, 3)
        
        # Append the results to the list
        result_dict  = {
            'peaktype': peaktype,
            'dv': dv,
            'model': key,
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'auc_roc': round(auc_roc, 3),
        }
        
        result_dict['classification_report'] = class_rep
        results_list.append(result_dict)
    
    
        
        # Visualize ROC Curve and Precision-Recall Curve
        skplt.metrics.plot_roc(y_test, model.predict_proba(X_test))
        plt.title(f'ROC Curve - {key}\n{filename_to_use}')
        plt.show()
        
        skplt.metrics.plot_precision_recall(y_test, model.predict_proba(X_test))
        plt.title(f'Precision-Recall Curve - {key}\n{filename_to_use}')
        plt.show()
    
        # Visualize Confusion Matrix
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
        plt.title(f'Confusion Matrix - {key}\n{filename_to_use}')
        plt.show()
        
        print(f"\nAUC-ROC Score for {key}\n({filename_to_use}): {round(auc_roc, 3)}")
        print(f"\nRuntime for {key}\n({filename_to_use}): {round((time.time() - start_time) / 60, 3)} minutes\n")
    
    # Convert the results list to a DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Bar plot comparing model performance
    df_results.plot(x='model', y=['accuracy', 'precision', 'recall', 'f1_score'], kind='bar')
    plt.ylabel('Score')
    plt.title(f'Model Comparison - {dv_short}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    all_results_list.append(df_results)
    
    
df_results_all = pd.concat(all_results_list)
    
# check = pd.DataFrame(df_results_all['classification_report'][0])
