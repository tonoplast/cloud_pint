# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 11:44:30 2023

@author: sungw
"""
from pathlib import Path
import scipy.stats as stats
import time
import math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
from src.utils.data_loader import load_processed_data
from src.preprocessing.preproc_tools import interpolate_outliers, remove_spikes, median_normalisation, detrend_signal
from src.utils.feature_extractor import remove_same_valued_columns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

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


def logged_box_plot(df, my_title):
    fig, ax = plt.subplots(figsize=(12, 6))
    g = sns.boxplot(data=df,linewidth=2.5,ax=ax)
    g.set_title(my_title)
    return g.set_yscale("log")
    

def get_corr(df, x_col, y_cols):
    '''
    Simple linear regression with r2 and p-value
    '''
    for y_col in y_cols:
        x = df[x_col]
        y = df[y_col]
          
        g = sns.jointplot(x=x, y=y, kind='reg', color='royalblue')
        r, p = stats.pearsonr(x, y)
        print(f"{x_col} vs {y_col}: The Rho is {'{:.3}'.format(r)} and p={'{:.3}'.format(p)}")
    
        g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                            xy=(0.1, 0.9), xycoords='axes fraction',
                            ha='left', va='center',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
        g.ax_joint.scatter(x, y)
        g.set_axis_labels(xlabel=x_col, ylabel=y_col, size=15)
        plt.tight_layout()


def eval_regression_models(model, y_test, y_pred, print_scores=False):
    '''
    root mean squared error (how much model's predictions differ from the actual labels)
    '''
    r2_score = metrics.r2_score(y_test, y_pred)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    pearsonr2 = stats.pearsonr(y_test, y_pred)
    
    if print_scores:
        print("R-squared: {:.3f}".format(r2_score))
        print("Mean Absolute Error: {:.3f}".format(MAE))
        print("Mean Squared Error: {:.3f}".format(MSE))
        print("Root Mean Squared Error: {:.3f}".format(RMSE))
        print("Pearson R2: {:.3f}".format(pearsonr2))
    return r2_score, MAE, MSE, RMSE, pearsonr2

def eval_classification_models(model, y_test, y_pred, print_scores=False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro') 
    f1 = f1_score(y_test, y_pred, average='micro') 
    
    if print_scores:
        print("Accuracy: {:.3f}".format(accuracy))
        print("Precision: {:.3f}".format(precision))
        print("Recall: {:.3f}".format(recall))
        print("F1: {:.3f}".format(f1))
    
    return accuracy, precision, recall, f1



def compute_vif(df, considered_features):
    '''
    compute the variance inflation factor (vif) for all given features
    (metric for gauging multicollinearity)
    The VIF directly measures the ratio of the variance of the entire model 
    to the variance of a model with only the feature in question.
    (how much a feature’s inclusion contributes to the overall variance of 
     the coefficients of the features in the model.)
    
    A VIF of 1 indicates that the feature has no correlation with any of the other features.
    A VIF value exceeding 5 or 10 is deemed to be too high. 
    Any feature with such VIF values is likely to be contributing to multicollinearity.

    '''
    X = df[considered_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

 
def select_features(X_train, y_train, X_test, score_func, k='all'):
   	# configure to select all features
   	fs = SelectKBest(score_func=score_func, k=k)
   	# learn relationship from training data
   	fs.fit(X_train, y_train)
   	# transform train input data
   	X_train_fs = fs.transform(X_train)
   	# transform test input data
   	X_test_fs = fs.transform(X_test)
   	return X_train_fs, X_test_fs, fs


def plot_actual_vs_predicted_value(index, y_pred, y_test, my_title):
    # test = pd.DataFrame({'Predicted value':y_pred, 'Actual value':y_test})
    plt.figure(figsize=(12,6))
    # test = test.reset_index(drop=True)
    # test = test.drop(['No'],axis=1)
    # plt.plot(test)
    plt.plot(index, y_test, alpha=1)
    plt.plot(index, y_pred, alpha=0.6)
    plt.title(my_title)

    
    plt.legend(['Actual value','Predicted value'])
    
    
def plot_feature_importance(X, regressor, regressor_key):
    plt.figure(figsize=(12,6))
    importance_model = pd.Series(regressor.feature_importances_, index=X.columns).sort_values()
    importance_model.plot(kind='barh', color='cornflowerblue')
    plt.title(f'Feature Importance: {regressor_key}')
    
    
def remove_common_features(df_features, df_correlated_features):
    # Find common values across all columns
    common_features = set(df_correlated_features[df_correlated_features.columns[0]])
    for col in df_correlated_features.columns[1:]:
        common_features = common_features.intersection(df_correlated_features[col])

    keep_features = df_features.columns[~df_features.columns.isin(common_features)]
    df_features = df_features[keep_features]
    return df_features


def run_pca_to_remove_correlated_features(df_features, excluded_columns=['source','time'], variance_threshold=0.95, top_n_component_per_pc=5, show_plot=True):
        
    # =============================================================================
    #     ## PCA - to reduce features further if .. needed
    # =============================================================================
    # import pdb; pdb.set_trace()
    data = df_features.drop(columns=excluded_columns).copy() if excluded_columns != [] else df_features.copy()
    
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardised_data = (data - mean) / std_dev
    
    # Create a PCA instance
    pca = PCA()
    
    # Fit the PCA model to your data
    pca.fit(standardised_data)
    
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Determine the number of components to retain (e.g., 95% of variance)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Transform the data using the selected number of components
    data_pca = pca.transform(standardised_data)[:, :n_components]
    df_pca_selected = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df_pca_selected = pd.concat([df_pca_selected, df_features[excluded_columns]], axis=1) if excluded_columns != [] else df_pca_selected

    
    # Access the loadings for each PC
    loadings = pca.components_
    
    # Create a DataFrame to display the loadings
    loadings_df = pd.DataFrame(loadings, columns=data.columns)
    
    # Get the top variables and variance explained for each PC
    top_variables_per_pc = {}
    explained_variance_per_pc = {}
    for i in range(n_components):
        # Sort the variables based on absolute loadings for the current PC
        sorted_variables = loadings_df.iloc[i].abs().sort_values(ascending=False)
        
        # Select the top variables
        top_n_variables = sorted_variables.head(n=top_n_component_per_pc)  # Adjust the number as needed
        
        # Calculate the variance explained by the current PC
        explained_variance = pca.explained_variance_ratio_[i]
        
        # Store the top variables and variance explained in dictionaries
        top_variables_per_pc[f'PC{i+1}'] = top_n_variables.index.tolist()
        explained_variance_per_pc[f'PC{i+1}'] = explained_variance
    
    # Convert dictionaries to DataFrames
    number_of_values = int(sum(len(value) for value in top_variables_per_pc.values())/len(top_variables_per_pc))
    if number_of_values < top_n_component_per_pc:
        top_variables_df =  pd.DataFrame.from_dict(top_variables_per_pc, orient='index', columns = [f'Top{i+1}' for i in range(0, number_of_values)])
    else:
        top_variables_df = pd.DataFrame.from_dict(top_variables_per_pc, orient='index', columns = [f'Top{i+1}' for i in range(0,top_n_component_per_pc)])
        
    explained_variance_df = pd.DataFrame.from_dict(explained_variance_per_pc, orient='index', columns=['Explained Variance'])
    
    df_top_features_and_exp_variance = pd.concat([top_variables_df, explained_variance_df], axis=1)
    
    if show_plot:
        # Create a scree plot
        plt.figure(figsize=(10, 5))
        
        # Explained variance ratio plot
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Ratio')
        
        # Cumulative explained variance plot
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.show()
    
    
        # Visualize the explained variance
        plt.figure(figsize=(8, 4))
        plt.bar(range(1, n_components + 1), explained_variance_per_pc.values())
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance per Principal Component')
        plt.xticks(range(1, n_components + 1))
        plt.show()
        
        # print("Top Variables and Explained Variance for Each PC:")
        # print(df_top_features_and_exp_variance)
    
    return df_pca_selected, df_top_features_and_exp_variance, loadings_df   
   



def assign_labels_by_percentage(df, train_percent, validate_percent, test_percent, leaveout_percent):
    # import pdb; pdb.set_trace()
    """
    Assign labels to DataFrame rows based on given percentages.
    
    Parameters:
    - df: The DataFrame to be labeled.
    - train_percent: Percentage of data to assign as "TRAIN."
    - validate_percent: Percentage of data to assign as "VALIDATE."
    - test_percent: Percentage of data to assign as "TEST."
    - leaveout_percent: Percentage of data to assign as "LEAVEOUT."
    
    Returns:
    - A new DataFrame with an additional "split_label" column.
    """
    total_rows = len(df)
    
    if train_percent + validate_percent + test_percent + leaveout_percent > 100:
        raise ValueError("Percentage values exceed 100%.")
    
    split_labels = []
    
    # Calculate the number of rows for each split
    train_rows = int(total_rows * (train_percent / 100))
    validate_rows = int(total_rows * (validate_percent / 100))
    test_rows = int(total_rows * (test_percent / 100))
    leaveout_rows = int(total_rows * (leaveout_percent / 100))
    leftover_rows = total_rows - (train_rows + validate_rows + test_rows + leaveout_rows)
    
    if (train_percent == 0) & (validate_percent == 0):
        unassigned_rows = total_rows - (train_rows + validate_rows + test_rows + leaveout_rows)   
        leaveout_rows = leaveout_rows + leftover_rows - unassigned_rows
    else:
        leaveout_rows = leaveout_rows + leftover_rows
        unassigned_rows = 0
    

    # Assign labels based on the row index
    for i in range(total_rows):
        if i < train_rows:
            split_labels.append("TRAIN")
        elif i < train_rows + validate_rows:
            split_labels.append("VALIDATE")
        elif i < train_rows + validate_rows + test_rows:
            split_labels.append("TEST")
        elif i < train_rows + validate_rows + test_rows + leaveout_rows:
            split_labels.append("LEAVEOUT")
        else:
            split_labels.append("UNASSIGNED")
    
    df_with_labels = df.copy()
    df_with_labels["split_label"] = split_labels
    
    return df_with_labels



def add_void_number_to_classification_data(df, label_col, start_label=None):
    
    temp_df = df.copy()[[label_col]]
    if not start_label:
        start_label = temp_df[label_col].min()
    
    temp_df['epoch'] = (temp_df[label_col] == start_label) & (temp_df[label_col].shift(1) != start_label)
    temp_df.loc[0, 'epoch'] = True
    temp_df['epoch'] = temp_df['epoch'].cumsum()
    return temp_df['epoch']
