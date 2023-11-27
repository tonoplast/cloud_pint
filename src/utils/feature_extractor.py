# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:42:53 2023

@author: WookS
"""

import tsfel

# Retrieves a pre-defined feature configuration file to extract all available features
def get_features_from_timeseries(x_signal, fs, domain=None, get_corr_features=True):
    '''
    x_signal: time_series signal
    fs: sampling rate
    domain: 'statistical', 'temporal', 'spectral', None
    '''
    cfg = tsfel.get_features_by_domain(domain)
    X = tsfel.time_series_features_extractor(cfg, x_signal, fs = fs)
    
    if get_corr_features:
        # Highly correlated features are removed
        corr_features = tsfel.correlated_features(X)
    else:
        corr_features = []
    
    ## make sure to remove corr_features from both Train and Test sets (using Train set corr_features)
    return X, corr_features

def remove_same_valued_columns(df):
    # Find columns with the same value across each column
    columns_to_drop = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            columns_to_drop.append(column)

    df = df.drop(columns=columns_to_drop)
    return df