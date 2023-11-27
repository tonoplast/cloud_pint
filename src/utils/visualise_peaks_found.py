# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:16:45 2023

@author: WookS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Create a subset of the DataFrame
peaks_found = 'peak_found-NonMovingPeakIndividual'
check = df_merged[[peaks_found, 'trough_to_trough', 'peaks']]
check['IsFirstInGroup'] = check.groupby('trough_to_trough').cumcount() == 0
colours = np.where(check[peaks_found], 'green', 'red')

# Create a scatter plot for the boolean values with labels
plt.figure(figsize=(16, 4))
plt.scatter(check.index, check[peaks_found], c=colours, marker='o', s=10, alpha=0.4)

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Peaks Found')
plt.title('Peaks Found Over Time', pad=10)

# Customize the y-axis ticks for boolean values (True/False)
plt.yticks([False, True], ['False', 'True'])

# Add vertical lines and labels for 'IsFirstInGroup'
for i, cp in enumerate(check[check['IsFirstInGroup']].index):
    plt.axvline(x=cp, color='blue', linestyle='-', alpha=0.3)
    plt.text(cp, plt.ylim()[1], str(i), va='bottom', ha='center', color='blue', fontsize=8, alpha=0.5)

# Add vertical lines for 'peaks'
for i, cp in enumerate(check[check['peaks']].index):
    plt.axvline(x=cp, color='green', linestyle='--', alpha=0.2)

# Show the plot
plt.show()






# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal

# # Create synthetic sine wave signals
# t = np.linspace(0, 1, 1000, endpoint=False)  # Time array
# signal1 = np.sin(2 * np.pi * 5 * t)  # Sine wave with frequency of 5 Hz
# signal2 = np.sin(2 * np.pi * 5 * t + np.pi/2)  # Sine wave with a 90-degree phase shift

# # Compute the cross-correlation between the two sine wave signals
# cross_correlation = signal.correlate(signal1, signal2, mode='full')

# # Create a lag array
# lag = np.arange(-999, 1000)  # Full range of lags

# # Plot the cross-correlation against the lag
# plt.figure(figsize=(12, 6))
# plt.plot(lag, cross_correlation)
# plt.xlabel('Lag')
# plt.ylabel('Cross-Correlation')
# plt.title('Cross-Correlation of Two Sine Waves with a Phase Shift')
# plt.grid(True)
# plt.show()

