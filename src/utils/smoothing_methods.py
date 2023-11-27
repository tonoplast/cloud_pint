# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:00:02 2023

@author: sungw
"""

import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from scipy.signal import medfilt, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from src.preprocessing.preproc_tools import butter_bandpass_filter, butter_lowpass_filter

class SmoothingMethods:
    
    def make_odd(self, x):
        if x % 2 == 0:  # Check if x is even
            x += 1  # Increment x by 1 to make it odd
        return x
    
    def smooth_envelope(self, signal, cutoff_frequency, sample_rate):
        # import pdb; pdb.set_trace()

        # Compute the Hilbert transform to obtain the analytic signal
        analytic_signal = hilbert(signal)
    
        # Compute the envelope by taking the magnitude of the analytic signal
        envelope = np.abs(analytic_signal)

        # Calculate the Nyquist frequency
        nyquist_frequency = 0.5 * sample_rate

        # Normalize the cutoff frequency relative to the Nyquist frequency
        normal_cutoff = cutoff_frequency / nyquist_frequency
        
        # Check if the normalized cutoff frequency is within the valid range (0, 1)
        if normal_cutoff >= 1.0:
            raise ValueError("Cutoff frequency exceeds the Nyquist frequency.")


        # Apply a low-pass Butterworth filter to smooth the envelope
        b, a = butter(4, normal_cutoff, btype='low')
        smoothed_envelope = filtfilt(b, a, envelope)

        return smoothed_envelope
    
    
    def exponential_moving_average(self, data, alpha):
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        return ema

    def savitzky_golay_filter(self, data, window_size, order):
        return savgol_filter(data, window_size, order)

    def lowess_smoothing(self, data, frac=0.3):
        return lowess(data, np.arange(len(data)), frac=frac, return_sorted=False)

    def moving_median(self, data, window_size):
        return medfilt(data, kernel_size=window_size)

    def kalman_filter(self, data):
        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
        kf = kf.em(data, n_iter=5)  # Fit the model to data
        smoothed_state_means, _ = kf.smooth(data)  # Extract smoothed state means from the output
        return smoothed_state_means.flatten()  # Flatten the array to (2953,)

    def seasonal_decomposition(self, data, seasonal_period=None, robust=False, show_plot=False, cutoff_frequency=10, sample_rate=2):
        # import pdb; pdb.set_trace()
        # Find the best seasonal period using ACF if not provided
        if seasonal_period is None:
            acf_values  = acf(data, fft=True)
            peaks, _ = find_peaks(acf_values)
            peak_lags = peaks[acf_values[peaks].argsort()][::-1]  # Sort by peak heights in descending order
            seasonal_period = peak_lags[0] if len(peak_lags) > 0 else 2 # period must be a positive integer >= 2
    
        # Perform seasonal decomposition using STL
        stl = STL(data, period=seasonal_period, robust=robust)
        result = stl.fit()
        seasonal, trend, residual = result.seasonal, result.trend, result.resid
        # envelope = np.abs(hilbert(seasonal))
        
        seasonal_abs = abs(seasonal)
        trend_abs = abs(trend)
        residual_abs = abs(residual)
        
        
        # Calculate the envelope of the 'seasonal' component
        envelope_seasonal = self.smooth_envelope(seasonal, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        envelope_seasonal_abs = self.smooth_envelope(seasonal_abs, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        envelope_trend = self.smooth_envelope(trend, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        envelope_trend_abs = self.smooth_envelope(trend_abs, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        envelope_residual = self.smooth_envelope(residual, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        envelope_residual_abs = self.smooth_envelope(residual_abs, cutoff_frequency, sample_rate)  # Adjust cutoff_frequency as needed
        
        if show_plot:
            # Plot the ACF to visualize the best seasonal period
            plt.plot(acf_values)
            plt.xlabel('Lags')
            plt.ylabel('Autocorrelation')
            plt.title('Autocorrelation Function (ACF)')
            plt.grid(True)
            plt.show()
        
            # Plot the decomposition results
            fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(data, label='Original Data')
            axs[0].legend()
            axs[0].set_title('Original Data')
            axs[1].plot(trend, label='Trend')
            axs[1].legend()
            axs[1].set_title('Trend')
            axs[2].plot(seasonal, label='Seasonal')
            axs[2].legend()
            axs[2].set_title('Seasonal')
            axs[3].plot(residual, label='Residual')
            axs[3].legend()
            axs[3].set_title('Residual')
            plt.xlabel('Time')
            plt.tight_layout()
            plt.show()
        return seasonal, trend, residual, envelope_seasonal, envelope_trend, envelope_residual, envelope_seasonal_abs, envelope_trend_abs, envelope_residual_abs

    def rolling_mean(self, data, window_size):
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')

    def cumulative_sum(self, data):
        return np.cumsum(data)
    
    def gaussian_filt(self, data, sigma):
        return gaussian_filter1d(data, sigma)
    
    def bandpass_filt(self, data, lowcut, highcut, fs, order=3, use_lfilter=True):
        return butter_bandpass_filter(data, lowcut, highcut, fs, order=3, use_lfilter=True)

    def lowpass_filt(self, data, cutoff, fs, order=3, use_lfilter=True):
        return butter_lowpass_filter(data, cutoff, fs, order=3, use_lfilter=True)
    
    def max_norm(self, data):
        return data/np.max(data)
    
    def adaptive_filter_predict(self, signal_A, signal_B, time, filter_order=16, step_size=0.01, show_plot=True):
        """
        Predicts signal B based on signal A using an adaptive filter.
        
        Args:
            signal_A (numpy.ndarray): The reference or desired signal.
            signal_B (numpy.ndarray): The primary input signal to be predicted.
            filter_order (int): The order of the adaptive filter.
            step_size (float): The step size for coefficient updates.

        Returns:
            numpy.ndarray: The estimated signal B.
        """
        # Initialize filter coefficients and other variables
        filter_coeffs = np.zeros(filter_order)
        estimated_B = np.zeros_like(signal_B)

        # Adaptive filter loop
        for i in range(len(signal_B)):
            # Extract the reference signal from signal A
            reference_signal = signal_A[max(0, i-filter_order+1):i+1]

            # Zero-pad the reference signal if it's shorter than filter_order
            if len(reference_signal) < filter_order:
                reference_signal = np.pad(reference_signal, (0, filter_order - len(reference_signal)))

            # Compute the filter output as an estimate of signal B
            filter_output = np.sum(filter_coeffs * reference_signal)

            # Calculate the error (desired signal - filter output)
            error = signal_B[i] - filter_output

            # Update the filter coefficients using the LMS algorithm
            filter_coeffs += 2 * step_size * error * reference_signal

            # Store the estimated signal B
            estimated_B[i] = filter_output
        
        if show_plot:
            # Plot the original signal B and the estimated signal B
            plt.figure(figsize=(10, 6))
            plt.plot(time, signal_A, label='Original Signal A', alpha=0.7)
            plt.plot(time, signal_B, label='Original Signal B', alpha=0.7)
            plt.plot(time, estimated_B, label='Estimated Signal B', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.show()

        return estimated_B
