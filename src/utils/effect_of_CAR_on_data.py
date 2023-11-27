import numpy as np
import matplotlib.pyplot as plt

# Generate two signals with a time lag
t = np.linspace(0, 1, 1000, endpoint=False)
freq = 5  # Frequency in Hz
signal1 = 0.5 * np.sin(2 * np.pi * freq * t)  # Signal 1
signal2 = 0.5 * np.sin(2 * np.pi * freq * t - np.pi/4)  # Signal 2 with a 45-degree phase lag

# Generate common noise
common_noise = 0.2 * np.random.randn(1000)

# Combine signals with common noise
noisy_signal1 = signal1 + common_noise
noisy_signal2 = signal2 + common_noise

# Calculate the common average
common_average = (noisy_signal1 + noisy_signal2) / 2

# Apply Common Average Referencing (CAR)
car_corrected_signal1 = noisy_signal1 - common_average
car_corrected_signal2 = noisy_signal2 - common_average

# Plot the original signals, common noise, and CAR-corrected signals
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, signal1, label='Signal 1')
plt.plot(t, signal2, label='Signal 2')
plt.title('Original Signals (with Time Lag)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, common_noise, label='Common Noise')
plt.title('Common Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, car_corrected_signal1, label='CAR-Corrected Signal 1')
plt.plot(t, car_corrected_signal2, label='CAR-Corrected Signal 2')
plt.title('CAR-Corrected Signals')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Generate two example signals
t = np.linspace(0, 1, 1000, endpoint=False)
freq = 5  # Frequency in Hz
signal1 = 0.5 * np.sin(2 * np.pi * freq * t)  # Signal 1
signal2 = 0.5 * np.sin(2 * np.pi * freq * t - np.pi / 4)  # Signal 2 with a phase lag

# Generate common noise
common_noise = 0.2 * np.random.randn(1000)

# Combine signals with common noise
noisy_signal1 = signal1 + common_noise
noisy_signal2 = signal2 + common_noise

# Calculate the common average
common_average = (noisy_signal1 + noisy_signal2) / 2

# Apply Common Average Referencing (CAR)
car_corrected_signal1 = noisy_signal1 - common_average
car_corrected_signal2 = noisy_signal2 - common_average

# Calculate cross-correlation before and after CAR
cross_corr_original = correlate(noisy_signal1, noisy_signal2, mode='full', method='auto')
cross_corr_car = correlate(car_corrected_signal1, car_corrected_signal2, mode='full', method='auto')

# Plot the cross-correlation results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(-999, 1000), cross_corr_original)
plt.title('Cross-Correlation Before CAR')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.subplot(2, 1, 2)
plt.plot(np.arange(-999, 1000), cross_corr_car)
plt.title('Cross-Correlation After CAR')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate two signals with a lag and add common noise
np.random.seed(0)
t = np.linspace(0, 1, num=1000)
signal1 = np.sin(2 * np.pi * 5 * t)
signal2 = np.sin(2 * np.pi * 5 * t - np.pi / 4)  # Signal with a lag
noise = 0.2 * np.random.randn(1000)
signal1 += noise
signal2 += noise

# Step 2: Apply Common Average Referencing (CAR) to both signals
mean_signal = (signal1 + signal2) / 2
car_signal1 = signal1 - mean_signal
car_signal2 = signal2 - mean_signal

# Step 3: Invert one of the CAR-processed signals
inverted_signal2 = -car_signal2

# Step 4: Introduce a known time lag by shifting the inverted signal
lag = 50
lagged_inverted_signal2 = np.roll(inverted_signal2, lag)

# Step 5: Sum the CAR-processed signal and the lagged, inverted signal
reconstructed_signal = car_signal1 + lagged_inverted_signal2

# Step 6: Plot each step
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.title("Original Signal 1")
plt.plot(t, signal1)

plt.subplot(3, 2, 2)
plt.title("Original Signal 2 with Lag")
plt.plot(t, signal2)

plt.subplot(3, 2, 3)
plt.title("CAR Signal 1")
plt.plot(t, car_signal1)

plt.subplot(3, 2, 4)
plt.title("CAR Signal 2 (Inverted)")
plt.plot(t, inverted_signal2)

plt.subplot(3, 2, 5)
plt.title("Reconstructed Signal 1")
plt.plot(t, reconstructed_signal)

plt.subplot(3, 2, 6)
plt.title(f"CAR Signal 2 with Lag ({lag} samples)")
plt.plot(t, lagged_inverted_signal2)


plt.tight_layout()
plt.show()



# Calculate cross-correlation before and after CAR
cross_corr_original = correlate(signal1, signal2, mode='full', method='auto')
cross_corr_car = correlate(reconstructed_signal, lagged_inverted_signal2, mode='full', method='auto')


# Plot the cross-correlation results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(np.arange(-999, 1000), cross_corr_original)
plt.title('Cross-Correlation Before CAR')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.subplot(2, 1, 2)
plt.plot(np.arange(-999, 1000), cross_corr_car)
plt.title('Cross-Correlation After CAR')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.tight_layout()
plt.show()











import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(0)

# Generate time vector
t = np.linspace(0, 1, 1000)

# Create the main signal (signal1) with a frequency of 100 Hz
frequency = 10  # Frequency of the main signal (in Hz)
signal1 = 2 * np.sin(2 * np.pi * frequency * t)

# Create a time-lagged version of signal1 (signal2) with the same frequency
lag = 0.02  # Lag in seconds
signal2 = 2 * np.sin(2 * np.pi * frequency * (t - lag))

# Create common noise
common_noise = np.random.normal(0, 0.5, len(t))

# Add common noise to both original signals
signal1_with_noise = signal1 + common_noise
signal2_with_noise = signal2 + common_noise

# Apply common average referencing
mean_signal = (signal1_with_noise + signal2_with_noise) / 2
car_signal1 = signal1_with_noise - mean_signal
car_signal2 = signal2_with_noise - mean_signal

# Plot the original signals and the CAR signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal1_with_noise, label='Signal 1 with Noise', color='b')
plt.title('Signal 1 with Noise')

plt.subplot(2, 2, 2)
plt.plot(t, signal2_with_noise, label='Signal 2 with Noise', color='b')
plt.title('Signal 2 with Noise')

plt.subplot(2, 2, 3)
plt.plot(t, car_signal1, label='CAR Signal 1', color='r')
plt.title('CAR Signal 1')

plt.subplot(2, 2, 4)
plt.plot(t, car_signal2, label='CAR Signal 2', color='r')
plt.title('CAR Signal 2')

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(0)

# Generate time vector
t = np.linspace(0, 1, 1000)

# Create the main signal (signal1) with a frequency of 100 Hz
frequency = 20  # Frequency of the main signal (in Hz)
signal1 = 2 * np.sin(2 * np.pi * frequency * t)

# Create a time-lagged version of signal1 (signal2) with the same frequency
lag = 0.02  # Lag in seconds
signal2 = 2 * np.sin(2 * np.pi * frequency * (t - lag))

# Create common noise
noise_color = 'gray'
signal1_color = 'blue'
signal2_color = 'green'
common_noise = np.random.normal(0, 0.5, len(t))

# Add common noise to both original signals
signal1_with_noise = signal1 + common_noise
signal2_with_noise = signal2 + common_noise

# Calculate mean signal
mean_signal = (signal1_with_noise + signal2_with_noise) / 2

# Apply common average referencing
car_signal1 = signal1_with_noise - mean_signal
car_signal2 = signal2_with_noise - mean_signal

# Create a single column graph with subplots
plt.figure(figsize=(8, 8))

# Plot signal1 (original)
plt.subplot(8, 1, 1)
plt.plot(t, signal1, label='Signal 1 (Original)', color=signal1_color)
plt.title('Signal 1 (Original)')

# Plot signal2 (original)
plt.subplot(8, 1, 2)
plt.plot(t, signal2, label='Signal 2 (Original with lag)', color=signal2_color)
plt.title('Signal 2 (Original with lag)')

# Plot common noise
plt.subplot(8, 1, 3)
plt.plot(t, common_noise, label='Common Noise', color=noise_color)
plt.title('Common Noise')

# Plot signal1 with noise
plt.subplot(8, 1, 4)
plt.plot(t, signal1_with_noise, label='Signal 1 + Noise', color=signal1_color)
plt.title('Signal 1 + Noise')

# Plot signal2 with noise
plt.subplot(8, 1, 5)
plt.plot(t, signal2_with_noise, label='Signal 2 + Noise', color=signal2_color)
plt.title('Signal 2 + Noise')

# Plot mean signal
plt.subplot(8, 1, 6)
plt.plot(t, mean_signal, label='Mean Signal', color='red')
plt.title('Mean Signal')

# Plot signal1 with noise - mean signal
plt.subplot(8, 1, 7)
plt.plot(t, car_signal1, label='Signal 1 + Noise - Mean Signal', color=signal1_color)
plt.title('Signal 1 + Noise - Mean Signal')

# Plot signal2 with noise - mean signal
plt.subplot(8, 1, 8)
plt.plot(t, car_signal2, label='Signal 2 + Noise - Mean Signal', color=signal2_color)
plt.title('Signal 2 + Noise - Mean Signal')

plt.tight_layout()
plt.show()


