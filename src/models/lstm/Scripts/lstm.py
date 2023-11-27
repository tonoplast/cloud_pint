# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from google.cloud import storage, bigquery
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy import stats
from datetime import datetime
import gcsfs
from tempfile import NamedTemporaryFile, TemporaryDirectory
import io
import shutil
import os
from tensorflow.keras import backend as K
import hypertune

#%% Parse arguments
import argparse

try:
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--input_window_size', type=int, default=60)
    parser.add_argument('--target_offset_smpl', type=int, default=10)



    args = parser.parse_args()
except:
    print("For running Jupyter Cells (with no argument entries)")
    args = argparse.Namespace(
        input_window_size=60,
        target_offset_sample=10
    )



# %% [markdown]
# # Load Data
current_timestamp = datetime.utcnow().isoformat()
# Store in GCS 
# Initialize the GCS client
storage_client = storage.Client()
# Name of the bucket and the path
bucket_name = 'bionics-data-store'
# Get the bucket reference
bucket = storage_client.bucket(bucket_name)
# Specify model filename and path within the bucket
# Replace colons in the timestamp with underscores
safe_timestamp = current_timestamp.replace(":", "_")
# Modify the model filename accordingly
path_inside_bucket = "PINT/Bladder/Models/hp-tune-2/model_"+safe_timestamp


def save_fig(fig, filename):
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)  # Seek back to the beginning of the buffer

    # Initialize the Google Cloud Storage client
    blob = bucket.blob(f"{path_inside_bucket}/{filename}")

    # Upload the BytesIO buffer to GCS
    blob.upload_from_file(buf, content_type='image/png')

    # Close the buffer
    buf.close()



# %%
# Read data from GCS
def list_files(bucket_name, prefix):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]

base_path=f"PINT/Bladder/Processed_raw_cleaned_xcor_cleaned"
subject_to_load = "12"

bucket_name = "bionics-data-store"

prefix = f"{base_path}/Neural/{subject_to_load}"
neural_file_names = list_files(bucket_name, prefix)

bp_prefix = f"{base_path}/BladderPressure/{subject_to_load}"
bp_file_names = list_files(bucket_name, bp_prefix)

# Initial value for time offset
time_offset = 0
dfs = []

for bp_file_name, neural_file_name in zip(bp_file_names, neural_file_names):
    path = f"gs://{bucket_name}/{neural_file_name}"
    df = pd.read_parquet(path)
    print(f"READING {neural_file_name}")
    # Adjust the time column
    if "time" in df.columns:
        df["time"] = df["time"]+time_offset
        time_offset = df["time"].iloc[-1]  # Update time offset for next DataFrame
    
    # Read bladder pressure 
    path = f"gs://{bucket_name}/{bp_file_name}"
    bp_df = pd.read_parquet(path)
    print(f"CONCAT {bp_file_name}")

    bp_df.update(df)
    df = pd.concat([bp_df, df.drop(df.columns.intersection(bp_df.columns), axis=1)], axis=1)



    dfs.append(df)

raw_df = pd.concat(dfs, ignore_index=True)

# %%

# Outlier removal
# Find indices where the condition is met
df = raw_df.copy()
df["neural_act"] = signal.medfilt(raw_df["neural_act-NonMovingPeakIndividual"], 3)


# Proc data vis
plt.plot(df['bladder_pressure'])
plt.title("Bladder Pressure")
save_fig(plt, "orig_target.png")

plt.figure()
plt.plot( df['neural_act'])
plt.title("Neural activity (Cross Correlated Peak)")
save_fig(plt, "orig_pred.png")


# %% [markdown]
# # Data Preparation

# %%
# Parameter Definitions

# Data split
split = {
    'train': 0.7, # TRAIN: 70%, TEST: 15%, VAL: 15%
    'test': 0.5,
    # val: 1-train-test
}

# Window signal splitting

prediction_frequency = 1; # Predict every sample

# Model
model = Sequential()

# CNN layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(args.input_window_size, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# LSTM layer
model.add(LSTM(50, return_sequences=True))  # Use return_sequences if adding more LSTM layers.
model.add(LSTM(50))

# Dense layer
model.add(Dense(50, activation='relu'))
model.add(Dense(1))


def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
model.compile(optimizer='adam', loss='mse', metrics=[r2_score])

EPOCHS = 30
BATCH_SIZE = 32



# %%
# Split data into train/test data
train_df, temp_df = train_test_split(df, test_size=1-split['train'], shuffle=False) 
val_df, test_df = train_test_split(temp_df, test_size=split['test'], shuffle=False) 

# Visualise
plt.plot(train_df['bladder_pressure'], label="Training Data")
plt.plot(val_df['bladder_pressure'], label="Validation Data")
plt.plot(test_df['bladder_pressure'], label="Testing Data")
plt.title("Bladder Pressure Test/Train Split")
plt.legend()

save_fig(plt, "train_val_test_split.png")

# %%
# Simulate live capture -> split signal into iterative windows
#
# To add: (Added) 
# - Target data offsets (i.e. how many samples into the future/past do we predict)
# - Window spacing (i.e. interwindow sample spacing -> i.e. how often are we predicting)
#
def create_windows(neural_activity, bladder_pressure, na_window_size=args.input_window_size, target_offset=args.target_offset_smpl, window_spacing=1):
    na_windows = []

    
    for i in range(0, len(neural_activity) - na_window_size - target_offset, window_spacing):
        na_windows.append(neural_activity[i:i+na_window_size])

    
    bp_arr = bladder_pressure[na_window_size+target_offset:]
                
    return np.array(na_windows), bp_arr.values


def live_window_to_signal():
    pass

na_windows_train, bp_sig_train = create_windows(train_df['neural_act'], 
                                        train_df['bladder_pressure'],
                                                   window_spacing=prediction_frequency)


na_windows_val, bp_sig_val = create_windows(val_df['neural_act'], 
                                        val_df['bladder_pressure'],
                                               window_spacing=prediction_frequency)



# %% [markdown]
# # Model Training

# %%
# Build & Train the LSTM Model
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

history = model.fit(
    na_windows_train, 
    bp_sig_train, 
    validation_data=(na_windows_val, bp_sig_val), 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE,
    callbacks=[early_stop])

# Metric Logging
hp_metric = history.history['r2_score'][-1]
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='r2_score',
    metric_value=hp_metric,
    global_step=EPOCHS
)

# %% [markdown]
# # Prediction

# %%
# Load model if required
"""
model_str = tf.io.read_file("gs://bionics-data-store/PINT/Bladder/Models/model_2023-09-25T03_44_11.821702.h5")
# Write the byte string to a temporary file
with NamedTemporaryFile(delete=True) as tmp:
    tmp.write(model_str.numpy())
    # Load the model from the temporary file
    model = tf.keras.models.load_model(tmp.name, custom_objects={"r2_score": r2_score})
"""

"""
model = tf.keras.models.load_model("model_2023-09-18T08_16_03.758013.h5")
"""



# %%
# Prediction Parameters


# Prediction

na_windows_test, bp_actual_signal = create_windows(test_df['neural_act'], 
                                                test_df['bladder_pressure'],
                                                 window_spacing=prediction_frequency)

bp_pred_signal = model.predict(na_windows_test)
bp_pred_signal = bp_pred_signal[:, 0]


# %%

plt.figure(figsize=(20,5))
plt.plot(bp_pred_signal, label="Predicted Signal")
plt.plot(np.concatenate([np.zeros(args.target_offset_smpl),bp_actual_signal]), label="Actual Signal")
plt.legend()
save_fig(plt, "pred_signal.png")

# %%

# Calculate r2 metric between estimated and real signal
r2 = stats.pearsonr(bp_pred_signal, bp_actual_signal)[0]
print(f"ARJ: R2 Value: {r2}, Input Win Size: {args.input_window_size}, Target Offset: {args.target_offset_smpl}")



# %%
# Save metrics to bigquery
brief_desc = "NonMovingPeak_RawBP"

# Initialize the BigQuery client
client = bigquery.Client(project='thematic-scope-395304')

# Set the table reference
table_ref = client.dataset('pint_metrics').table('models')


model_type = "CNN/LSTM"
time = datetime.utcnow().isoformat()
# CREATE MODEL DESCRIPTION
from io import StringIO
import sys
# Backup the standard output
original_stdout = sys.stdout
# Set the new standard output to the in-memory stream
sys.stdout = StringIO()
# Print the summary to the in-memory stream
model.summary()
# Retrieve the contents of the in-memory stream
model_description = sys.stdout.getvalue()
# Restore the standard output
sys.stdout = original_stdout
description = f"""
Parameter Definitions:

Data Split:
- Train: {split['train'] * 100}%
- Test: {split['test'] * 100}%
- Val: {100 - (split['train'] + split['test']) * 100}%

Window Signal Splitting:
- Input Window Size: {args.input_window_size}
- Target Offset Samples: {args.target_offset_smpl}
- Prediction Frequency: {prediction_frequency} (Predict every sample)

Model Info:
{model_description}

Training Parameters:
- Epochs: {EPOCHS}
- Batch Size: {BATCH_SIZE}
"""



# Construct GCS path for the model
model_gcs_path = f"gs://{bucket_name}/{path_inside_bucket}"

local_model_temp = "./model"
model.save(local_model_temp)
blob = bucket.blob(f"{path_inside_bucket}/saved_model.pb")
blob.upload_from_filename(f"{local_model_temp}/saved_model.pb")
shutil.rmtree(local_model_temp)

blob = bucket.blob(f"{path_inside_bucket}/saved_model.h5")
# Save the model locally
model.save("saved_model.h5")
# Upload the saved model to GCS
blob.upload_from_filename("saved_model.h5")
os.remove("saved_model.h5")


# Construct the row data
rows_to_insert = [
    {"ModelType": model_type, "Description": description, "Time": time, "R2": r2, "ModelGCSPath": model_gcs_path, "BriefDesc": brief_desc}
]

# Insert rows
errors = client.insert_rows_json(table_ref, rows_to_insert)

# Handle potential errors
if errors:
    print('Errors:', errors)
else:
    print('Rows inserted!')




# %%
