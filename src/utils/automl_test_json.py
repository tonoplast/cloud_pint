# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 07:29:55 2023

@author: sungw
"""

## wsl2 docker
## leaving this commented out just in case

## Model needs to be 'downloaded' first. Windows has problem because of the folder name that works with Linux but not windows
## Authentication seems dodgy when using Linux (Requires XLaunch).. so it's a bit of problem. 
## You can also download model within the folder rather than from the model directory itself
# docker pull asia-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20230910_1325

# =============================================================================
# 80/10/10 (train/validate/test)
# =============================================================================
## notice the folder letters are all in lower case
# classification_model_folder="/mnt/o/pint/_projects/bladder/012_19 sparc/data/curated data/processed_raw_cleaned_xcor_cleaned/automl/classification/output"
# classification_xtest_folder="/mnt/o/pint/_projects/bladder/012_19 sparc/data/curated data/processed_raw_cleaned_xcor_cleaned/automl/classification/input"
# docker run -v "${classification_model_folder}/model-2022661590456729600_1/tf-saved-model/classification:/models/default" -p 8080:8080 -it asia-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20230910_1325

# =============================================================================
# 70/10/10/10 (train/validate/test/leaveout)
# =============================================================================
# docker run -v "${classification_model_folder}/model-2022661590456729600_2/tf-saved-model/classification:/models/default" -p 8080:8080 -it asia-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20230910_1325

# =============================================================================
# leaveout_1_12_225
# =============================================================================
# docker run -v "${classification_model_folder}/model-1534469631631163392_2/tf-saved-model/classification:/models/default" -p 8080:8080 -it asia-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20230910_1325

# curl -X POST --data "@${classification_xtest_folder}/x_test_neural_act-NonMovingPeakIndividual_50s_win_auto_end_automl_40_pc.json" http://localhost:8080/predict
# curl -X POST --data "@${classification_xtest_folder}/x_test.json" http://localhost:8080/predict | python -m json.tool
# curl -X POST --data "@${classification_xtest_folder}/x_test_leftout_1_225_leftover.json" http://localhost:8080/predict | python -m json.tool

# def dos2unix(source_file, dest_file):
#     with open(source_file, 'rb') as src:
#         content = src.read()

#     content = content.replace(b'\r\n', b'\n')

#     with open(dest_file, 'wb') as dst:
#         dst.write(content)

# # Convert the shell script
# dos2unix('./src/utils/run_docker_for_test_model.sh', './src/utils/run_docker_for_test_model_unix.sh')


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import json
import requests

# file_suffix = '_70_10_10_10_leaveout' #70/10/10/10 (last 10% left out)
file_suffix = '_leftout_1_225_leftover' #left out 12_225


class_path = Path(r"O:\PINT\_Projects\Bladder\012_19 SPARC\Data\Curated data\Processed_raw_cleaned_xcor_cleaned\AutoML\Classification\Input")
# mydata = class_path.joinpath("neural_act_nonmovingpeakindividual_50s_win_auto_end_automl_40_pc.csv") #80/10/10
mydata = class_path.joinpath(f"neural_act_nonmovingpeakindividual_50s_win_auto_end_automl_40_pc{file_suffix}.csv")
test = pd.read_csv(mydata)

sam = test.copy()
actual_labels = sam['bp_label_auto_end']
sam = sam.drop(columns="bp_label_auto_end").astype(str)

jsonx_test_data = sam.to_dict(orient='records')
json_x_test = {"instances": jsonx_test_data}
json_save_path = Path(rf"{class_path}/x_test{file_suffix}.json")

with open(json_save_path, 'w') as json_file:
    json.dump(json_x_test, json_file, indent=2)
    


# Define the URL and data file path
url = 'http://localhost:8080/predict'
data_file_path = f'{class_path}/x_test{file_suffix}.json'  # Replace with the path to your data file

# Read the JSON data from the file
with open(json_save_path, 'r') as data_file:
    data = data_file.read()

# Send a POST request to the URL with the JSON data
response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()

    # Print the formatted JSON response
    print(json.dumps(result, indent=2))
else:
    print(f"Request failed with status code {response.status_code}")



# =============================================================================
# ## if single
# =============================================================================
# # Extract scores and classes lists
# scores = result['predictions'][0]['scores']
# classes = result['predictions'][0]['classes']

# # Create a DataFrame with 'scores' and 'classes' columns
# df = pd.DataFrame({'scores': scores, 'classes': classes})

# # Display the DataFrame
# print(df)


# # Initialize an empty DataFrame
dfs = []

# Iterate through each set of predictions and append to the DataFrame
for e, (actual_label, prediction_set) in enumerate(zip(actual_labels, result['predictions'])):
    scores = prediction_set['scores']
    classes = prediction_set['classes']
    set_df = pd.DataFrame({'grouping': e, 'scores': scores, 'classes': classes, 'actual_label': actual_label})
    dfs.append(set_df)

# Display the combined DataFrame
df = pd.concat(dfs, ignore_index=True)
df['classes'] = df['classes'].astype(int)

# print(df)
max_indices = df.groupby('grouping')['scores'].idxmax()
df_max = df.loc[max_indices].reset_index()




# =============================================================================
# Plot
# =============================================================================
# Create a figure with two subplots
fig, ax1 = plt.subplots(figsize=(14, 6))

# Create a scatter plot with the primary y-axis
# ax1.scatter(df_max.index.values, df_max['scores'], c=df_max['classes'], cmap='viridis', marker='o', s=40, alpha=0.8)

# Create a scatter plot with the primary y-axis and colorbar
sc = ax1.scatter(df_max.index.values, df_max['scores'], c=df_max['classes'], cmap='viridis', marker='o', s=40, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Class', rotation=270, labelpad=15)  # Label for the colorbar



# Add labels and title for the primary y-axis
ax1.set_xlabel('Samples')
ax1.set_ylabel('Scores (probabilities)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(0, 1.1)  # Set y-axis limits for ax1

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot the 'actual_label' data on the secondary y-axis
ax2.scatter(df_max.index.values, df_max['actual_label'], color='red', marker='.', s=20, alpha=0.4)

# Add labels and title for the secondary y-axis
ax2.set_ylabel('Actual Label', color='r')
ax2.tick_params('y', colors='r')
ax2.set_ylim(0, 11)  # Adjust the limits for ax2
ax2.grid(True)  # Enable grid for ax2

# plt.title('Score Over Time by Sample (One void)')
plt.title('Score Over Time by Sample (Multiple Voids)')

# plt.colorbar(label='Predicted Labels')
# Show the plot
plt.grid(True)
plt.show()


incorrect_prediction = (df_max['classes'] != df_max['actual_label'])
incorrect_prediction.sum()
len(df_max)

df_max[incorrect_prediction].rename(columns={'classes': 'predicted_label'})


# from sklearn.metrics import confusion_matrix
# confusion_matrix(df_max['actual_label'], df_max['classes'])


pd.crosstab(df_max['actual_label'], df_max['classes'], 
            rownames=['True label'], colnames=['Predicted Label'])


from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

y_test = df_max['actual_label']
y_pred = df_max['classes']


print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


