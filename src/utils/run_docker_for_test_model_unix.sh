
## wsl2 docker
## Model needs to be 'downloaded' first. Windows has problem because of the folder name that works with Linux but not windows
## Authentication seems dodgy when using Linux (Requires XLaunch).. so it's a bit of problem. 
## You can also download model within the folder rather than from the model directory itself

docker_image="asia-docker.pkg.dev/vertex-ai/automl-tabular/prediction-server:20230910_1325"
sudo service docker start
docker pull ${docker_image}

## Folder settings
## notice the folder letters are all in lower case
classification_model_folder="/mnt/o/pint/_projects/bladder/012_19 sparc/data/curated data/processed_raw_cleaned_xcor_cleaned/automl/classification/output"
classification_xtest_folder="/mnt/o/pint/_projects/bladder/012_19 sparc/data/curated data/processed_raw_cleaned_xcor_cleaned/automl/classification/input"

input_model=$1
docker run -v "${classification_model_folder}/${input_model}/tf-saved-model/classification:/models/default" -p 8080:8080 -it ${docker_image}



# =============================================================================
# 80/10/10 (train/validate/test)
# =============================================================================
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
