

@echo off

set PATH=%PATH%;C:\Users\SanjayanA\AppData\Local\Google\Cloud SDK\google-cloud-sdk\.install\.backup\bin


:: Build Docker Image
docker build -t lstm_image .

:: Tag the Docker Image
docker tag lstm_image gcr.io/thematic-scope-395304/lstm_image


:: Push to Google Container Registry
docker push gcr.io/thematic-scope-395304/lstm_image


gcloud ai hp-tuning-jobs create ^
    --region="australia-southeast1" ^
    --display-name="LSTM-CNN-RawBP" ^
    --config=hptuning_config.yaml ^
    --project="thematic-scope-395304" ^
    --max-trial-count=30 ^
    --parallel-trial-count=2
    

echo Done
