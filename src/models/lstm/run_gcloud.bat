set PATH=%PATH%;C:\Users\SanjayanA\AppData\Local\Google\Cloud SDK\google-cloud-sdk\.install\.backup\bin


gcloud ai custom-jobs create ^
    --region="australia-southeast1" ^
    --display-name="LSTM-CNN-RawBP" ^
    --worker-pool-spec="machine-type=n1-standard-4,executor-image-uri=asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest,local-package-path=./Scripts,script=lstm.py" ^
    --project="thematic-scope-395304"