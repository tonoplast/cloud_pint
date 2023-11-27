FROM asia-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12.py310:latest

WORKDIR /app

COPY ./Scripts /app/Scripts

ENTRYPOINT ["python", "/app/Scripts/lstm.py"]

