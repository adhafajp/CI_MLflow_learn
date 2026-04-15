FROM python:3.12.7-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    setuptools==82.0.1 \
    mlflow==3.10.1 \
    fastapi==0.112.2 \
    uvicorn==0.44.0 \
    scikit-learn==1.5.2 \
    pandas==2.3.3 \
    numpy==2.4.4 \
    psutil==6.0.0 \
    waitress==3.0.2

COPY model_dir /app/model

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "5000", "--no-conda"]