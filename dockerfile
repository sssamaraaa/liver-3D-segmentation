FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN set -eux; \
    apt-get update; \
    apt-get install -y python3-pip python3-venv; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    ln -sf /usr/bin/python3 /usr/bin/python; \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /proj

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

COPY ml/model/unet.pth /proj/ml/model/unet.pth
COPY ml/__init__.py /proj/ml/__init__.py
COPY ml/src/__init__.py /proj/ml/src/__init__.py
COPY ml/src/data_preprocessing.py /proj/ml/src/data_preprocessing.py
COPY ml/src/inference.py /proj/ml/src/inference.py
COPY ml/src/model.py /proj/ml/src/model.py
COPY ml/src/logging_conf.py /proj/ml/src/logging_conf.py
COPY ml/src/utils.py /proj/ml/src/utils.py
COPY app/api/predict.py /proj/app/api/predict.py
COPY app/services/model_service.py /proj/app/services/model_service.py
COPY app/logging_conf.py /proj/app/logging_conf.py
COPY app/main.py /proj/app/main.py

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]