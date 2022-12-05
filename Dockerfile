FROM ghcr.io/datalab-mi/lab-cv:v0.1.2
LABEL maintainer="datalab-mi"

RUN chown -R 42420:42420 /workspace
WORKDIR /workspace
COPY classification/requirements.txt classification/prepare_data.py classification/README.md ./
RUN pip install --no-cache-dir -r requirements.txt 