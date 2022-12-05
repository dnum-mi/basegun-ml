FROM ghcr.io/datalab-mi/lab-cv:v0.1.2
LABEL maintainer="datalab-mi"

RUN chown -R 42420:42420 /workspace
WORKDIR /workspace
COPY training/requirements.txt training/prepare_data.py training/README.md ./
RUN pip install --no-cache-dir -r requirements.txt 