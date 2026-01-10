# train.dockerfile
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

WORKDIR /

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

RUN mkdir -p models
RUN mkdir -p models reports/figures



ENTRYPOINT ["python", "-u", "src/mnist_cnn/train.py"]
