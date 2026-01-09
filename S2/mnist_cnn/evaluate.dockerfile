# evaluate.dockerfile
FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy dependency files first (better caching)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

# Install dependencies (reuse pip cache)
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy code (no data, no models baked in)
COPY src/ src/

# Install project itself
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/mnist_cnn/evaluate.py"]
