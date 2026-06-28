FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

WORKDIR /opt/soundpost
COPY requirements.txt ./
RUN python -m pip install --upgrade "pip==23.3" && \
    python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["soundpost"]
CMD ["--help"]
