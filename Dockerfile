# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG NUMPY_VERSION=1.26.4
ARG INSTALL_FLASH_ATTN=0
ARG TRANSFORMERS_VERSION=4.57.3
ARG ULTRALYTICS_VERSION=8.3.101
ARG ULTRALYTICS_CLIP_REF=81ff68ed7ffcac3b40484c914f104f816757308d
ARG HTTP_PROXY=
ARG HTTPS_PROXY=
ARG ALL_PROXY=
ARG NO_PROXY=localhost,127.0.0.1
ARG http_proxy=
ARG https_proxy=
ARG all_proxy=
ARG no_proxy=localhost,127.0.0.1

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub \
    TORCH_HOME=/models/torch \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCH_CUDA_ARCH_LIST="8.0" \
    MAX_JOBS=8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxext6 \
      libxrender1 \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv \
      python3-pip \
      build-essential \
      ninja-build \
      pkg-config \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja

WORKDIR /workspace/MarkIt

COPY requirements.txt /tmp/requirements.txt

# The repository requirements are LongVA-era pins. For Qwen2.5-VL deployment,
# install CUDA PyTorch, Transformers, and video helpers separately to avoid
# incompatible training/attention package pins. numpy is pinned separately
# because accelerate 0.32.1 is not compatible with numpy 2.x.
RUN grep -v -E '^(torch|torchvision|transformers|xformers|flash_attn|ring_flash_attn|deepspeed|bitsandbytes|huggingface_hub|tokenizers|qwen_vl_utils|pyav|numpy)==' \
      /tmp/requirements.txt > /tmp/requirements-runtime.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/cu124

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      numpy==${NUMPY_VERSION} \
      -r /tmp/requirements-runtime.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      av==13.1.0 \
      "qwen-vl-utils[decord]==0.0.8" \
      "ultralytics==${ULTRALYTICS_VERSION}" \
      hf_transfer

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      "git+https://github.com/ultralytics/CLIP.git@${ULTRALYTICS_CLIP_REF}"

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
      "transformers==${TRANSFORMERS_VERSION}"

RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then \
      python -m pip install --no-build-isolation flash-attn==2.6.3; \
    else \
      echo "Skipping flash-attn installation"; \
    fi

COPY . /workspace/MarkIt

RUN mkdir -p /models/huggingface /models/torch outputs

CMD ["python", "eval/vlm_mr_markit.py", "--help"]
