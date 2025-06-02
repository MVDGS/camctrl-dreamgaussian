FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 로그인 셸로 설정해 conda 초기화 스크립트가 자동 로드되도록 함
SHELL ["/bin/bash", "--login", "-c"]
ENV CONDA_DEFAULT_ENV=dreamgaussian

# 1) 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl build-essential cmake ninja-build \
      ffmpeg libgl1-mesa-glx libegl1-mesa-dev libgles2-mesa-dev \
      libgl1-mesa-dev libgtk2.0-0 libgtk-3-0 vim \
    && rm -rf /var/lib/apt/lists/*

# 2) Miniconda 설치
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 3) Conda 환경 생성 (mkl 버전 고정 포함)
COPY environment.yml /workspace/environment.yml
RUN conda env create -f /workspace/environment.yml && \
    conda clean -afy

# 4) CUDA 아키텍처 지정
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

# 5) 코드 복사
WORKDIR /workspace
COPY . /workspace

# 6) 확장 모듈 및 추가 패키지 설치
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate dreamgaussian && \
    # 로컬 패키지 설치
    pip install --no-cache-dir ./diff-gaussian-rasterization && \
    pip install --no-cache-dir ./simple-knn && \
    # GitHub 패키지 설치
    pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git && \
    pip install --no-cache-dir git+https://github.com/ashawkey/kiuikit.git && \
    # diff-gaussian-rasterization 최종 빌드
    cd diff-gaussian-rasterization && \
    pip install --no-cache-dir .

# 7) 기본 셸으로 복원
SHELL ["/bin/bash", "-lc"]
ENV CONDA_DEFAULT_ENV=dreamgaussian