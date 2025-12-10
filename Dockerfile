# CUDA 12.3 지원 PyTorch 베이스 이미지 사용
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 환경 변수 설정 (프롬프트 방지)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 참고: PyTorch 공식 이미지는 CUDA 12.1이 최신이지만, 
# CUDA 12.3과 호환됩니다 (하위 호환성)

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    opencv-python-headless

# 포트 노출 (필요시)
EXPOSE 8888

# 기본 명령어
CMD ["/bin/bash"]