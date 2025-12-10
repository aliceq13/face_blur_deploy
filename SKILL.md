# FaceBlur AI Model Serving & Web Service Development Guide

## 목차
1. [기술 스택 개요](#기술-스택-개요)
2. [아키텍처 설계](#아키텍처-설계)
3. [FastAPI AI 모델 서빙](#fastapi-ai-모델-서빙)
4. [Django 웹 프레임워크](#django-웹-프레임워크)
5. [AWS 인프라 구성](#aws-인프라-구성)
6. [데이터베이스 설계](#데이터베이스-설계)
7. [배포 전략](#배포-전략)
8. [보안 및 성능 최적화](#보안-및-성능-최적화)

---

## 기술 스택 개요

### 핵심 기술 스택
```
┌─────────────────────────────────────────────────────────┐
│                    Technology Stack                      │
├─────────────────────────────────────────────────────────┤
│ Frontend:    HTML5, CSS3, JavaScript (Vanilla/React)    │
│ Web Backend: Django 4.2+, Django REST Framework         │
│ AI Backend:  FastAPI 0.104+, Uvicorn                    │
│ AI/ML:       YOLOv8, MediaPipe, OpenCV, PyTorch         │
│ Database:    PostgreSQL 15+, Redis 7+                   │
│ Storage:     AWS S3                                      │
│ Queue:       Celery, Redis (Broker)                     │
│ Server:      AWS EC2, Nginx, Gunicorn                   │
│ Container:   Docker, Docker Compose                     │
└─────────────────────────────────────────────────────────┘
```

### 서비스 분리 철학
- **Django**: 사용자 관리, 웹 UI, 비즈니스 로직, 데이터베이스 관리
- **FastAPI**: AI 모델 추론 전용 (빠른 응답, GPU 활용 최적화)
- **Celery**: 비동기 긴 작업 처리 (영상 전체 프레임 처리)

---

## 아키텍처 설계

### 전체 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                        Internet (HTTPS)                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  AWS Route 53  │ (DNS)
                    └────────┬───────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │  AWS Application Load Balancer │
            └────────┬───────────┬────────────┘
                     │           │
         ┌───────────┘           └──────────┐
         ▼                                  ▼
┌──────────────────┐              ┌──────────────────┐
│   EC2 Instance   │              │   EC2 Instance   │
│  (Django Web)    │              │  (FastAPI AI)    │
│                  │              │                  │
│ ┌──────────────┐ │              │ ┌──────────────┐ │
│ │    Nginx     │ │              │ │    Nginx     │ │
│ │  (Port 80)   │ │              │ │  (Port 80)   │ │
│ └──────┬───────┘ │              │ └──────┬───────┘ │
│        │         │              │        │         │
│ ┌──────▼───────┐ │              │ ┌──────▼───────┐ │
│ │   Gunicorn   │ │              │ │   Uvicorn    │ │
│ │   (Django)   │ │              │ │  (FastAPI)   │ │
│ │  (Port 8000) │ │              │ │  (Port 8001) │ │
│ └──────┬───────┘ │              │ └──────┬───────┘ │
│        │         │              │        │         │
│ ┌──────▼───────┐ │              │ ┌──────▼───────┐ │
│ │    Django    │◄┼──HTTP API──►│ │   FastAPI    │ │
│ │  Application │ │              │ │     API      │ │
│ └──────┬───────┘ │              │ └──────┬───────┘ │
│        │         │              │        │         │
│        │         │              │ ┌──────▼───────┐ │
│        │         │              │ │  AI Models   │ │
│        │         │              │ │  - YOLOv8    │ │
│        │         │              │ │  - FaceNet   │ │
│        │         │              │ │  - OpenCV    │ │
│        │         │              │ └──────────────┘ │
│        │         │              │                  │
│        │         │              │  GPU: NVIDIA T4  │
└────────┼─────────┘              └──────────────────┘
         │
         │
    ┌────┴─────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│  PostgreSQL  │  │    Redis     │  │     AWS S3       │
│   (RDS)      │  │ (ElastiCache)│  │                  │
│              │  │              │  │  - 원본 영상      │
│ - 사용자 정보 │  │ - Celery Q   │  │  - 처리 영상      │
│ - 영상 메타   │  │ - 캐시       │  │  - 임시 프레임    │
│ - 작업 로그   │  │ - 세션       │  │                  │
└──────────────┘  └──────────────┘  └──────────────────┘
         │
         ▼
┌──────────────────┐
│  Celery Workers  │
│   (EC2 별도)     │
│                  │
│  - 영상 분석     │
│  - 프레임 추출   │
│  - 후처리        │
└──────────────────┘
```

### 요청 흐름

#### 1. 영상 업로드 흐름
```
사용자 → Nginx → Django → S3 (원본 저장) → Celery Task 생성
                                           ↓
                    Redis ← Celery Worker → FastAPI (얼굴 검출)
                      ↓
                  Django (DB 업데이트)
                      ↓
                  WebSocket → 사용자 (실시간 진행률)
```

#### 2. 얼굴 인식 요청 흐름
```
Django → FastAPI (/detect-faces)
           ↓
    YOLOv8 모델 추론
           ↓
    얼굴 좌표 반환
           ↓
        Django (결과 저장)
```

#### 3. 영상 처리 흐름
```
사용자 (얼굴 선택) → Django → Celery Task
                                  ↓
                    Celery Worker (프레임 추출)
                                  ↓
                    FastAPI (배치 얼굴 검출) ×N
                                  ↓
                    OpenCV (블러 처리)
                                  ↓
                    FFmpeg (영상 재인코딩)
                                  ↓
                    S3 업로드 → Django DB 업데이트
                                  ↓
                    사용자 알림 (완료)
```

---

## FastAPI AI 모델 서빙

### FastAPI 프로젝트 구조

```
fastapi_ai_server/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 애플리케이션 진입점
│   ├── config.py                  # 설정 파일
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face_detector.py      # YOLOv8 얼굴 검출 모델
│   │   ├── face_embedder.py      # FaceNet 임베딩 모델
│   │   └── model_loader.py       # 모델 로딩 유틸리티
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── face.py               # Pydantic 모델
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── face_detection.py    # 얼굴 검출 API
│   │   └── health.py            # 헬스체크 API
│   └── utils/
│       ├── __init__.py
│       ├── image_processing.py  # OpenCV 유틸리티
│       └── video_processing.py  # 영상 처리 유틸리티
├── weights/                      # AI 모델 가중치
│   ├── yolov8n-face.pt
│   └── facenet_weights.pth
├── Dockerfile
├── requirements.txt
└── .env
```

### FastAPI 핵심 코드

#### `app/main.py`
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import torch
from app.models.model_loader import ModelLoader
from app.routers import face_detection, health
from app.config import settings

# 모델 전역 상태 관리
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 모델 로드, 종료 시 메모리 해제"""
    print("🚀 Loading AI models...")
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # 모델 로드
    models["face_detector"] = ModelLoader.load_face_detector(device)
    models["face_embedder"] = ModelLoader.load_face_embedder(device)
    
    print("✅ Models loaded successfully")
    yield
    
    # 앱 종료 시 메모리 해제
    print("🧹 Cleaning up resources...")
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# FastAPI 앱 생성
app = FastAPI(
    title="FaceBlur AI API",
    description="AI 기반 얼굴 검출 및 임베딩 추출 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (Django에서 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(
    face_detection.router, 
    prefix="/api/v1/faces", 
    tags=["Face Detection"]
)

# 모델 전역 접근을 위한 의존성
def get_models():
    return models

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,  # 프로덕션에서는 False
        workers=1      # GPU 메모리 고려하여 1개 워커 권장
    )
```

#### `app/routers/face_detection.py`
```python
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from typing import List
import numpy as np
import cv2
from app.schemas.face import FaceDetectionResponse, FaceBox
from app.main import get_models
import io

router = APIRouter()

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    models: dict = Depends(get_models)
):
    """
    이미지에서 얼굴을 검출합니다.
    
    - **file**: 이미지 파일 (JPEG, PNG)
    - **confidence_threshold**: 검출 신뢰도 임계값 (0.0-1.0)
    """
    try:
        # 업로드된 파일을 numpy 배열로 변환
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 얼굴 검출
        face_detector = models["face_detector"]
        results = face_detector(image)
        
        # 결과 파싱
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    faces.append(FaceBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=conf
                    ))
        
        return FaceDetectionResponse(
            success=True,
            face_count=len(faces),
            faces=faces,
            message=f"Detected {len(faces)} face(s)"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-batch", response_model=List[FaceDetectionResponse])
async def detect_faces_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5,
    models: dict = Depends(get_models)
):
    """
    여러 이미지에서 배치로 얼굴을 검출합니다.
    대량의 프레임 처리 시 사용.
    """
    results = []
    
    for file in files:
        try:
            result = await detect_faces(file, confidence_threshold, models)
            results.append(result)
        except Exception as e:
            results.append(FaceDetectionResponse(
                success=False,
                face_count=0,
                faces=[],
                message=f"Error: {str(e)}"
            ))
    
    return results

@router.post("/extract-embedding")
async def extract_face_embedding(
    file: UploadFile = File(...),
    models: dict = Depends(get_models)
):
    """
    얼굴 이미지에서 512차원 임베딩 벡터를 추출합니다.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # 임베딩 추출
        embedder = models["face_embedder"]
        embedding = embedder.extract(image)
        
        return {
            "success": True,
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### `app/models/face_detector.py`
```python
from ultralytics import YOLO
import torch
import numpy as np

class FaceDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        YOLOv8 기반 얼굴 검출 모델
        
        Args:
            model_path: YOLOv8 모델 가중치 경로
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        
    def __call__(self, image: np.ndarray, conf_threshold: float = 0.5):
        """
        이미지에서 얼굴 검출
        
        Args:
            image: OpenCV 이미지 (BGR)
            conf_threshold: 신뢰도 임계값
            
        Returns:
            YOLO 결과 객체
        """
        results = self.model.predict(
            image,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        return results
```

#### `app/models/face_embedder.py`
```python
import torch
import torch.nn as nn
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1

class FaceEmbedder:
    def __init__(self, device: str = "cuda"):
        """
        FaceNet 기반 얼굴 임베딩 추출 모델
        
        Args:
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.to(device)
        
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        얼굴 이미지를 모델 입력 형식으로 변환
        
        Args:
            face_image: 얼굴 이미지 (BGR, 임의 크기)
            
        Returns:
            전처리된 텐서 (1, 3, 160, 160)
        """
        # BGR → RGB 변환
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # 크기 조정 (160x160)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        # 정규화 [-1, 1]
        face_normalized = (face_resized - 127.5) / 128.0
        
        # (H, W, C) → (C, H, W)
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1)
        face_tensor = face_tensor.unsqueeze(0).float()
        
        return face_tensor.to(self.device)
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        얼굴 이미지에서 512차원 임베딩 벡터 추출
        
        Args:
            face_image: 얼굴 이미지
            
        Returns:
            임베딩 벡터 (512,)
        """
        with torch.no_grad():
            face_tensor = self.preprocess(face_image)
            embedding = self.model(face_tensor)
            return embedding.cpu().numpy().flatten()
```

#### `app/config.py`
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # FastAPI 설정
    APP_NAME: str = "FaceBlur AI API"
    DEBUG: bool = False
    
    # CORS 설정
    ALLOWED_ORIGINS: list = [
        "http://localhost:8000",
        "http://localhost:3000",
        "https://yourdomain.com"
    ]
    
    # 모델 경로
    YOLO_MODEL_PATH: str = "weights/yolov8n-face.pt"
    FACENET_MODEL_PATH: str = "weights/facenet_weights.pth"
    
    # GPU 설정
    USE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    
    # 이미지 처리 설정
    MAX_IMAGE_SIZE: int = 1920  # 최대 가로/세로 크기
    SUPPORTED_FORMATS: list = ["jpg", "jpeg", "png"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### `Dockerfile` (FastAPI)
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# 모델 가중치 다운로드 (별도 스크립트)
RUN python3 download_weights.py

# 포트 노출
EXPOSE 8001

# Uvicorn 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

---

## Django 웹 프레임워크

### Django 프로젝트 구조

```
django_web_server/
├── config/                       # Django 프로젝트 설정
│   ├── __init__.py
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py              # 공통 설정
│   │   ├── development.py       # 개발 환경
│   │   └── production.py        # 프로덕션 환경
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py                  # WebSocket 지원
├── apps/
│   ├── accounts/                # 사용자 관리
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   └── urls.py
│   ├── videos/                  # 영상 관리
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   ├── tasks.py            # Celery 태스크
│   │   └── urls.py
│   └── processing/              # 영상 처리
│       ├── models.py
│       ├── views.py
│       ├── services.py         # FastAPI 통신 로직
│       ├── tasks.py
│       └── urls.py
├── templates/                   # HTML 템플릿
│   ├── base.html
│   ├── home.html
│   ├── upload.html
│   └── processing.html
├── static/                      # 정적 파일
│   ├── css/
│   ├── js/
│   └── images/
├── media/                       # 업로드 파일 (개발용)
├── manage.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Django 핵심 코드

#### `apps/videos/models.py`
```python
from django.db import models
from django.contrib.auth.models import User
import uuid

class Video(models.Model):
    STATUS_CHOICES = [
        ('uploaded', '업로드 완료'),
        ('analyzing', '얼굴 분석 중'),
        ('ready', '처리 대기'),
        ('processing', '처리 중'),
        ('completed', '처리 완료'),
        ('failed', '실패'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='videos')
    
    # 파일 정보
    title = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    original_file_url = models.URLField()  # S3 URL
    processed_file_url = models.URLField(blank=True, null=True)
    
    # 영상 메타데이터
    duration = models.FloatField(help_text="영상 길이 (초)")
    width = models.IntegerField()
    height = models.IntegerField()
    fps = models.FloatField(help_text="프레임 레이트")
    file_size = models.BigIntegerField(help_text="파일 크기 (bytes)")
    
    # 처리 상태
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    progress = models.IntegerField(default=0, help_text="처리 진행률 (0-100)")
    error_message = models.TextField(blank=True, null=True)
    
    # 타임스탬프
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True, help_text="처리 완료 시각")
    expires_at = models.DateTimeField(blank=True, null=True, help_text="파일 만료 시각 (7일 후)")
    
    class Meta:
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.title} ({self.user.username})"
    
    def save(self, *args, **kwargs):
        # 완료 시 만료일 자동 설정 (7일 후)
        if self.status == 'completed' and not self.expires_at:
            from datetime import timedelta
            self.expires_at = timezone.now() + timedelta(days=7)
        super().save(*args, **kwargs)

class Face(models.Model):
    """영상에서 발견된 고유 얼굴"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='faces')
    
    # 얼굴 정보
    face_index = models.IntegerField(help_text="얼굴 순번 (1부터 시작)")
    thumbnail_url = models.URLField(help_text="대표 얼굴 이미지 URL")
    embedding = models.JSONField(help_text="512차원 임베딩 벡터")
    
    # 통계
    appearance_count = models.IntegerField(default=0, help_text="영상 내 등장 횟수")
    first_frame = models.IntegerField(help_text="첫 등장 프레임 번호")
    last_frame = models.IntegerField(help_text="마지막 등장 프레임 번호")
    
    # 사용자 선택
    is_blurred = models.BooleanField(default=True, help_text="블러 적용 여부")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['face_index']
        unique_together = ['video', 'face_index']
    
    def __str__(self):
        return f"Face {self.face_index} in {self.video.title}"

class ProcessingJob(models.Model):
    """Celery 작업 추적"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    
    job_type = models.CharField(max_length=50, choices=[
        ('face_analysis', '얼굴 분석'),
        ('video_processing', '영상 처리'),
    ])
    
    celery_task_id = models.CharField(max_length=255, unique=True)
    status = models.CharField(max_length=20, default='pending')
    progress = models.IntegerField(default=0)
    result_data = models.JSONField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.job_type} - {self.video.title}"
```

#### `apps/processing/services.py` (FastAPI 통신)
```python
import httpx
import asyncio
from django.conf import settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FastAPIClient:
    """FastAPI 서버와 통신하는 클라이언트"""
    
    def __init__(self):
        self.base_url = settings.FASTAPI_BASE_URL
        self.timeout = httpx.Timeout(30.0, connect=10.0)
    
    async def detect_faces(self, image_bytes: bytes) -> Dict:
        """
        이미지에서 얼굴 검출
        
        Args:
            image_bytes: 이미지 바이너리 데이터
            
        Returns:
            {
                "success": True,
                "face_count": 2,
                "faces": [
                    {"x1": 100, "y1": 50, "x2": 200, "y2": 150, "confidence": 0.95},
                    ...
                ]
            }
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                response = await client.post(
                    f"{self.base_url}/api/v1/faces/detect",
                    files=files
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"FastAPI 얼굴 검출 실패: {e}")
                raise
    
    async def detect_faces_batch(self, images_bytes: List[bytes]) -> List[Dict]:
        """
        배치 얼굴 검출 (여러 이미지 동시 처리)
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                files = [
                    ("files", (f"frame_{i}.jpg", img, "image/jpeg"))
                    for i, img in enumerate(images_bytes)
                ]
                response = await client.post(
                    f"{self.base_url}/api/v1/faces/detect-batch",
                    files=files
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"FastAPI 배치 검출 실패: {e}")
                raise
    
    async def extract_embedding(self, face_image_bytes: bytes) -> List[float]:
        """
        얼굴 이미지에서 임베딩 벡터 추출
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                files = {"file": ("face.jpg", face_image_bytes, "image/jpeg")}
                response = await client.post(
                    f"{self.base_url}/api/v1/faces/extract-embedding",
                    files=files
                )
                response.raise_for_status()
                result = response.json()
                return result["embedding"]
            except httpx.HTTPError as e:
                logger.error(f"FastAPI 임베딩 추출 실패: {e}")
                raise

# 싱글톤 인스턴스
fastapi_client = FastAPIClient()
```

#### `apps/videos/tasks.py` (Celery 태스크)
```python
from celery import shared_task
from django.core.files.storage import default_storage
from apps.videos.models import Video, Face, ProcessingJob
from apps.processing.services import fastapi_client
import cv2
import numpy as np
import asyncio
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def analyze_faces_task(self, video_id: str):
    """
    영상 전체를 분석하여 고유 얼굴 추출
    
    Phase 1: 초기 얼굴 추출
    """
    try:
        video = Video.objects.get(id=video_id)
        video.status = 'analyzing'
        video.save()
        
        # S3에서 영상 다운로드
        video_path = download_from_s3(video.original_file_url)
        
        # 1. 프레임 샘플링 (매 30프레임마다)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = 30
        
        sampled_frames = []
        frame_numbers = []
        
        for frame_num in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                sampled_frames.append(frame)
                frame_numbers.append(frame_num)
            
            # 진행률 업데이트
            progress = int((frame_num / total_frames) * 30)  # 0-30%
            self.update_state(state='PROGRESS', meta={'progress': progress})
        
        cap.release()
        
        # 2. FastAPI로 얼굴 검출 (배치 처리)
        batch_size = 10
        all_face_data = []
        
        for i in range(0, len(sampled_frames), batch_size):
            batch = sampled_frames[i:i+batch_size]
            batch_bytes = [cv2.imencode('.jpg', frame)[1].tobytes() for frame in batch]
            
            # 비동기 호출을 동기로 변환
            results = asyncio.run(fastapi_client.detect_faces_batch(batch_bytes))
            
            for j, result in enumerate(results):
                if result['success'] and result['face_count'] > 0:
                    frame_idx = i + j
                    all_face_data.append({
                        'frame_number': frame_numbers[frame_idx],
                        'frame': sampled_frames[frame_idx],
                        'faces': result['faces']
                    })
            
            progress = 30 + int((i / len(sampled_frames)) * 30)  # 30-60%
            self.update_state(state='PROGRESS', meta={'progress': progress})
        
        # 3. 얼굴 임베딩 추출
        embeddings = []
        embedding_metadata = []
        
        for data in all_face_data:
            frame = data['frame']
            for face in data['faces']:
                x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    face_bytes = cv2.imencode('.jpg', face_img)[1].tobytes()
                    embedding = asyncio.run(fastapi_client.extract_embedding(face_bytes))
                    
                    embeddings.append(embedding)
                    embedding_metadata.append({
                        'frame_number': data['frame_number'],
                        'bbox': (x1, y1, x2, y2),
                        'face_img': face_img
                    })
        
        progress = 70
        self.update_state(state='PROGRESS', meta={'progress': progress})
        
        # 4. 얼굴 클러스터링 (DBSCAN)
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings_array)
        
        # 5. 각 클러스터(고유 얼굴)의 대표 이미지 선택
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 노이즈 제거
        
        for face_index, label in enumerate(sorted(unique_labels), start=1):
            cluster_indices = np.where(labels == label)[0]
            
            # 가장 선명한 얼굴 선택 (면적이 큰 것)
            best_idx = max(
                cluster_indices,
                key=lambda idx: (
                    embedding_metadata[idx]['bbox'][2] - embedding_metadata[idx]['bbox'][0]
                ) * (
                    embedding_metadata[idx]['bbox'][3] - embedding_metadata[idx]['bbox'][1]
                )
            )
            
            best_face = embedding_metadata[best_idx]
            
            # S3에 썸네일 업로드
            thumbnail_url = upload_thumbnail_to_s3(
                best_face['face_img'],
                video_id,
                face_index
            )
            
            # Face 모델 생성
            Face.objects.create(
                video=video,
                face_index=face_index,
                thumbnail_url=thumbnail_url,
                embedding=embeddings[best_idx],
                appearance_count=len(cluster_indices),
                first_frame=min([embedding_metadata[i]['frame_number'] for i in cluster_indices]),
                last_frame=max([embedding_metadata[i]['frame_number'] for i in cluster_indices]),
                is_blurred=True  # 기본값: 블러 적용
            )
        
        # 완료
        video.status = 'ready'
        video.progress = 100
        video.save()
        
        self.update_state(state='SUCCESS', meta={'progress': 100})
        return {
            'video_id': str(video_id),
            'face_count': len(unique_labels),
            'status': 'completed'
        }
    
    except Exception as e:
        logger.error(f"얼굴 분석 실패: {e}")
        video.status = 'failed'
        video.error_message = str(e)
        video.save()
        raise

@shared_task(bind=True)
def process_video_task(self, video_id: str):
    """
    선택된 얼굴만 블러 처리하여 영상 생성
    
    Phase 2: 전체 영상 처리
    """
    try:
        video = Video.objects.get(id=video_id)
        video.status = 'processing'
        video.save()
        
        # 블러 처리할 얼굴 목록 가져오기
        faces_to_blur = video.faces.filter(is_blurred=True)
        
        # S3에서 영상 다운로드
        video_path = download_from_s3(video.original_file_url)
        
        # 영상 처리 로직 (생략 - 매우 길어서 개요만)
        # 1. 전체 프레임 순회
        # 2. FastAPI로 얼굴 검출
        # 3. 검출된 얼굴과 faces_to_blur 임베딩 비교 (코사인 유사도)
        # 4. 매칭되는 얼굴만 OpenCV로 블러 처리
        # 5. FFmpeg로 영상 재인코딩
        # 6. S3 업로드
        
        # 처리 완료
        video.status = 'completed'
        video.processed_file_url = processed_video_url
        video.progress = 100
        video.save()
        
        return {'video_id': str(video_id), 'status': 'completed'}
    
    except Exception as e:
        logger.error(f"영상 처리 실패: {e}")
        video.status = 'failed'
        video.error_message = str(e)
        video.save()
        raise
```

#### `config/settings/production.py`
```python
from .base import *

DEBUG = False

ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# FastAPI 서버 URL
FASTAPI_BASE_URL = "http://fastapi-server:8001"  # Docker 내부 통신
# 또는
# FASTAPI_BASE_URL = "http://internal-lb-fastapi.amazonaws.com"  # ALB 사용 시

# AWS S3 설정
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_STORAGE_BUCKET_NAME = 'faceblur-videos'
AWS_S3_REGION_NAME = 'ap-northeast-2'
AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com'

# 미디어 파일을 S3에 저장
DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/'

# PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'faceblur'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'db'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Redis (Celery + 캐시)
REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Celery
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'Asia/Seoul'
```

---

## AWS 인프라 구성

### AWS 리소스 목록

```
┌─────────────────────────────────────────────────────────┐
│                     AWS Services                         │
├─────────────────────────────────────────────────────────┤
│ ✓ VPC (Virtual Private Cloud)                           │
│ ✓ EC2 (Django 웹 서버) - t3.large                        │
│ ✓ EC2 (FastAPI AI 서버) - g4dn.xlarge (GPU)             │
│ ✓ EC2 (Celery Worker) - t3.xlarge                       │
│ ✓ RDS (PostgreSQL)                                      │
│ ✓ ElastiCache (Redis)                                   │
│ ✓ S3 (영상 스토리지)                                     │
│ ✓ Application Load Balancer                             │
│ ✓ Route 53 (DNS)                                        │
│ ✓ CloudFront (CDN, 선택사항)                             │
│ ✓ Certificate Manager (SSL/TLS)                         │
│ ✓ CloudWatch (모니터링)                                 │
│ ✓ IAM (권한 관리)                                        │
└─────────────────────────────────────────────────────────┘
```

### EC2 인스턴스 사양

#### Django 웹 서버 (t3.large)
```
- vCPU: 2
- RAM: 8GB
- 스토리지: 50GB (gp3)
- 용도: Django 애플리케이션, Nginx, Gunicorn
- 예상 비용: ~$70/월
```

#### FastAPI AI 서버 (g4dn.xlarge)
```
- vCPU: 4
- RAM: 16GB
- GPU: NVIDIA T4 (16GB VRAM)
- 스토리지: 100GB (gp3)
- 용도: AI 모델 추론
- 예상 비용: ~$400/월
```

#### Celery Worker (t3.xlarge)
```
- vCPU: 4
- RAM: 16GB
- 스토리지: 100GB (gp3)
- 용도: 영상 처리, 프레임 추출
- 예상 비용: ~$140/월
```

### S3 버킷 구조

```
faceblur-videos/
├── original/                # 원본 영상
│   └── {user_id}/
│       └── {video_id}.mp4
├── processed/              # 처리 완료 영상
│   └── {user_id}/
│       └── {video_id}_processed.mp4
├── thumbnails/             # 얼굴 썸네일
│   └── {video_id}/
│       ├── face_1.jpg
│       ├── face_2.jpg
│       └── ...
└── temp/                   # 임시 프레임 (일시적)
    └── {job_id}/
        ├── frame_0001.jpg
        ├── frame_0002.jpg
        └── ...
```

### 보안 그룹 규칙

#### Django 웹 서버 SG
```
Inbound:
- Port 80 (HTTP) from ALB
- Port 443 (HTTPS) from ALB
- Port 22 (SSH) from 관리자 IP

Outbound:
- All traffic
```

#### FastAPI AI 서버 SG
```
Inbound:
- Port 8001 from Django SG
- Port 22 (SSH) from 관리자 IP

Outbound:
- All traffic
```

#### RDS SG
```
Inbound:
- Port 5432 from Django SG, Celery SG

Outbound:
- N/A
```

### Terraform 인프라 코드 예시

```hcl
# main.tf

provider "aws" {
  region = "ap-northeast-2"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "faceblur-vpc"
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "faceblur-public-${count.index + 1}"
  }
}

# Private Subnet
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "faceblur-private-${count.index + 1}"
  }
}

# EC2 - Django Web Server
resource "aws_instance" "django_web" {
  ami           = "ami-0c9c942bd7bf113a2"  # Ubuntu 22.04 LTS
  instance_type = "t3.large"
  subnet_id     = aws_subnet.public[0].id
  
  vpc_security_group_ids = [aws_security_group.django_sg.id]
  
  user_data = file("user_data/django_setup.sh")
  
  tags = {
    Name = "faceblur-django-web"
  }
}

# EC2 - FastAPI AI Server
resource "aws_instance" "fastapi_ai" {
  ami           = "ami-gpu-ubuntu-22.04"  # GPU 지원 AMI
  instance_type = "g4dn.xlarge"
  subnet_id     = aws_subnet.private[0].id
  
  vpc_security_group_ids = [aws_security_group.fastapi_sg.id]
  
  user_data = file("user_data/fastapi_setup.sh")
  
  tags = {
    Name = "faceblur-fastapi-ai"
  }
}

# RDS - PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier           = "faceblur-db"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.medium"
  allocated_storage    = 100
  storage_type         = "gp3"
  
  db_name  = "faceblur"
  username = "admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  skip_final_snapshot = false
  final_snapshot_identifier = "faceblur-db-final-snapshot"
  
  backup_retention_period = 7
  
  tags = {
    Name = "faceblur-postgres"
  }
}

# ElastiCache - Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "faceblur-redis"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis_sg.id]
  
  tags = {
    Name = "faceblur-redis"
  }
}

# S3 Bucket
resource "aws_s3_bucket" "videos" {
  bucket = "faceblur-videos"
  
  tags = {
    Name = "faceblur-videos"
  }
}

resource "aws_s3_bucket_versioning" "videos" {
  bucket = aws_s3_bucket.videos.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "videos" {
  bucket = aws_s3_bucket.videos.id

  rule {
    id     = "delete-temp-files"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 1
    }
  }
}
```

---

## 데이터베이스 설계

### ERD (Entity-Relationship Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        auth_user                             │
├─────────────────────────────────────────────────────────────┤
│ id (PK)                                                      │
│ username                                                     │
│ email                                                        │
│ password                                                     │
│ date_joined                                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ 1:N
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                        videos_video                          │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                               │
│ user_id (FK → auth_user.id)                                │
│ title                                                       │
│ original_filename                                           │
│ original_file_url                                           │
│ processed_file_url                                          │
│ duration                                                    │
│ width, height, fps                                          │
│ file_size                                                   │
│ status (uploaded/analyzing/ready/processing/completed)      │
│ progress (0-100)                                            │
│ error_message                                               │
│ created_at, updated_at                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ 1:N
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       videos_face                            │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                               │
│ video_id (FK → videos_video.id)                            │
│ face_index (1, 2, 3, ...)                                  │
│ thumbnail_url                                               │
│ embedding (JSON, 512 floats)                               │
│ appearance_count                                            │
│ first_frame, last_frame                                     │
│ is_blurred (Boolean)                                        │
│ created_at                                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   processing_processingjob                   │
├─────────────────────────────────────────────────────────────┤
│ id (PK, UUID)                                               │
│ video_id (FK → videos_video.id)                            │
│ job_type (face_analysis / video_processing)                 │
│ celery_task_id                                              │
│ status (pending/running/completed/failed)                   │
│ progress (0-100)                                            │
│ result_data (JSON)                                          │
│ error_message                                               │
│ started_at, completed_at                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 배포 전략

### Docker Compose (개발 환경)

```yaml
# docker-compose.yml

version: '3.9'

services:
  # PostgreSQL
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: faceblur
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Django Web Server
  django:
    build:
      context: ./django_web_server
      dockerfile: Dockerfile
    command: >
      sh -c "python manage.py migrate &&
             python manage.py collectstatic --noinput &&
             gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 4"
    volumes:
      - ./django_web_server:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    environment:
      - DEBUG=True
      - DATABASE_URL=postgresql://postgres:postgres123@db:5432/faceblur
      - REDIS_URL=redis://redis:6379/0
      - FASTAPI_BASE_URL=http://fastapi:8001
    depends_on:
      - db
      - redis

  # FastAPI AI Server
  fastapi:
    build:
      context: ./fastapi_ai_server
      dockerfile: Dockerfile
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
    volumes:
      - ./fastapi_ai_server:/app
      - model_weights:/app/weights
    ports:
      - "8001:8001"
    environment:
      - USE_GPU=False  # 개발 환경에서는 CPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]  # GPU 사용 시

  # Celery Worker
  celery_worker:
    build:
      context: ./django_web_server
      dockerfile: Dockerfile
    command: celery -A config worker -l info -Q default,video_processing
    volumes:
      - ./django_web_server:/app
      - media_volume:/app/media
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@db:5432/faceblur
      - REDIS_URL=redis://redis:6379/0
      - FASTAPI_BASE_URL=http://fastapi:8001
    depends_on:
      - db
      - redis
      - django

  # Celery Beat (스케줄러, 선택사항)
  celery_beat:
    build:
      context: ./django_web_server
      dockerfile: Dockerfile
    command: celery -A config beat -l info
    volumes:
      - ./django_web_server:/app
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@db:5432/faceblur
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  # Nginx
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/static
      - media_volume:/media
    ports:
      - "80:80"
    depends_on:
      - django

volumes:
  postgres_data:
  static_volume:
  media_volume:
  model_weights:
```

### 프로덕션 배포 단계

#### 1단계: AWS 인프라 구축
```bash
# Terraform으로 인프라 프로비저닝
cd terraform
terraform init
terraform plan
terraform apply

# 출력: EC2 인스턴스 IP, RDS 엔드포인트, S3 버킷 이름 등
```

#### 2단계: EC2 인스턴스 설정

**Django 웹 서버 (EC2)**
```bash
# SSH 접속
ssh -i your-key.pem ubuntu@<django-ec2-public-ip>

# Docker 설치
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# 코드 배포
git clone https://github.com/yourrepo/faceblur.git
cd faceblur/django_web_server

# 환경 변수 설정
cp .env.example .env
nano .env  # AWS RDS, S3 정보 입력

# Docker 이미지 빌드 및 실행
docker build -t faceblur-django .
docker run -d -p 8000:8000 --env-file .env faceblur-django
```

**FastAPI AI 서버 (EC2 with GPU)**
```bash
# SSH 접속
ssh -i your-key.pem ubuntu@<fastapi-ec2-public-ip>

# NVIDIA 드라이버 설치
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 코드 배포
git clone https://github.com/yourrepo/faceblur.git
cd faceblur/fastapi_ai_server

# Docker 이미지 빌드 및 실행
docker build -t faceblur-fastapi .
docker run -d --gpus all -p 8001:8001 faceblur-fastapi

# GPU 동작 확인
docker exec -it <container_id> nvidia-smi
```

#### 3단계: Nginx 리버스 프록시 설정

```nginx
# /etc/nginx/sites-available/faceblur

upstream django_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    client_max_body_size 500M;  # 대용량 영상 업로드

    location / {
        proxy_pass http://django_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /var/www/static/;
    }

    location /media/ {
        alias /var/www/media/;
    }

    # WebSocket 지원 (실시간 진행률)
    location /ws/ {
        proxy_pass http://django_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# SSL 설정 (Let's Encrypt)
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # (위의 location 블록들 동일)
}
```

#### 4단계: CI/CD 파이프라인 (GitHub Actions)

```yaml
# .github/workflows/deploy.yml

name: Deploy to Production

on:
  push:
    branches:
      - main

jobs:
  deploy-django:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          cd django_web_server
          docker build -t faceblur-django:${{ github.sha }} .
      
      - name: Push to ECR
        run: |
          aws ecr get-login-password --region ap-northeast-2 | \
            docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
          docker tag faceblur-django:${{ github.sha }} ${{ secrets.ECR_REGISTRY }}/faceblur-django:latest
          docker push ${{ secrets.ECR_REGISTRY }}/faceblur-django:latest
      
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            docker pull ${{ secrets.ECR_REGISTRY }}/faceblur-django:latest
            docker stop faceblur-django || true
            docker rm faceblur-django || true
            docker run -d --name faceblur-django -p 8000:8000 \
              --env-file /home/ubuntu/.env \
              ${{ secrets.ECR_REGISTRY }}/faceblur-django:latest

  deploy-fastapi:
    runs-on: ubuntu-latest
    steps:
      # (동일한 패턴으로 FastAPI 배포)
```

---

## 보안 및 성능 최적화

### 보안 체크리스트

✅ **인증 및 권한**
- Django User 모델 사용
- JWT 토큰 인증 (선택사항)
- CSRF 보호 활성화
- CORS 설정 제한

✅ **데이터 보호**
- S3 버킷 비공개 설정
- Presigned URL로 임시 접근 권한 부여
- 민감 정보 환경 변수 저장 (.env)
- SSL/TLS 인증서 (Let's Encrypt)

✅ **API 보안**
- FastAPI와 Django 간 내부 통신만 허용 (Security Group)
- Rate Limiting (Django Ratelimit)
- 요청 크기 제한 (client_max_body_size)

✅ **서버 보안**
- SSH 키 기반 인증
- 불필요한 포트 차단
- 정기적인 보안 패치

### 성능 최적화 전략

#### 1. AI 모델 최적화
```python
# TorchScript로 모델 컴파일 (추론 속도 향상)
import torch

model = load_model()
model.eval()

example_input = torch.randn(1, 3, 160, 160).cuda()
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_optimized.pt")
```

#### 2. 캐싱 전략
```python
# Django 뷰에서 Redis 캐시 활용
from django.core.cache import cache

@api_view(['GET'])
def get_video_faces(request, video_id):
    cache_key = f"video_{video_id}_faces"
    
    # 캐시에서 조회
    cached_data = cache.get(cache_key)
    if cached_data:
        return Response(cached_data)
    
    # DB 조회
    faces = Face.objects.filter(video_id=video_id)
    data = FaceSerializer(faces, many=True).data
    
    # 캐시 저장 (10분)
    cache.set(cache_key, data, timeout=600)
    
    return Response(data)
```

#### 3. 프레임 처리 병렬화
```python
from concurrent.futures import ThreadPoolExecutor

def process_frames_parallel(frames, batch_size=10):
    """여러 프레임을 병렬로 처리"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            future = executor.submit(process_batch, batch)
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    return results
```

#### 4. S3 전송 최적화
```python
import boto3
from boto3.s3.transfer import TransferConfig

# 멀티파트 업로드로 대용량 파일 빠르게 전송
config = TransferConfig(
    multipart_threshold=1024 * 25,  # 25MB 이상이면 멀티파트
    max_concurrency=10,
    multipart_chunksize=1024 * 25,
    use_threads=True
)

s3_client = boto3.client('s3')
s3_client.upload_file(
    'large_video.mp4',
    'faceblur-videos',
    'processed/video.mp4',
    Config=config
)
```

#### 5. 데이터베이스 쿼리 최적화
```python
# N+1 문제 해결: select_related, prefetch_related 사용

# ❌ 나쁜 예 (N+1 쿼리)
videos = Video.objects.all()
for video in videos:
    print(video.user.username)  # 매번 DB 쿼리!

# ✅ 좋은 예 (1번의 JOIN 쿼리)
videos = Video.objects.select_related('user').all()
for video in videos:
    print(video.user.username)  # 이미 로드됨

# ✅ Face 관계도 함께 로드
videos = Video.objects.prefetch_related('faces').all()
```

---

## 모니터링 및 로깅

### CloudWatch 설정
```python
# Django settings.py에 로깅 설정

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': '/var/log/django/app.log',
        },
        'cloudwatch': {
            'level': 'INFO',
            'class': 'watchtower.CloudWatchLogHandler',
            'log_group': 'faceblur-django',
            'stream_name': 'production',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'cloudwatch'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

### 성능 메트릭 수집
```python
# Celery 태스크 실행 시간 측정
import time
from django.core.cache import cache

@shared_task
def process_video_task(video_id):
    start_time = time.time()
    
    try:
        # 영상 처리 로직
        pass
    finally:
        duration = time.time() - start_time
        
        # 메트릭 저장
        cache.set(
            f"metrics:process_time:{video_id}",
            duration,
            timeout=86400  # 24시간
        )
```

---

## 마무리

이 SKILL.md는 FaceBlur 프로젝트의 기술적 구현을 위한 포괄적인 가이드입니다. 실제 개발 시 다음 순서로 진행하는 것을 권장합니다:

1. **개발 환경 구축** (Docker Compose)
2. **FastAPI AI 모델 서빙 구현**
3. **Django 웹 애플리케이션 개발**
4. **Celery 비동기 작업 구현**
5. **로컬 테스트 및 디버깅**
6. **AWS 인프라 구축 (Terraform)**
7. **프로덕션 배포**
8. **모니터링 및 최적화**

각 단계에서 이 문서를 참고하여 모범 사례를 따르세요.
