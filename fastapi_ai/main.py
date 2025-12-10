"""
FastAPI AI Service - Main Application
======================================

AI 모델 추론 전용 REST API 서버
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import logging
import sys
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Face Blur AI Service",
    description="AI model inference service for face detection, recognition, and quality assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정 (Django와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 Django URL만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 모델 전역 변수 (서버 시작 시 로드)
ai_models = {
    "face_detector": None,
    "embedding_extractor": None,
    "quality_assessor": None
}


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    models_loaded: Dict[str, bool]
    cuda_version: Optional[str] = None


class FaceDetectionRequest(BaseModel):
    video_path: str
    frame_indices: List[int]  # 처리할 프레임 인덱스 리스트


class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: Optional[int] = None


class FaceDetectionResponse(BaseModel):
    frame_idx: int
    faces: List[FaceBox]
    processing_time: float


class EmbeddingRequest(BaseModel):
    """임베딩 추출 요청 (이미지 바이너리)"""
    pass  # UploadFile로 받음


class EmbeddingResponse(BaseModel):
    embedding: List[float]  # 512-dim
    quality: float
    processing_time: float


class VideoAnalysisRequest(BaseModel):
    video_path: str
    start_frame: int = 0
    end_frame: Optional[int] = None
    skip_frames: int = 1  # 프레임 스킵 (기본 1 = 모든 프레임)


class VideoAnalysisResponse(BaseModel):
    total_frames: int
    processed_frames: int
    faces_per_frame: List[Dict[str, Any]]  # [{frame_idx: int, faces: [...]}]
    total_processing_time: float


# ============================================================================
# Startup & Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 AI 모델 로드
    """
    logger.info("=" * 80)
    logger.info("Starting FastAPI AI Service...")
    logger.info("=" * 80)

    # GPU 확인
    if torch.cuda.is_available():
        logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("GPU not available, using CPU")

    try:
        # ai_pipeline에서 모델 로더 임포트
        from .ai_pipeline import load_models

        logger.info("Loading AI models...")
        global ai_models
        ai_models = load_models(device='cuda' if torch.cuda.is_available() else 'cpu')

        logger.info("✅ All AI models loaded successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Failed to load AI models: {e}")
        logger.error("Server will start but AI endpoints will not work!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    서버 종료 시 정리
    """
    logger.info("Shutting down FastAPI AI Service...")

    # CUDA 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 상태 및 모델 로딩 확인
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    cuda_version = torch.version.cuda if gpu_available else None

    models_loaded = {
        "yolo_face": ai_models.get("face_detector") is not None,
        "adaface": ai_models.get("embedding_extractor") is not None,
        "nima": ai_models.get("quality_assessor") is not None
    }

    all_models_loaded = all(models_loaded.values())
    status = "healthy" if all_models_loaded else "degraded"

    return HealthResponse(
        status=status,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        models_loaded=models_loaded,
        cuda_version=cuda_version
    )


# ============================================================================
# AI Inference Endpoints
# ============================================================================

@app.post("/detect_faces", response_model=List[FaceDetectionResponse])
async def detect_faces(request: FaceDetectionRequest):
    """
    YOLO v11s 얼굴 감지

    Parameters:
    -----------
    video_path : str
        비디오 파일 경로
    frame_indices : List[int]
        처리할 프레임 인덱스 리스트

    Returns:
    --------
    List[FaceDetectionResponse]
        각 프레임의 감지된 얼굴 정보
    """
    if ai_models["face_detector"] is None:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    try:
        from .ai_pipeline import detect_faces_in_frames

        results = detect_faces_in_frames(
            ai_models["face_detector"],
            request.video_path,
            request.frame_indices
        )

        return results

    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_embedding", response_model=EmbeddingResponse)
async def extract_embedding(face_image: UploadFile = File(...)):
    """
    AdaFace 임베딩 추출

    Parameters:
    -----------
    face_image : UploadFile
        얼굴 이미지 파일 (JPEG, PNG)

    Returns:
    --------
    EmbeddingResponse
        512-dim 임베딩 벡터 및 품질 점수
    """
    if ai_models["embedding_extractor"] is None or ai_models["quality_assessor"] is None:
        raise HTTPException(status_code=503, detail="Embedding/Quality models not loaded")

    try:
        import io
        import cv2
        import numpy as np
        from .ai_pipeline import extract_embedding_and_quality

        # 이미지 파일 읽기
        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # 임베딩 및 품질 추출
        result = extract_embedding_and_quality(
            ai_models["embedding_extractor"],
            ai_models["quality_assessor"],
            image
        )

        return result

    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    전체 비디오 분석 (얼굴 감지 + 임베딩 + 품질 평가)

    Parameters:
    -----------
    video_path : str
        비디오 파일 경로
    start_frame : int
        시작 프레임 (기본 0)
    end_frame : Optional[int]
        종료 프레임 (None이면 끝까지)
    skip_frames : int
        프레임 스킵 간격 (기본 1)

    Returns:
    --------
    VideoAnalysisResponse
        전체 프레임의 얼굴 정보, 임베딩, 품질 데이터
    """
    if any(model is None for model in ai_models.values()):
        raise HTTPException(status_code=503, detail="Not all AI models loaded")

    try:
        from .ai_pipeline import analyze_video_full

        result = analyze_video_full(
            face_detector=ai_models["face_detector"],
            embedding_extractor=ai_models["embedding_extractor"],
            quality_assessor=ai_models["quality_assessor"],
            video_path=request.video_path,
            start_frame=request.start_frame,
            end_frame=request.end_frame,
            skip_frames=request.skip_frames
        )

        return result

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """
    API 정보
    """
    return {
        "service": "Face Blur AI Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
