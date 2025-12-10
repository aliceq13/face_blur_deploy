"""
AI Pipeline for FastAPI Service
================================

기존 Django 코드를 재사용하여 AI 모델 추론 수행
- YOLO v11s Face Detection
- AdaFace ViT Embedding Extraction
- NIMA Quality Assessment
"""

import cv2
import numpy as np
import torch
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 기존 Django 모듈 임포트 (Django ORM 제외)
from ultralytics import YOLO
from apps.videos.adaface_wrapper import AdaFaceWrapper
from apps.videos.maniqa_wrapper import MANIQAWrapper
from apps.videos.face_aligner import align_face

logger = logging.getLogger(__name__)


# ============================================================================
# 모델 로딩
# ============================================================================

def load_models(device: str = 'cuda') -> Dict[str, Any]:
    """
    모든 AI 모델 로드

    Parameters:
    -----------
    device : str
        'cuda' or 'cpu'

    Returns:
    --------
    models : dict
        {
            'face_detector': YOLO model,
            'embedding_extractor': AdaFaceWrapper,
            'quality_assessor': MANIQAWrapper
        }
    """
    logger.info(f"Loading AI models on device: {device}")

    models = {}

    try:
        # 1. YOLO Face Detector
        yolo_path = project_root / 'models' / 'yolov11s-face.pt'
        logger.info(f"Loading YOLO from: {yolo_path}")
        models['face_detector'] = YOLO(str(yolo_path))
        logger.info("✅ YOLO Face model loaded")

        # 2. AdaFace Embedding Extractor
        adaface_path = project_root / 'apps' / 'videos' / 'weights' / 'adaface_vit_base_kprpe_webface12m.pt'
        logger.info(f"Loading AdaFace from: {adaface_path}")
        models['embedding_extractor'] = AdaFaceWrapper(
            model_path=str(adaface_path),
            architecture='vit',
            device=device
        )
        logger.info("✅ AdaFace model loaded")

        # 3. NIMA Quality Assessor
        logger.info("Loading NIMA quality assessor...")
        models['quality_assessor'] = MANIQAWrapper(
            device=device,
            batch_size=8,
            model_name='nima'  # MANIQA 대신 NIMA 사용 (15배 빠름)
        )
        logger.info("✅ NIMA quality assessor loaded")

        return models

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


# ============================================================================
# 얼굴 감지 (YOLO)
# ============================================================================

def detect_faces_in_frames(
    face_detector: YOLO,
    video_path: str,
    frame_indices: List[int]
) -> List[Dict[str, Any]]:
    """
    지정된 프레임들에서 얼굴 감지

    Parameters:
    -----------
    face_detector : YOLO
        YOLO 얼굴 감지 모델
    video_path : str
        비디오 파일 경로
    frame_indices : List[int]
        처리할 프레임 인덱스 리스트

    Returns:
    --------
    results : List[Dict]
        [
            {
                'frame_idx': int,
                'faces': [{'x1', 'y1', 'x2', 'y2', 'confidence', 'track_id'}, ...],
                'processing_time': float
            }
        ]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    results = []

    try:
        for frame_idx in frame_indices:
            start_time = time.time()

            # 프레임 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Cannot read frame {frame_idx}")
                continue

            # YOLO 추론 (tracking 활성화)
            yolo_results = face_detector.track(
                frame,
                persist=True,
                conf=0.5,
                iou=0.45,
                verbose=False
            )

            # 결과 파싱
            faces = []
            if yolo_results and yolo_results[0].boxes is not None:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                    faces.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': conf,
                        'track_id': track_id
                    })

            processing_time = time.time() - start_time

            results.append({
                'frame_idx': frame_idx,
                'faces': faces,
                'processing_time': processing_time
            })

    finally:
        cap.release()

    return results


# ============================================================================
# 임베딩 추출 + 품질 평가
# ============================================================================

def extract_embedding_and_quality(
    embedding_extractor: AdaFaceWrapper,
    quality_assessor: MANIQAWrapper,
    face_image: np.ndarray
) -> Dict[str, Any]:
    """
    얼굴 이미지에서 임베딩 및 품질 추출

    Parameters:
    -----------
    embedding_extractor : AdaFaceWrapper
        AdaFace 모델
    quality_assessor : MANIQAWrapper
        NIMA 품질 평가 모델
    face_image : np.ndarray
        얼굴 이미지 (BGR)

    Returns:
    --------
    result : dict
        {
            'embedding': List[float],  # 512-dim
            'quality': float,
            'processing_time': float
        }
    """
    start_time = time.time()

    try:
        # 1. Face Alignment
        aligned_face = align_face(face_image)
        if aligned_face is None:
            raise ValueError("Face alignment failed")

        # 2. Embedding Extraction (AdaFace)
        embedding = embedding_extractor.get_embedding(aligned_face)
        if embedding is None:
            raise ValueError("Embedding extraction failed")

        # 3. Quality Assessment (NIMA)
        quality = quality_assessor.assess_quality(aligned_face)

        processing_time = time.time() - start_time

        return {
            'embedding': embedding.tolist(),  # numpy array → list
            'quality': float(quality),
            'processing_time': processing_time
        }

    except Exception as e:
        logger.error(f"Embedding/Quality extraction error: {e}")
        raise


# ============================================================================
# 전체 비디오 분석
# ============================================================================

def analyze_video_full(
    face_detector: YOLO,
    embedding_extractor: AdaFaceWrapper,
    quality_assessor: MANIQAWrapper,
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    skip_frames: int = 1
) -> Dict[str, Any]:
    """
    비디오 전체 분석 (얼굴 감지 + 임베딩 + 품질)

    Parameters:
    -----------
    face_detector : YOLO
    embedding_extractor : AdaFaceWrapper
    quality_assessor : MANIQAWrapper
    video_path : str
    start_frame : int
    end_frame : Optional[int]
    skip_frames : int

    Returns:
    --------
    result : dict
        {
            'total_frames': int,
            'processed_frames': int,
            'faces_per_frame': [
                {
                    'frame_idx': int,
                    'faces': [
                        {
                            'bbox': {'x1', 'y1', 'x2', 'y2'},
                            'confidence': float,
                            'track_id': int,
                            'embedding': List[float],
                            'quality': float
                        }
                    ]
                }
            ],
            'total_processing_time': float
        }
    """
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    faces_per_frame = []
    processed_frames = 0

    try:
        for frame_idx in range(start_frame, end_frame, skip_frames):
            # 프레임 읽기
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Cannot read frame {frame_idx}")
                continue

            # YOLO 얼굴 감지
            yolo_results = face_detector.track(
                frame,
                persist=True,
                conf=0.5,
                iou=0.45,
                verbose=False
            )

            frame_faces = []

            if yolo_results and yolo_results[0].boxes is not None:
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                    # 얼굴 crop
                    face_img = frame[y1:y2, x1:x2].copy()

                    if face_img.size == 0:
                        continue

                    try:
                        # Alignment
                        aligned_face = align_face(face_img)
                        if aligned_face is None:
                            continue

                        # Embedding 추출
                        embedding = embedding_extractor.get_embedding(aligned_face)
                        if embedding is None:
                            continue

                        # 품질 평가
                        quality = quality_assessor.assess_quality(aligned_face)

                        frame_faces.append({
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                            'confidence': conf,
                            'track_id': track_id,
                            'embedding': embedding.tolist(),
                            'quality': float(quality)
                        })

                    except Exception as e:
                        logger.warning(f"Face processing failed at frame {frame_idx}: {e}")
                        continue

            faces_per_frame.append({
                'frame_idx': frame_idx,
                'faces': frame_faces
            })

            processed_frames += 1

            # 진행 상황 로그 (100 프레임마다)
            if processed_frames % 100 == 0:
                logger.info(f"Processed {processed_frames} frames...")

    finally:
        cap.release()

    total_processing_time = time.time() - start_time

    logger.info(f"Video analysis completed: {processed_frames} frames in {total_processing_time:.2f}s")

    return {
        'total_frames': total_frames,
        'processed_frames': processed_frames,
        'faces_per_frame': faces_per_frame,
        'total_processing_time': total_processing_time
    }
