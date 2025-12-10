"""
Face Instance Detection with Re-Identification
===============================================

YOLO Face v11s + BoTSORT + Two-Stage Re-ID를 사용한
얼굴 instance 감지 및 추적 시스템.

장면 전환이나 화면 밖 이탈에도 같은 사람을 같은 instance로 유지.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from ultralytics import YOLO
from django.conf import settings

from .adaface_wrapper import AdaFaceWrapper
from .face_aligner import align_face, align_face_with_landmarks
from .track_reid import TwoStageTrackReID
from .maniqa_wrapper import MANIQAWrapper

logger = logging.getLogger(__name__)


def expand_bbox(x1: int, y1: int, x2: int, y2: int,
                img_width: int, img_height: int,
                margin: float = 0.2) -> Tuple[int, int, int, int]:
    """
    Bbox를 확장하여 얼굴 전체가 보이도록 함

    Parameters:
    -----------
    x1, y1, x2, y2 : int
        원본 bbox 좌표
    img_width, img_height : int
        이미지 크기
    margin : float
        확장 비율 (0.2 = 20% 확장, 기본값)

    Returns:
    --------
    expanded_bbox : (x1, y1, x2, y2)
        확장된 bbox 좌표
    """
    w = x2 - x1
    h = y2 - y1

    # 각 방향으로 margin만큼 확장
    margin_w = int(w * margin)
    margin_h = int(h * margin)

    # 확장된 좌표 계산
    new_x1 = max(0, x1 - margin_w)
    new_y1 = max(0, y1 - margin_h)
    new_x2 = min(img_width, x2 + margin_w)
    new_y2 = min(img_height, y2 + margin_h)

    return new_x1, new_y1, new_x2, new_y2


def make_square_thumbnail(image: np.ndarray, size: int = 256) -> np.ndarray:
    """
    이미지를 정사각형 썸네일로 변환 (resize + padding)

    Parameters:
    -----------
    image : np.ndarray
        입력 이미지 (BGR)
    size : int
        출력 정사각형 크기 (기본 256x256)

    Returns:
    --------
    square_image : np.ndarray
        정사각형 이미지 (size x size, BGR)
    """
    h, w = image.shape[:2]

    # 긴 쪽을 기준으로 리사이즈
    if h > w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)

    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 정사각형 캔버스 생성 (검은색 배경)
    square = np.zeros((size, size, 3), dtype=np.uint8)

    # 중앙에 배치
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return square


class FaceInstanceDetector:
    """
    얼굴 Instance 감지 및 Re-identification

    Pipeline:
    1. YOLO Face v11s로 얼굴 감지
    2. BoTSORT로 tracking (track_id 부여)
    3. CVLface alignment
    4. AdaFace ViT-12M으로 임베딩 추출
    5. Two-Stage Re-ID로 instance_id 할당
    6. 각 instance의 최고 품질 썸네일 반환

    Parameters:
    -----------
    yolo_model_path : str
        YOLO Face 모델 경로

    adaface_model_path : str
        AdaFace 모델 경로

    adaface_architecture : str
        AdaFace 아키텍처 ('vit', 'ir_50', 'ir_101')

    device : str
        'cuda', 'cpu', 또는 'auto'

    reid_config : dict
        Re-ID 시스템 설정
        - fast_threshold: 빠른 매칭 임계값 (기본 0.90)
        - slow_threshold: 정밀 매칭 임계값 (기본 0.82)
        - top_k: Stage 2에서 비교할 상위 임베딩 개수 (기본 5)
    """

    def __init__(
        self,
        yolo_model_path: str,
        adaface_model_path: str,
        adaface_architecture: str = 'vit',
        device: str = 'auto',
        reid_config: Optional[Dict] = None,
        use_maniqa: bool = True
    ):
        # Device 설정
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"FaceInstanceDetector initializing on device: {self.device}")

        # YOLO Face 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        logger.info(f"Loaded YOLO Face model from {yolo_model_path}")

        # AdaFace 모델 로드
        self.face_recognizer = AdaFaceWrapper(
            model_path=adaface_model_path,
            architecture=adaface_architecture,
            device=self.device
        )
        logger.info(f"Loaded AdaFace {adaface_architecture} model")

        # MANIQA 품질 평가 모델 (선택적)
        self.use_maniqa = use_maniqa
        if use_maniqa:
            try:
                self.quality_assessor = MANIQAWrapper(device=self.device)
                logger.info("Using MANIQA for quality assessment")
            except Exception as e:
                logger.warning(f"Failed to load MANIQA, falling back to Laplacian Variance: {e}")
                self.quality_assessor = None
                self.use_maniqa = False
        else:
            self.quality_assessor = None
            logger.info("Using Laplacian Variance for quality assessment")

        # Re-ID 시스템 초기화
        reid_config = reid_config or {}
        self.reid = TwoStageTrackReID(
            fast_threshold=reid_config.get('fast_threshold', 0.90),
            slow_threshold=reid_config.get('slow_threshold', 0.82),
            top_k=reid_config.get('top_k', 5),
            max_embeddings_per_instance=reid_config.get('max_embeddings', 20),
            min_quality_for_update=reid_config.get('min_quality', 30.0)
        )

        # 통계
        self.stats = {
            'total_frames': 0,
            'frames_with_faces': 0,
            'total_detections': 0,
            'failed_alignments': 0,
            'failed_embeddings': 0
        }

    def _calculate_quality(self, aligned_face: np.ndarray) -> float:
        """
        얼굴 이미지 품질 평가

        Parameters:
        -----------
        aligned_face : np.ndarray
            정렬된 얼굴 이미지 (BGR)

        Returns:
        --------
        quality : float
            품질 점수
        """
        if self.use_maniqa and self.quality_assessor is not None:
            # MANIQA 사용
            return self.quality_assessor.assess_quality(aligned_face)
        else:
            # Laplacian Variance 사용 (기존 방식)
            gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        conf_threshold: float = 0.4
    ) -> Dict[int, Dict[str, Any]]:
        """
        영상 처리 및 instance별 썸네일 반환

        Parameters:
        -----------
        video_path : str
            영상 파일 경로

        progress_callback : callable, optional
            진행률 콜백 함수 (0-100)

        conf_threshold : float
            YOLO 감지 신뢰도 임계값

        Returns:
        --------
        instances : dict
            {
                instance_id: {
                    'thumbnail': np.ndarray,  # BGR 이미지
                    'embedding': list,  # 512-d 임베딩
                    'quality': float,  # Laplacian variance
                    'frame_idx': int,
                    'bbox': (x1, y1, x2, y2),
                    'track_ids': [id1, id2, ...],
                    'total_frames': int,
                    'frame_data': {frame_idx: [x1, y1, x2, y2, conf], ...}
                }
            }
        """
        logger.info(f"Starting video processing: {video_path}")

        # 영상 정보 확인
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames"
        )

        cap.release()

        # Re-ID 초기화
        self.reid.reset()

        # YOLO tracking 실행
        results_generator = self.yolo_model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            stream=True,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )

        # 프레임별 처리
        frame_idx = 0

        for result in results_generator:
            frame = result.orig_img  # BGR 이미지
            self.stats['total_frames'] += 1

            if result.boxes is None or len(result.boxes) == 0:
                frame_idx += 1
                continue

            self.stats['frames_with_faces'] += 1

            # 각 감지된 얼굴 처리
            for box in result.boxes:
                # Track ID 확인
                if box.id is None:
                    continue

                track_id = int(box.id.cpu().numpy()[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                # 좌표 보정
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # 너무 작은 얼굴 스킵
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                self.stats['total_detections'] += 1

                # ⭐ 핵심 변경: Alignment용 크롭에 40% 마진 추가
                # YOLO Face가 tight bbox를 제공하는 경향이 있어 DFA가 얼굴 전체를 볼 수 있도록 함
                align_x1, align_y1, align_x2, align_y2 = expand_bbox(
                    x1, y1, x2, y2, width, height, margin=0.4  # 40% 마진
                )
                face_img_for_align = frame[align_y1:align_y2, align_x1:align_x2].copy()

                # 썸네일용 확장된 bbox (20% 확장 유지)
                thumb_x1, thumb_y1, thumb_x2, thumb_y2 = expand_bbox(
                    x1, y1, x2, y2, width, height, margin=0.2
                )
                face_img_expanded = frame[thumb_y1:thumb_y2, thumb_x1:thumb_x2].copy()

                # Alignment (with landmarks for future KPRPE support)
                try:
                    aligned_face, aligned_ldmks = align_face_with_landmarks(face_img_for_align)
                    if aligned_face is None:
                        self.stats['failed_alignments'] += 1
                        continue
                except Exception as e:
                    logger.warning(f"Alignment failed for track {track_id}: {e}")
                    self.stats['failed_alignments'] += 1
                    continue

                # 품질 점수 계산 (MANIQA 또는 Laplacian variance)
                quality = self._calculate_quality(aligned_face)

                # 임베딩 추출
                # ⚠️ aligned_face는 이미 RGB 112x112로 정렬됨
                # adaface_wrapper에서 다시 정렬하지 않도록 skip_alignment=True
                # ⭐ KPRPE: aligned_ldmks를 keypoints로 전달
                try:
                    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                    embedding = self.face_recognizer.get_embedding(
                        aligned_face_bgr, 
                        skip_alignment=True,
                        keypoints=aligned_ldmks  # KPRPE keypoints 전달
                    )
                    if embedding is None:
                        self.stats['failed_embeddings'] += 1
                        continue
                except Exception as e:
                    logger.warning(f"Embedding extraction failed for track {track_id}: {e}")
                    self.stats['failed_embeddings'] += 1
                    continue

                # Re-ID: Instance 매칭 또는 생성
                # ⭐ face_image는 정사각형 썸네일 (유저 표시용) - 확장된 bbox 사용
                # ⭐ aligned_face는 임베딩 추출에만 사용
                square_thumbnail = make_square_thumbnail(face_img_expanded, size=256)

                instance_id = self.reid.match_or_create_instance(
                    track_id=track_id,
                    embedding=embedding,
                    quality=quality,
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    face_image=square_thumbnail,  # 정사각형 썸네일
                    confidence=conf
                )

                # 디버깅 로그 (100 프레임마다)
                if frame_idx % 100 == 0:
                    logger.debug(
                        f"Frame {frame_idx}: Track {track_id} → Instance {instance_id}, "
                        f"quality={quality:.1f}, conf={conf:.3f}"
                    )

            # 진행률 업데이트
            frame_idx += 1
            if progress_callback and frame_idx % 30 == 0:
                progress = int((frame_idx / total_frames) * 100)
                progress_callback(min(progress, 100))

        # 최종 진행률
        if progress_callback:
            progress_callback(100)

        # 통계 로그
        logger.info(f"Processing completed: {frame_idx} frames")
        logger.info(f"Stats: {self.stats}")

        reid_stats = self.reid.get_statistics()
        logger.info(f"Re-ID Stats: {reid_stats}")

        # Instance 썸네일 반환
        instances = self.reid.get_instance_thumbnails()

        logger.info(f"Detected {len(instances)} unique instances")

        return instances

    def get_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            'detector_stats': self.stats,
            'reid_stats': self.reid.get_statistics()
        }


def create_face_instance_detector(
    yolo_model_path: Optional[str] = None,
    adaface_model_path: Optional[str] = None,
    adaface_architecture: Optional[str] = None,
    device: str = 'auto',
    reid_config: Optional[Dict] = None,
    use_maniqa: bool = True
) -> FaceInstanceDetector:
    """
    FaceInstanceDetector 생성 헬퍼 함수 (Django settings 사용)

    Parameters:
    -----------
    yolo_model_path : str, optional
        YOLO 모델 경로 (기본: settings.YOLO_FACE_MODEL_PATH)

    adaface_model_path : str, optional
        AdaFace 모델 경로 (기본: settings.ADAFACE_MODEL_PATH)

    adaface_architecture : str, optional
        AdaFace 아키텍처 (기본: settings.ADAFACE_ARCHITECTURE)

    device : str
        디바이스 ('cuda', 'cpu', 'auto')

    reid_config : dict, optional
        Re-ID 설정

    use_maniqa : bool
        MANIQA 품질 평가 사용 여부 (기본: True)

    Returns:
    --------
    detector : FaceInstanceDetector
    """
    if yolo_model_path is None:
        yolo_model_path = str(getattr(settings, 'YOLO_FACE_MODEL_PATH', ''))

    if adaface_model_path is None:
        adaface_model_path = str(getattr(settings, 'ADAFACE_MODEL_PATH', ''))

    if adaface_architecture is None:
        adaface_architecture = getattr(settings, 'ADAFACE_ARCHITECTURE', 'vit')

    return FaceInstanceDetector(
        yolo_model_path=yolo_model_path,
        adaface_model_path=adaface_model_path,
        adaface_architecture=adaface_architecture,
        device=device,
        reid_config=reid_config,
        use_maniqa=use_maniqa
    )
