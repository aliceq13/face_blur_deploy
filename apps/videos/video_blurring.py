# -*- coding: utf-8 -*-
"""
비디오 블러 처리 모듈 (Optimized with Saved Tracking Data)

이 모듈은 원본 비디오와 얼굴 정보를 받아, 지정된 얼굴을 블러 처리한 새로운 비디오를 생성합니다.

핵심 최적화:
1. Saved Frame-level BBox: 분석 단계에서 저장한 bbox 사용
2. Instance-based Blur: instance_id 기반 블러 결정
3. Efficient Single-Pass: 한 번의 순회로 처리 완료
4. 적응형 블러: 얼굴 크기에 따른 동적 블러 강도
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional
import logging
import os
import subprocess
import shutil
import gc

logger = logging.getLogger(__name__)


class VideoBlurrer:
    """
    비디오 블러 처리 클래스 (Optimized)

    매 프레임마다:
    1. 저장된 bbox 데이터 조회 (frame_data에서)
    2. instance_id 기반 블러 여부 판단 (is_blurred 필드)
    3. is_blurred=True이면 블러 적용, False이면 보존
    4. 프레임 저장
    """

    def __init__(
        self,
        yolo_model_path: str,
        device: str = 'auto',
        threshold: float = 0.92,
        use_multi_embedding: bool = False
    ):
        """
        VideoBlurrer 초기화

        Note: 현재 최적화된 방식에서는 저장된 bbox를 사용하므로
        YOLO/AdaFace 모델은 로드되지만 process_video()에서는 사용되지 않음.
        향후 실시간 처리가 필요한 경우를 위해 유지.
        """
        self.threshold = threshold
        self.use_multi_embedding = use_multi_embedding

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing VideoBlurrer on {self.device}")

        # YOLO Face 모델 로드 (향후 실시간 처리용)
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"YOLO Face model loaded: {yolo_model_path}")

    def _apply_blur(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        blur_type: str = 'pixelate',
        blur_strength: int = 15,
        padding: int = 20
    ) -> np.ndarray:
        """
        얼굴 영역에 블러 적용

        Args:
            frame: 원본 프레임
            x1, y1, x2, y2: 얼굴 바운딩 박스
            blur_type: 'pixelate' 또는 'gaussian'
            blur_strength: 블러 강도
            padding: 얼굴 주변 패딩

        Returns:
            블러 처리된 프레임
        """
        h, w = frame.shape[:2]

        # 패딩 추가
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if roi.size == 0:
            return frame

        if blur_type == 'pixelate':
            # 픽셀화 (모자이크)
            roi_h, roi_w = roi.shape[:2]
            factor = max(1, max(roi_h, roi_w) // blur_strength)

            if factor > 1:
                small = cv2.resize(roi, (roi_w // factor, roi_h // factor), interpolation=cv2.INTER_NEAREST)
                blurred_roi = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                # 너무 작으면 가우시안 블러 fallback
                k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
                k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
                blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        elif blur_type == 'gaussian':
            # 가우시안 블러
            k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
            k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        else:
            # 기본값: 가우시안 블러
            k_w = max((x2_pad - x1_pad) // 3 | 1, 3)
            k_h = max((y2_pad - y1_pad) // 3 | 1, 3)
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)

        frame[y1_pad:y2_pad, x1_pad:x2_pad] = blurred_roi
        return frame

    def process_video(
        self,
        video_path: str,
        output_path: str,
        face_models: List[Dict],
        progress_callback: Optional[callable] = None,
        blur_type: str = 'pixelate',
        blur_strength: int = 15,
        threshold: float = 0.6
    ) -> bool:
        """
        비디오 블러 처리 파이프라인 (Optimized with Saved Tracking Data)

        Args:
            video_path: 입력 비디오 경로
            output_path: 출력 비디오 경로
            face_models: Face 모델 리스트 [{'id', 'instance_id', 'frame_data', 'is_blurred'}, ...]
            progress_callback: 진행률 콜백 함수
            blur_type: 'pixelate' 또는 'gaussian'
            blur_strength: 블러 강도 (높을수록 약함)
            threshold: 유사도 임계값 (기본 0.6)

        Returns:
            성공 여부
        """
        try:
            logger.info("=" * 80)
            logger.info("🎬 Starting Video Blur Processing (Single-Pass)")
            logger.info("=" * 80)

            # threshold 업데이트
            self.threshold = threshold

            # 1. 프레임별 bbox 데이터 구조화
            # frame_faces[frame_idx] = [(instance_id, bbox, is_blurred), ...]
            frame_faces = {}

            logger.info(f"📋 Total face_models: {len(face_models)}")
            logger.info("🔄 Building frame-level bbox index...")

            for fm in face_models:
                is_blurred_val = fm.get('is_blurred', True)
                instance_id = fm.get('instance_id')
                frame_data = fm.get('frame_data', {})

                logger.info(f"Face instance {instance_id}: is_blurred={is_blurred_val}, frames={len(frame_data)}")

                # frame_data 구조: {frame_idx: [x1, y1, x2, y2, conf], ...}
                for frame_idx_str, bbox_with_conf in frame_data.items():
                    frame_idx = int(frame_idx_str)

                    if frame_idx not in frame_faces:
                        frame_faces[frame_idx] = []

                    # bbox는 [x1, y1, x2, y2, conf] 형태
                    x1, y1, x2, y2 = map(int, bbox_with_conf[:4])

                    frame_faces[frame_idx].append({
                        'instance_id': instance_id,
                        'bbox': (x1, y1, x2, y2),
                        'is_blurred': is_blurred_val
                    })

            logger.info(f"✅ Indexed {len(frame_faces)} frames with face data")

            # 통계
            total_indexed_faces = sum(len(faces) for faces in frame_faces.values())
            preserved_instances = sum(1 for fm in face_models if not fm.get('is_blurred', True))
            logger.info(f"📊 Total indexed faces: {total_indexed_faces}")
            logger.info(f"🎯 Instances to preserve (not blur): {preserved_instances}")

            # 2. 비디오 파일 열기
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            logger.info(f"📹 Video: {width}x{height}, {fps} fps, {total_frames} frames")

            # 3. VideoWriter 생성
            temp_output = output_path + ".temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            if not out.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {temp_output}")

            # 4. 프레임별 처리 (최적화: 저장된 bbox 사용, YOLO/AdaFace 재실행 불필요)
            frame_idx = 0
            blur_count = 0
            preserved_count = 0

            logger.info("🎞️  Processing frames with saved tracking data...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 저장된 bbox 데이터 사용 (YOLO 재실행 없음!)
                if frame_idx in frame_faces:
                    for face_info in frame_faces[frame_idx]:
                        x1, y1, x2, y2 = face_info['bbox']
                        is_blurred = face_info['is_blurred']

                        # 좌표 보정
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        # 너무 작은 얼굴은 스킵
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue

                        # ⭐ instance_id 기반 블러 결정 (AdaFace 재실행 없음!)
                        if is_blurred:
                            # is_blurred=True → 블러 처리
                            frame = self._apply_blur(
                                frame, x1, y1, x2, y2,
                                blur_type=blur_type,
                                blur_strength=blur_strength
                            )
                            blur_count += 1
                        else:
                            # is_blurred=False → 보존 (블러 없음)
                            preserved_count += 1

                # 프레임 저장
                out.write(frame)

                # 진행률 업데이트
                frame_idx += 1
                if progress_callback and frame_idx % 30 == 0:
                    pct = int((frame_idx / total_frames) * 90)
                    pct = min(pct, 90)
                    progress_callback(pct)

                    if frame_idx % 300 == 0:
                        logger.info(
                            f"📊 Processed {frame_idx}/{total_frames} frames | "
                            f"Blurred: {blur_count} | Preserved: {preserved_count}"
                        )

                # 메모리 정리
                if frame_idx % 500 == 0:
                    gc.collect()

            cap.release()
            out.release()

            logger.info(f"✅ Processing completed: {frame_idx} frames")
            logger.info(f"📊 Blurred: {blur_count}, Preserved: {preserved_count}")

            # 5. H.264 인코딩
            logger.info("🎞️  Encoding to H.264...")
            self._encode_h264(temp_output, output_path)

            if progress_callback:
                progress_callback(100)

            logger.info("=" * 80)
            logger.info("✅ Video processing completed successfully!")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"❌ Video processing failed: {e}", exc_info=True)
            return False

    def _encode_h264(self, input_path: str, output_path: str):
        """FFmpeg로 H.264 인코딩"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]

            logger.info(f"🎬 Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3600  # 1시간 타임아웃
            )

            # 임시 파일 삭제
            if os.path.exists(input_path):
                os.remove(input_path)

            logger.info("✅ FFmpeg encoding completed")

        except subprocess.TimeoutExpired as e:
            logger.error("❌ FFmpeg encoding timeout")
            # Fallback: 임시 파일을 그대로 사용
            if os.path.exists(input_path) and not os.path.exists(output_path):
                shutil.move(input_path, output_path)
                logger.warning("⚠️  Using temp file as output (FFmpeg timeout)")

        except subprocess.CalledProcessError as e:
            # FFmpeg 에러 시 stderr 로그 출력
            stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'No stderr'
            logger.error(f"❌ FFmpeg encoding failed (returncode={e.returncode}): {stderr_output[:500]}")
            # Fallback: 임시 파일을 그대로 사용
            if os.path.exists(input_path) and not os.path.exists(output_path):
                shutil.move(input_path, output_path)
                logger.warning("⚠️  Using temp file as output (FFmpeg failed)")

        except Exception as e:
            logger.error(f"❌ FFmpeg encoding failed: {e}")
            # Fallback: 임시 파일을 그대로 사용
            if os.path.exists(input_path) and not os.path.exists(output_path):
                shutil.move(input_path, output_path)
                logger.warning("⚠️  Using temp file as output (FFmpeg failed)")
