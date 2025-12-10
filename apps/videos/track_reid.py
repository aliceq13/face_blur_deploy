"""
Two-Stage Track Re-Identification System
==========================================

Instance ID를 유지하기 위한 Track Re-identification 시스템.
장면 전환이나 화면 밖 이탈 후 재등장 시에도 같은 사람을 같은 instance로 인식.

Method: Two-Stage Verification
- Stage 1: Fast matching with best representative embedding (high threshold)
- Stage 2: Detailed matching with top-K embeddings (lower threshold)
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingData:
    """단일 임베딩 데이터"""
    embedding: np.ndarray
    quality: float  # Laplacian variance
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    track_id: int


@dataclass
class InstanceData:
    """Instance (사람) 데이터"""
    instance_id: int
    representative: EmbeddingData  # 가장 품질 좋은 대표 임베딩
    embeddings: List[EmbeddingData] = field(default_factory=list)  # 모든 임베딩 (품질순 정렬)
    track_ids: set = field(default_factory=set)  # 이 instance에 속한 모든 track_id
    best_thumbnail: Optional[np.ndarray] = None  # 최고 품질 얼굴 이미지
    total_frames: int = 0  # 등장한 총 프레임 수
    frame_data: Dict[int, List[float]] = field(default_factory=dict)  # 프레임별 bbox: {frame_idx: [x1, y1, x2, y2, conf]}


class TwoStageTrackReID:
    """
    Two-Stage Track Re-Identification

    Stage 1: Fast matching - 대표 임베딩과 비교 (빠르고 확실한 매칭)
    Stage 2: Slow matching - 상위 K개 임베딩과 비교 (정교한 매칭)

    Parameters:
    -----------
    fast_threshold : float
        Stage 1 빠른 매칭 임계값 (0.88-0.92 권장)
        이 값보다 높으면 즉시 매칭

    slow_threshold : float
        Stage 2 정밀 매칭 임계값 (0.80-0.85 권장)
        상위 K개 임베딩 평균이 이 값보다 높으면 매칭

    top_k : int
        Stage 2에서 비교할 상위 임베딩 개수 (기본 5)

    max_embeddings_per_instance : int
        각 instance당 저장할 최대 임베딩 수 (메모리 관리)

    min_quality_for_update : float
        Instance 업데이트에 사용할 최소 품질 점수
    """

    def __init__(
        self,
        fast_threshold: float = 0.90,
        slow_threshold: float = 0.82,
        top_k: int = 5,
        max_embeddings_per_instance: int = 20,
        min_quality_for_update: float = 30.0
    ):
        self.fast_threshold = fast_threshold
        self.slow_threshold = slow_threshold
        self.top_k = top_k
        self.max_embeddings_per_instance = max_embeddings_per_instance
        self.min_quality_for_update = min_quality_for_update

        # Instance 데이터베이스
        self.instances: Dict[int, InstanceData] = {}

        # Track ID → Instance ID 매핑 (빠른 조회)
        self.track_to_instance: Dict[int, int] = {}

        # 다음 instance ID
        self.next_instance_id = 0

        # 통계
        self.stats = {
            'total_tracks': 0,
            'total_instances': 0,
            'fast_matches': 0,
            'slow_matches': 0,
            'new_instances': 0,
            'track_reassignments': 0
        }

        logger.info(
            f"TwoStageTrackReID initialized: "
            f"fast_threshold={fast_threshold}, "
            f"slow_threshold={slow_threshold}, "
            f"top_k={top_k}"
        )

    def match_or_create_instance(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        face_image: Optional[np.ndarray] = None,
        confidence: float = 1.0
    ) -> int:
        """
        Track ID를 Instance ID로 매칭하거나 새로 생성

        Returns:
        --------
        instance_id : int
            매칭되거나 새로 생성된 instance ID
        """
        self.stats['total_tracks'] += 1

        # L2 정규화
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # 1. 기존 track_id가 이미 매핑되어 있는 경우
        if track_id in self.track_to_instance:
            instance_id = self.track_to_instance[track_id]
            self._update_instance(
                instance_id, track_id, embedding, quality,
                frame_idx, bbox, face_image, confidence
            )
            return instance_id

        # 2. 새로운 track_id → Re-identification 수행
        instance_id, match_type = self._find_matching_instance(embedding)

        if instance_id is not None:
            # 기존 instance에 매칭
            logger.info(
                f"Track {track_id} matched to Instance {instance_id} "
                f"via {match_type} (frame {frame_idx})"
            )

            self.track_to_instance[track_id] = instance_id
            self.instances[instance_id].track_ids.add(track_id)

            if match_type == 'fast':
                self.stats['fast_matches'] += 1
            else:
                self.stats['slow_matches'] += 1

            # Instance 업데이트
            self._update_instance(
                instance_id, track_id, embedding, quality,
                frame_idx, bbox, face_image, confidence
            )

        else:
            # 새 instance 생성
            instance_id = self._create_new_instance(
                track_id, embedding, quality, frame_idx, bbox, face_image, confidence
            )

            self.stats['new_instances'] += 1
            self.stats['total_instances'] += 1

            logger.info(
                f"Created new Instance {instance_id} for Track {track_id} "
                f"(frame {frame_idx}, quality {quality:.1f})"
            )

        return instance_id

    def _find_matching_instance(
        self,
        embedding: np.ndarray
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Two-Stage matching

        Returns:
        --------
        (instance_id, match_type) : (int | None, str | None)
            매칭된 instance_id와 매칭 타입 ('fast' | 'slow' | None)
        """
        if not self.instances:
            return None, None

        # Stage 1: Fast matching with representative
        max_fast_sim = -1.0
        for instance_id, instance_data in self.instances.items():
            rep_emb = instance_data.representative.embedding
            similarity = np.dot(embedding, rep_emb)
            max_fast_sim = max(max_fast_sim, similarity)

            if similarity > self.fast_threshold:
                # 확실한 매칭
                logger.debug(f"Fast match: similarity={similarity:.4f} > threshold={self.fast_threshold}")
                return instance_id, 'fast'

        logger.debug(f"Fast match failed: max_similarity={max_fast_sim:.4f} <= threshold={self.fast_threshold}")

        # Stage 2: Detailed matching with top-K embeddings
        best_instance_id = None
        best_avg_similarity = -1.0

        for instance_id, instance_data in self.instances.items():
            # 상위 K개 임베딩 선택 (품질순 정렬되어 있음)
            top_embeddings = instance_data.embeddings[:self.top_k]

            if not top_embeddings:
                continue

            # 평균 유사도 계산
            similarities = [
                np.dot(embedding, emb_data.embedding)
                for emb_data in top_embeddings
            ]
            avg_similarity = np.mean(similarities)
            max_similarity = max(similarities)

            # 평균과 최대값 모두 고려 (가중 평균)
            weighted_similarity = 0.6 * avg_similarity + 0.4 * max_similarity

            if weighted_similarity > best_avg_similarity:
                best_avg_similarity = weighted_similarity
                best_instance_id = instance_id

        # Slow threshold 체크
        if best_avg_similarity > self.slow_threshold:
            logger.debug(f"Slow match: weighted_sim={best_avg_similarity:.4f} > threshold={self.slow_threshold}")
            return best_instance_id, 'slow'

        # 매칭 실패
        logger.debug(f"Slow match failed: best_weighted_sim={best_avg_similarity:.4f} <= threshold={self.slow_threshold}")
        return None, None

    def _create_new_instance(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        face_image: Optional[np.ndarray],
        confidence: float = 1.0
    ) -> int:
        """새 instance 생성"""
        instance_id = self.next_instance_id
        self.next_instance_id += 1

        emb_data = EmbeddingData(
            embedding=embedding,
            quality=quality,
            frame_idx=frame_idx,
            bbox=bbox,
            track_id=track_id
        )

        # 프레임별 bbox 데이터 초기화
        frame_data = {
            frame_idx: [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(confidence)]
        }

        instance_data = InstanceData(
            instance_id=instance_id,
            representative=emb_data,
            embeddings=[emb_data],
            track_ids={track_id},
            best_thumbnail=face_image.copy() if face_image is not None else None,
            total_frames=1,
            frame_data=frame_data
        )

        self.instances[instance_id] = instance_data
        self.track_to_instance[track_id] = instance_id

        return instance_id

    def _update_instance(
        self,
        instance_id: int,
        track_id: int,
        embedding: np.ndarray,
        quality: float,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        face_image: Optional[np.ndarray],
        confidence: float = 1.0
    ):
        """기존 instance 업데이트"""
        instance_data = self.instances[instance_id]
        instance_data.total_frames += 1

        # 프레임별 bbox 저장 (모든 프레임 저장)
        instance_data.frame_data[frame_idx] = [
            float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(confidence)
        ]

        # 품질이 너무 낮으면 임베딩 업데이트 스킵 (bbox는 이미 저장됨)
        if quality < self.min_quality_for_update:
            return

        # 새 임베딩 데이터 생성
        emb_data = EmbeddingData(
            embedding=embedding,
            quality=quality,
            frame_idx=frame_idx,
            bbox=bbox,
            track_id=track_id
        )

        # 임베딩 리스트에 추가
        instance_data.embeddings.append(emb_data)

        # 품질순 정렬
        instance_data.embeddings.sort(key=lambda x: x.quality, reverse=True)

        # 최대 개수 제한
        if len(instance_data.embeddings) > self.max_embeddings_per_instance:
            instance_data.embeddings = instance_data.embeddings[:self.max_embeddings_per_instance]

        # 대표 임베딩 업데이트 (가장 품질 좋은 것)
        if quality > instance_data.representative.quality:
            instance_data.representative = emb_data

            # 썸네일도 업데이트
            if face_image is not None:
                instance_data.best_thumbnail = face_image.copy()

            logger.debug(
                f"Instance {instance_id} representative updated: "
                f"quality {quality:.1f} at frame {frame_idx}"
            )

    def get_instance_thumbnails(self) -> Dict[int, Dict[str, Any]]:
        """
        각 instance의 최고 품질 썸네일 반환

        Returns:
        --------
        thumbnails : dict
            {
                instance_id: {
                    'thumbnail': np.ndarray,
                    'embedding': np.ndarray,
                    'quality': float,
                    'frame_idx': int,
                    'bbox': (x1, y1, x2, y2),
                    'track_ids': [id1, id2, ...],
                    'total_frames': int,
                    'frame_data': {frame_idx: [x1, y1, x2, y2, conf], ...}
                }
            }
        """
        thumbnails = {}

        for instance_id, instance_data in self.instances.items():
            rep = instance_data.representative

            thumbnails[instance_id] = {
                'thumbnail': instance_data.best_thumbnail,
                'embedding': rep.embedding.tolist(),
                'quality': rep.quality,
                'frame_idx': rep.frame_idx,
                'bbox': rep.bbox,
                'track_ids': sorted(list(instance_data.track_ids)),
                'total_frames': instance_data.total_frames,
                'frame_data': instance_data.frame_data  # 프레임별 bbox 데이터 포함
            }

        logger.info(
            f"Generated {len(thumbnails)} instance thumbnails "
            f"from {len(self.track_to_instance)} tracks"
        )

        return thumbnails

    def get_statistics(self) -> Dict[str, Any]:
        """Re-ID 통계 반환"""
        return {
            **self.stats,
            'avg_tracks_per_instance': (
                self.stats['total_tracks'] / max(self.stats['total_instances'], 1)
            ),
            'fast_match_rate': (
                self.stats['fast_matches'] / max(self.stats['total_tracks'], 1)
            ),
            'slow_match_rate': (
                self.stats['slow_matches'] / max(self.stats['total_tracks'], 1)
            )
        }

    def reset(self):
        """상태 초기화"""
        self.instances.clear()
        self.track_to_instance.clear()
        self.next_instance_id = 0
        self.stats = {
            'total_tracks': 0,
            'total_instances': 0,
            'fast_matches': 0,
            'slow_matches': 0,
            'new_instances': 0,
            'track_reassignments': 0
        }
