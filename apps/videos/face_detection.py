# -*- coding: utf-8 -*-
"""
얼굴 감지 및 추적 파이프라인 (YOLO Face v11s + BoTSORT + AdaFace ViT-12M)

이 모듈은 최신 기술을 사용하여 비디오에서 얼굴을 감지하고 동일 인물을 그룹화합니다.

핵심 기술:
1. YOLO Face v11s: 최신 얼굴 감지 모델 (빠르고 정확)
2. BoTSORT: 고급 추적 알고리즘 (ByteTrack 개선 버전)
3. AdaFace ViT-12M: 최고 성능의 얼굴 인식 모델
4. TW-FINCH/HDBSCAN: 시간 가중 클러스터링
5. Quality-Weighted Embedding: 선명도 기반 임베딩 평균
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import normalize
import logging
import gc

logger = logging.getLogger(__name__)


class DetectedFace:
    """감지된 얼굴 정보를 담는 데이터 클래스"""

    def __init__(
        self,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        face_img: np.ndarray,
        track_id: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        clarity: float = 0.0
    ):
        self.frame_idx = frame_idx
        self.bbox = bbox
        self.confidence = confidence
        self.face_img = face_img
        self.track_id = track_id
        self.embedding = embedding
        self.clarity = clarity

    @property
    def bbox_area(self) -> int:
        """바운딩 박스 면적"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class Tracklet:
    """
    동일한 Track ID를 가진 얼굴들의 집합
    메모리 최적화: 상위 3개 얼굴만 이미지로 저장, 나머지는 임베딩만 저장
    """

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.embeddings: List[np.ndarray] = []
        self.clarity_scores: List[float] = []
        self.top_faces: List[DetectedFace] = []  # 상위 3개만 유지
        self.avg_embedding: Optional[np.ndarray] = None
        self.appearance_count: int = 0
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None

    def add_face(self, face: DetectedFace):
        """얼굴 추가 - 상위 3개 선명한 얼굴 유지"""
        self.appearance_count += 1

        if self.first_frame is None:
            self.first_frame = face.frame_idx
        self.last_frame = face.frame_idx

        # 임베딩 저장
        if face.embedding is not None:
            self.embeddings.append(face.embedding)
            self.clarity_scores.append(face.clarity)

        # Top 3 얼굴 유지
        self.top_faces.append(face)
        self.top_faces.sort(key=lambda x: x.clarity, reverse=True)

        if len(self.top_faces) > 3:
            # 4위 이하는 이미지 메모리 해제
            for rm_face in self.top_faces[3:]:
                rm_face.face_img = None
            self.top_faces = self.top_faces[:3]

    def compute_average_embedding(self, outlier_threshold=0.7):
        """
        Quality-Weighted Averaging + Outlier Filtering

        Args:
            outlier_threshold: Cosine similarity threshold (기본값 0.7)
        """
        if not self.embeddings:
            return

        embeddings = np.array(self.embeddings)
        clarity_scores = np.array(self.clarity_scores)

        # Step 1: 단순 평균 계산 (outlier 감지용)
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

        # Step 2: Outlier Filtering
        similarities = np.dot(embeddings, mean_emb_norm)
        valid_mask = similarities >= outlier_threshold

        if not np.any(valid_mask):
            valid_embeddings = embeddings
            valid_clarity = clarity_scores
        else:
            valid_embeddings = embeddings[valid_mask]
            valid_clarity = clarity_scores[valid_mask]

        # Step 3: Quality-Weighted Averaging
        normalized_clarity = valid_clarity - np.max(valid_clarity)
        exp_clarity = np.exp(normalized_clarity)
        weights = exp_clarity / (np.sum(exp_clarity) + 1e-8)

        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            weights = np.ones(len(valid_embeddings)) / len(valid_embeddings)

        weighted_emb = np.sum(valid_embeddings * weights[:, np.newaxis], axis=0)

        # Step 4: L2 정규화
        norm = np.linalg.norm(weighted_emb)
        if norm > 1e-8:
            self.avg_embedding = weighted_emb / norm
        else:
            self.avg_embedding = valid_embeddings[0] / (np.linalg.norm(valid_embeddings[0]) + 1e-8)

        # 메모리 절약: 개별 임베딩 삭제
        self.embeddings = []
        self.clarity_scores = []


class FaceDetectionPipeline:
    """
    얼굴 감지 및 인식 파이프라인

    YOLO Face v11s + BoTSORT + AdaFace ViT-12M 사용
    """

    # 청크 기반 처리 설정
    CHUNK_SIZE = 2000  # 2000 프레임씩 처리
    MEMORY_CHECK_INTERVAL = 200  # 200 프레임마다 메모리 체크
    MEMORY_LIMIT_GB = 7.0  # 7GB 제한

    def __init__(
        self,
        yolo_model_path: str = None,
        device: str = 'auto',
        sample_rate: int = 1  # 모든 프레임 처리 권장 (Tracking 정확도)
    ):
        self.sample_rate = sample_rate
        self.frames_processed = 0

        # Device 설정
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"🚀 Initializing Face Detection Pipeline on {self.device}")

        # YOLO 모델 경로
        if yolo_model_path is None:
            from django.conf import settings
            yolo_model_path = str(settings.YOLO_FACE_MODEL_PATH)

        # 1. YOLO Face v11s 로드
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        logger.info(f"✅ YOLO Face model loaded: {yolo_model_path}")

        # 2. AdaFace ViT-12M 로드
        try:
            from .adaface_wrapper import AdaFaceWrapper
            from django.conf import settings

            model_path = str(getattr(settings, 'ADAFACE_MODEL_PATH', None))
            model_arch = getattr(settings, 'ADAFACE_ARCHITECTURE', 'ir_101')

            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"AdaFace model not found: {model_path}")

            self.face_recognizer = AdaFaceWrapper(
                model_path=model_path,
                architecture=model_arch,
                device=self.device
            )
            logger.info(f"✅ AdaFace {model_arch} loaded from {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load AdaFace: {e}")
            self.face_recognizer = None

    def _calculate_clarity(self, img: np.ndarray) -> float:
        """이미지 선명도 계산 (Laplacian Variance)"""
        try:
            if img.size == 0:
                return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _check_memory_usage(self):
        """메모리 사용량 체크 및 자동 정리"""
        import psutil

        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)

        if memory_gb > self.MEMORY_LIMIT_GB:
            logger.warning(f"⚠️  Memory high: {memory_gb:.2f} GB, cleaning...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return memory_gb

    def _process_chunk(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        tracklets: Dict[int, Tracklet],
        conf_threshold: float = 0.5
    ) -> Tuple[int, int]:
        """
        청크 단위로 프레임 처리

        Returns:
            (embedding_success, embedding_fail) 튜플
        """
        embedding_success = 0
        embedding_fail = 0

        # YOLO tracking 실행 (BoTSORT)
        results = self.yolo_model.track(
            source=video_path,
            conf=0.4,  # 낮은 threshold로 더 많은 얼굴 감지
            iou=0.5,
            persist=True,
            verbose=False,
            stream=True,
            vid_stride=self.sample_rate,
            device=self.device,
            tracker="botsort.yaml",  # BoTSORT 사용
            imgsz=640,
            half=True  # FP16 사용 (GPU 성능 향상)
        )

        # 청크 범위 내 프레임만 처리
        for idx, result in enumerate(results):
            frame_idx = start_frame + (idx * self.sample_rate)
            if frame_idx >= end_frame:
                break

            frame = result.orig_img

            if not result.boxes or result.boxes.id is None:
                continue

            for box in result.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # 좌표 보정
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue

                # 얼굴 크롭
                face_img_bgr = frame[y1:y2, x1:x2]
                face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
                clarity = self._calculate_clarity(face_img_rgb)

                detected_face = DetectedFace(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    face_img=face_img_rgb,
                    track_id=track_id,
                    clarity=clarity
                )

                # AdaFace 임베딩 추출
                if self.face_recognizer:
                    embedding = self.face_recognizer.get_embedding(face_img_bgr)
                    if embedding is not None:
                        detected_face.embedding = embedding
                        embedding_success += 1
                    else:
                        embedding_fail += 1

                # Tracklet에 추가
                if track_id not in tracklets:
                    tracklets[track_id] = Tracklet(track_id)
                tracklets[track_id].add_face(detected_face)

            # 메모리 체크
            self.frames_processed += 1
            if self.frames_processed % self.MEMORY_CHECK_INTERVAL == 0:
                self._check_memory_usage()

        return embedding_success, embedding_fail

    def _cluster_tracklets(
        self,
        valid_tracklets: List[Tracklet],
        embeddings: List[np.ndarray],
        method: str = 'finch',
        sim_threshold: float = 0.6
    ) -> np.ndarray:
        """
        Tracklet 클러스터링

        Args:
            valid_tracklets: 유효한 tracklet 리스트
            embeddings: 임베딩 리스트
            method: 'finch', 'tw-finch', 'hdbscan', 'hac'
            sim_threshold: HAC 유사도 임계값

        Returns:
            labels: 클러스터 레이블 배열
        """
        if len(valid_tracklets) == 1:
            logger.info("Only 1 tracklet, skipping clustering")
            return np.array([0])

        embeddings_array = np.array(embeddings)

        # FINCH: Parameter-free clustering
        if method == 'finch':
            try:
                from finch import FINCH
                logger.info("🔍 Using FINCH clustering (parameter-free)")

                c, num_clust, req_c = FINCH(embeddings_array, distance='cosine', verbose=False)
                req_c = req_c if req_c is not None else len(num_clust) - 1
                labels = c[:, req_c]

                logger.info(f"✅ FINCH: {num_clust[req_c]} unique persons (level {req_c})")
                return labels

            except ImportError:
                logger.warning("FINCH not available, falling back to HAC")
                method = 'hac'

        # TW-FINCH: Time-Weighted FINCH
        if method == 'tw-finch':
            try:
                from finch import FINCH
                from scipy.spatial.distance import pdist, squareform

                logger.info("🔍 Using TW-FINCH (time-weighted)")

                # 시간 feature 추가
                temporal_features = []
                fps = 30
                for t in valid_tracklets:
                    mid_time = (t.first_frame + t.last_frame) / 2 / fps
                    temporal_features.append([mid_time])

                temporal_features = np.array(temporal_features)
                temporal_range = temporal_features.max() - temporal_features.min()

                if temporal_range > 1e-6:
                    temporal_features = (temporal_features - temporal_features.min()) / temporal_range
                else:
                    temporal_features = np.zeros_like(temporal_features)

                # Embedding + temporal feature
                enhanced_embeddings = np.concatenate([
                    embeddings_array,
                    0.1 * temporal_features
                ], axis=1)

                c, num_clust, req_c = FINCH(enhanced_embeddings, distance='cosine', verbose=False)
                req_c = req_c if req_c is not None else len(num_clust) - 1
                labels = c[:, req_c]

                logger.info(f"✅ TW-FINCH: {num_clust[req_c]} unique persons (level {req_c})")
                return labels

            except ImportError:
                logger.warning("FINCH not available, falling back to HAC")
                method = 'hac'

        # HDBSCAN: State-of-the-art clustering
        if method == 'hdbscan':
            try:
                import hdbscan
                logger.info("🔍 Using HDBSCAN clustering")

                embeddings_array = embeddings_array.astype('float64')
                embeddings_array = normalize(embeddings_array, norm='l2', axis=1).astype('float64')

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=3,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )

                labels = clusterer.fit_predict(embeddings_array)

                # 노이즈(-1) 처리
                if -1 in labels:
                    noise_indices = np.where(labels == -1)[0]
                    max_label = labels.max()
                    for idx, noise_idx in enumerate(noise_indices):
                        labels[noise_idx] = max_label + 1 + idx

                num_clusters = len(set(labels))
                logger.info(f"✅ HDBSCAN: {num_clusters} unique persons")
                return labels

            except ImportError:
                logger.warning("HDBSCAN not available, falling back to HAC")
                method = 'hac'

        # HAC: Fallback method
        logger.info("🔍 Using HAC clustering")
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - sim_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings_array)

        num_clusters = len(set(labels))
        logger.info(f"✅ HAC: {num_clusters} unique persons")
        return labels

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        conf_threshold: float = 0.5,
        sim_threshold: float = 0.6,
        clustering_method: str = 'finch',
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        전체 파이프라인 실행

        Tracking -> Embedding -> Clustering -> Thumbnail Generation
        """
        logger.info("=" * 80)
        logger.info(f"🎬 Starting Face Detection Pipeline: {video_path}")
        logger.info("=" * 80)

        # 비디오 정보
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logger.info(f"📹 Total frames: {total_frames}, chunk size: {self.CHUNK_SIZE}")

        # 청크 계산
        num_chunks = (total_frames + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
        tracklets: Dict[int, Tracklet] = {}

        total_embedding_success = 0
        total_embedding_fail = 0

        # 청크별 처리
        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * self.CHUNK_SIZE
            end_frame = min((chunk_idx + 1) * self.CHUNK_SIZE, total_frames)

            logger.info(f"📦 Chunk {chunk_idx + 1}/{num_chunks} (frames {start_frame}-{end_frame})")

            success, fail = self._process_chunk(
                video_path, start_frame, end_frame, tracklets, conf_threshold
            )

            total_embedding_success += success
            total_embedding_fail += fail

            if progress_callback:
                progress_pct = int((end_frame / total_frames) * 80)  # 80%까지
                progress_callback(progress_pct, f"Chunk {chunk_idx + 1}/{num_chunks} 완료")

            # 청크 간 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"✅ Tracking completed: {len(tracklets)} tracklets")
        logger.info(f"📊 Embedding: {total_embedding_success} success, {total_embedding_fail} fail")

        # Tracklet 평균 임베딩 계산
        valid_tracklets = []
        embeddings = []

        for t in tracklets.values():
            t.compute_average_embedding()
            if t.avg_embedding is not None:
                valid_tracklets.append(t)
                embeddings.append(t.avg_embedding)

        if not valid_tracklets:
            logger.warning("⚠️  No valid tracklets found")
            return []

        logger.info(f"🎯 Valid tracklets: {len(valid_tracklets)}")

        # 클러스터링
        labels = self._cluster_tracklets(
            valid_tracklets, embeddings, clustering_method, sim_threshold
        )

        # 클러스터별 그룹화
        clusters: Dict[int, List[Tracklet]] = {}
        for tracklet, label in zip(valid_tracklets, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tracklet)

        logger.info(f"👥 Unique persons: {len(clusters)}")

        if progress_callback:
            progress_callback(90, "썸네일 생성 중...")

        # 썸네일 생성
        result_faces = []
        face_index = 1

        for cluster_id, cluster_tracklets in clusters.items():
            # 클러스터 내 모든 Top 얼굴 수집
            all_top_faces = []
            for t in cluster_tracklets:
                all_top_faces.extend(t.top_faces)

            if not all_top_faces:
                continue

            # 선명도 순 정렬
            all_top_faces.sort(key=lambda x: x.clarity, reverse=True)
            final_top_3 = all_top_faces[:3]
            best_face = final_top_3[0]

            # 썸네일 저장
            thumbnail_filename = f"face_{face_index}.jpg"
            thumbnail_path = os.path.join(output_dir, thumbnail_filename)
            self._save_thumbnail(best_face, thumbnail_path)

            # 프레임 정보
            all_first_frames = [t.first_frame for t in cluster_tracklets if t.first_frame is not None]
            all_last_frames = [t.last_frame for t in cluster_tracklets if t.last_frame is not None]
            total_appearances = sum(t.appearance_count for t in cluster_tracklets)

            # 클러스터 평균 임베딩
            cluster_avg_emb = np.mean([t.avg_embedding for t in cluster_tracklets], axis=0)
            cluster_avg_emb = normalize(cluster_avg_emb.reshape(1, -1))[0]

            # Multi-thumbnail 임베딩 (상위 3개)
            multi_embeddings = []
            for face in final_top_3:
                if face.embedding is not None:
                    multi_embeddings.append(face.embedding.tolist())

            result_faces.append({
                'face_index': face_index,
                'thumbnail_path': thumbnail_path,
                'embedding': cluster_avg_emb.tolist(),
                'embeddings': multi_embeddings,  # Multi-thumbnail
                'appearance_count': total_appearances,
                'first_frame': min(all_first_frames) if all_first_frames else 0,
                'last_frame': max(all_last_frames) if all_last_frames else 0
            })

            face_index += 1

        if progress_callback:
            progress_callback(100, "완료")

        logger.info("=" * 80)
        logger.info(f"✅ Pipeline completed: {len(result_faces)} unique faces")
        logger.info("=" * 80)

        return result_faces

    def _save_thumbnail(self, face: DetectedFace, output_path: str, size=(160, 160)):
        """썸네일 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        resized = cv2.resize(face.face_img, size, interpolation=cv2.INTER_AREA)
        bgr_img = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr_img)
