# -*- coding: utf-8 -*-
"""
비디오 처리 Celery Tasks

Pipeline:
- YOLO Face v11s + BoTSORT: 얼굴 감지 및 tracking
- CVLface DFA: 얼굴 정렬
- AdaFace ViT-12M: 임베딩 추출
- Two-Stage Re-ID: Instance 유지

Tasks:
- analyze_faces_task: 얼굴 instance 분석
- process_video_blur_task: 비디오 블러 처리
"""

import os
import logging
import cv2
import json
from celery import shared_task
from django.conf import settings
from .models import Video, Face, ProcessingJob
from .face_instance_detector import create_face_instance_detector
from .memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=1,
    soft_time_limit=10800,  # 3시간 soft limit
    time_limit=10900,       # 3시간 1분 hard limit
)
def analyze_faces_task(self, video_id: str):
    """
    비디오 얼굴 Instance 분석 Celery Task (Two-Stage Re-ID)

    처리 흐름:
    1. Video 객체 로드
    2. 상태를 'analyzing'으로 변경
    3. ProcessingJob 생성/업데이트
    4. FaceInstanceDetector 실행
       - YOLO Face v11s: 얼굴 감지
       - BoTSORT: track_id 할당
       - CVLface: alignment
       - AdaFace ViT-12M: 임베딩 추출
       - Two-Stage Re-ID: instance_id 할당 (같은 사람은 같은 ID 유지)
       - 각 instance의 최고 품질 썸네일만 저장
    5. Face 모델 생성 (각 instance마다 1개)
    6. Video 상태를 'ready'로 변경
    7. ProcessingJob 완료 처리

    Args:
        video_id: Video 모델 UUID (문자열)

    Returns:
        {
            'video_id': str,
            'instances_count': int,
            'status': 'success'
        }

    Raises:
        ValueError: Video를 찾을 수 없는 경우
        Exception: 처리 중 오류 발생 시 재시도
    """
    logger.info(f"[Task {self.request.id}] Starting face instance analysis for video {video_id}")

    # 메모리 모니터 초기화 (7GB 제한)
    memory_monitor = MemoryMonitor(limit_gb=7.0, warning_threshold=0.85)
    memory_monitor.log_usage("Task start")

    try:
        # ====================================================================
        # 1. Video 객체 로드
        # ====================================================================
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            logger.error(f"Video {video_id} not found")
            raise ValueError(f"Video {video_id} does not exist")

        # ====================================================================
        # 2. 상태 업데이트: uploaded → analyzing
        # ====================================================================
        video.status = 'analyzing'
        video.progress = 0
        video.save(update_fields=['status', 'progress'])

        logger.info(f"Video {video_id} status changed to 'analyzing'")

        # ====================================================================
        # 3. ProcessingJob 생성 또는 업데이트
        # ====================================================================
        job, created = ProcessingJob.objects.get_or_create(
            video=video,
            job_type='face_analysis',
            celery_task_id=self.request.id,
            defaults={
                'status': 'started',
                'progress': 0
            }
        )

        if not created:
            job.status = 'started'
            job.progress = 0
            job.celery_task_id = self.request.id
            job.save(update_fields=['status', 'progress', 'celery_task_id'])

        logger.info(f"ProcessingJob created/updated: {job.id}")

        # ====================================================================
        # 4. 비디오 파일 경로 확인
        # ====================================================================
        video_filename = video.original_file_url.split('/')[-1]
        video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Video file path: {video_path}")

        # ====================================================================
        # 5. 출력 디렉토리 설정
        # ====================================================================
        thumbnail_dir = os.path.join(
            settings.MEDIA_ROOT,
            'faces',
            'thumbnails',
            str(video.id)
        )
        os.makedirs(thumbnail_dir, exist_ok=True)

        # ====================================================================
        # 6. FaceInstanceDetector 실행 (Two-Stage Re-ID)
        # ====================================================================
        logger.info("=" * 80)
        logger.info("Initializing FaceInstanceDetector with Two-Stage Re-ID...")
        logger.info("=" * 80)
        memory_monitor.log_usage("Before detector init")

        # Re-ID 설정
        reid_config = {
            'fast_threshold': 0.65,
            'slow_threshold': 0.65,
            'top_k': 5,
            'max_embeddings': 20,
            'min_quality': 50.0
        }

        detector = create_face_instance_detector(
            device='auto',
            reid_config=reid_config,
            use_maniqa=True
        )

        # 진행률 업데이트: 10%
        video.progress = 10
        video.save(update_fields=['progress'])
        job.progress = 10
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 10})

        memory_monitor.log_usage("After detector init")

        # 진행률 콜백 함수
        def update_progress(percent: int):
            """진행률 업데이트 콜백"""
            # 10% (초기화) + 80% (처리) = 90%
            actual_percent = 10 + int(percent * 0.8)
            actual_percent = min(actual_percent, 90)

            video.progress = actual_percent
            video.save(update_fields=['progress'])
            job.progress = actual_percent
            job.save(update_fields=['progress'])
            self.update_state(state='PROGRESS', meta={
                'progress': actual_percent
            })

            # 메모리 체크
            if actual_percent % 10 == 0:
                memory_monitor.check_and_cleanup()

        logger.info("Running face instance detection with Re-ID...")
        instances = detector.process_video(
            video_path=video_path,
            progress_callback=update_progress,
            conf_threshold=0.4  # YOLO 신뢰도 임계값
        )

        logger.info(f"Detection completed: {len(instances)} unique instances found")

        # Re-ID 통계 출력
        reid_stats = detector.get_statistics()
        logger.info(f"Re-ID Statistics: {json.dumps(reid_stats, indent=2)}")

        memory_monitor.log_usage("After detection")

        # 진행률 업데이트: 90%
        video.progress = 90
        video.save(update_fields=['progress'])
        job.progress = 90
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 90})

        # ====================================================================
        # 7. 썸네일 저장 및 Face 모델 생성
        # ====================================================================
        logger.info("Saving thumbnails and creating Face model instances...")
        created_faces = []

        for instance_id, instance_data in instances.items():
            # 썸네일 저장
            thumbnail_filename = f"instance_{instance_id}.jpg"
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)

            # BGR 이미지 저장
            cv2.imwrite(thumbnail_path, instance_data['thumbnail'])

            # URL 생성
            thumbnail_url = f"{settings.MEDIA_URL}faces/thumbnails/{str(video.id)}/{thumbnail_filename}"

            # Face 모델 생성
            face = Face.objects.create(
                video=video,
                instance_id=instance_id,
                track_ids=instance_data['track_ids'],
                thumbnail_url=thumbnail_url,
                embedding=instance_data['embedding'],  # JSON list
                quality_score=instance_data['quality'],
                frame_index=instance_data['frame_idx'],
                bbox=list(instance_data['bbox']),
                total_frames=instance_data['total_frames'],
                is_blurred=True,  # 기본값: 모두 블러 처리, 사용자가 선택 해제
                frame_data=instance_data.get('frame_data', {})  # 프레임별 bbox 데이터
            )

            logger.info(
                f"Created Face instance {instance_id}: "
                f"quality={instance_data['quality']:.1f}, "
                f"tracks={len(instance_data['track_ids'])}, "
                f"frames={instance_data['total_frames']}"
            )
            created_faces.append(face)

        logger.info(f"Created {len(created_faces)} Face instances")

        # 진행률 업데이트: 95%
        video.progress = 95
        video.save(update_fields=['progress'])
        job.progress = 95
        job.save(update_fields=['progress'])
        self.update_state(state='PROGRESS', meta={'progress': 95})

        # ====================================================================
        # 8. Video 상태 업데이트: analyzing → ready
        # ====================================================================
        video.status = 'ready'
        video.progress = 100
        video.save(update_fields=['status', 'progress'])

        logger.info(f"Video {video_id} status changed to 'ready'")

        # ====================================================================
        # 9. ProcessingJob 완료 처리
        # ====================================================================
        job.status = 'success'
        job.progress = 100
        job.result_data = {
            'faces_count': len(created_faces),
            'face_ids': [str(f.id) for f in created_faces]
        }
        job.save(update_fields=['status', 'progress', 'result_data'])

        logger.info("=" * 80)
        logger.info(
            f"✅ Face analysis completed successfully for video {video_id}"
        )
        logger.info("=" * 80)

        return {
            'video_id': str(video.id),
            'faces_count': len(created_faces),
            'status': 'success'
        }

    except (FileNotFoundError, ValueError) as exc:
        # ====================================================================
        # 복구 불가능한 에러 (파일 없음, 모델 손상 등)
        # ====================================================================
        logger.error(
            f"[Task {self.request.id}] Unrecoverable error for video {video_id}: {exc}",
            exc_info=True
        )

        # Video 상태 업데이트
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.save(update_fields=['status'])
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

        # ProcessingJob 업데이트
        try:
            job = ProcessingJob.objects.filter(
                video_id=video_id,
                job_type='face_analysis'
            ).order_by('-started_at').first()

            if job:
                job.status = 'failure'
                job.error_message = str(exc)
                job.save(update_fields=['status', 'error_message'])
        except Exception as e:
            logger.error(f"Failed to update ProcessingJob: {e}")

        # 재시도하지 않고 즉시 실패 처리
        raise exc

    except Exception as exc:
        # ====================================================================
        # 일시적 에러 (네트워크, GPU 메모리 등) - 재시도 가능
        # ====================================================================
        logger.error(
            f"[Task {self.request.id}] Face analysis failed for video {video_id}: {exc}",
            exc_info=True
        )

        # Video 상태 업데이트
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.save(update_fields=['status'])
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

        # ProcessingJob 업데이트
        try:
            job = ProcessingJob.objects.filter(
                video_id=video_id,
                job_type='face_analysis'
            ).order_by('-started_at').first()

            if job:
                job.status = 'failure'
                job.error_message = str(exc)
                job.save(update_fields=['status', 'error_message'])
        except Exception as e:
            logger.error(f"Failed to update ProcessingJob: {e}")

        # Celery 재시도 (최대 1회, 지수 백오프)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        else:
            # 재시도 횟수 초과 시 최종 실패 처리
            raise exc


@shared_task(bind=True, max_retries=3)
def process_video_blur_task(self, video_id: str):
    """
    비디오 블러 처리 Celery Task

    처리 흐름:
    1. Video 및 Face 데이터 로드
    2. VideoBlurrer 초기화
    3. 비디오 처리 (블러링)
    4. 결과 저장 및 상태 업데이트
    """
    from .video_blurring import VideoBlurrer

    logger.info(f"[Task {self.request.id}] Starting video blur processing for video {video_id}")

    # 메모리 모니터 초기화
    memory_monitor = MemoryMonitor(limit_gb=7.0, warning_threshold=0.85)
    memory_monitor.log_usage("Blur task start")

    try:
        # 1. Video 객체 로드
        try:
            video = Video.objects.get(id=video_id)
        except Video.DoesNotExist:
            raise ValueError(f"Video {video_id} does not exist")

        # 상태 업데이트: processing
        video.status = 'processing'
        video.progress = 0
        video.save(update_fields=['status', 'progress'])

        # ProcessingJob 생성/업데이트
        job, created = ProcessingJob.objects.get_or_create(
            video=video,
            job_type='video_processing',
            defaults={
                'status': 'started',
                'progress': 0,
                'celery_task_id': self.request.id
            }
        )

        if not created:
            job.status = 'started'
            job.progress = 0
            job.celery_task_id = self.request.id
            job.save(update_fields=['status', 'progress', 'celery_task_id'])

        # 2. 파일 경로 준비
        video_filename = video.original_file_url.split('/')[-1]
        video_path = os.path.join(settings.MEDIA_ROOT, 'videos', video_filename)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 출력 파일 경로 설정
        output_filename = f"processed_{video_filename}"
        output_path = os.path.join(settings.MEDIA_ROOT, 'videos', 'processed', output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 3. Face Instance 데이터 로드 (프레임별 bbox 포함)
        faces = Face.objects.filter(video=video)
        face_models = []
        for face in faces:
            if face.embedding:
                face_models.append({
                    'id': str(face.id),
                    'instance_id': face.instance_id,
                    'embedding': face.embedding,  # 단일 임베딩만 (최고 품질)
                    'embeddings': [],  # Multi-embedding 비활성화
                    'is_blurred': face.is_blurred,
                    'frame_data': face.frame_data  # 프레임별 bbox 데이터
                })

        logger.info(f"Loaded {len(face_models)} face instance models (with frame_data) for optimized blur")

        # 4. VideoBlurrer 실행
        blurrer = VideoBlurrer(
            yolo_model_path=str(settings.YOLO_FACE_MODEL_PATH),
            device='auto',
            threshold=0.92,  # 매우 엄격한 매칭 (0.92)
            use_multi_embedding=False  # 메인 임베딩만 사용 (더 엄격)
        )

        # 진행률 콜백
        def update_progress(percent: int):
            percent = min(percent, 100)
            video.progress = percent
            video.save(update_fields=['progress'])
            job.progress = percent
            job.save(update_fields=['progress'])
            self.update_state(state='PROGRESS', meta={'progress': percent})

            # 메모리 체크
            if percent % 10 == 0:
                memory_monitor.check_and_cleanup()

        success = blurrer.process_video(
            video_path=video_path,
            output_path=output_path,
            face_models=face_models,
            progress_callback=update_progress,
            blur_type='pixelate',  # 모자이크 효과
            blur_strength=15,  # 모자이크 크기
            threshold=0.92  # 매우 엄격한 매칭 (0.92)
        )

        if not success:
            raise RuntimeError("Video processing failed")

        # 5. 완료 처리
        processed_url = output_path.replace(str(settings.MEDIA_ROOT), settings.MEDIA_URL.rstrip('/'))

        logger.info(f"Saving processed_file_url: {processed_url}")

        video.processed_file_url = processed_url
        video.status = 'completed'
        video.progress = 100
        video.save(update_fields=['processed_file_url', 'status', 'progress'])

        job.status = 'success'
        job.progress = 100
        job.result_data = {'output_path': processed_url}
        job.save(update_fields=['status', 'progress', 'result_data'])

        logger.info("=" * 80)
        logger.info(f"✅ Video blur processing completed successfully")
        logger.info("=" * 80)

        return {
            'video_id': str(video.id),
            'processed_file_url': processed_url,
            'status': 'success'
        }

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Video blur processing failed: {exc}", exc_info=True)

        # 실패 상태 업데이트
        try:
            video = Video.objects.get(id=video_id)
            video.status = 'failed'
            video.save(update_fields=['status'])

            job = ProcessingJob.objects.filter(video=video, job_type='video_processing').first()
            if job:
                job.status = 'failure'
                job.error_message = str(exc)
                job.save(update_fields=['status', 'error_message'])
        except Exception:
            pass

        raise exc
