# -*- coding: utf-8 -*-
"""
Celery 설정 파일

이 파일은 Celery 애플리케이션을 초기화하고 Django 설정과 통합합니다.

Celery란?
- 분산 작업 큐 시스템
- 오래 걸리는 작업(영상 처리, 얼굴 분석)을 백그라운드에서 비동기적으로 실행
- Redis를 메시지 브로커로 사용하여 작업 전달
"""

from __future__ import absolute_import, unicode_literals
import os
import gc
import logging
import multiprocessing
from celery import Celery
from celery.signals import worker_process_init, task_prerun, task_postrun

# CUDA 호환성을 위해 spawn 방식 사용
# fork 방식은 CUDA 재초기화 불가능
# 모듈 재로드 시 RuntimeError 방지를 위해 조건부 설정
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 이미 설정된 경우 무시
    pass

logger = logging.getLogger(__name__)

# Django 설정 모듈 지정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')

# Celery 앱 생성
app = Celery('face_blur_web')

# Django settings.py에서 CELERY_ 접두사가 붙은 설정 자동 로드
# 예: CELERY_BROKER_URL, CELERY_RESULT_BACKEND 등
app.config_from_object('django.conf:settings', namespace='CELERY')

# Django 앱들에서 tasks.py 파일을 자동으로 찾아 등록
# apps/videos/tasks.py, apps/accounts/tasks.py 등
app.autodiscover_tasks()


# Worker 프로세스 초기화 시그널
@worker_process_init.connect
def init_worker(**kwargs):
    """
    Worker 프로세스 초기화 시 실행
    - 메모리 정리
    - GPU 캐시 초기화
    """
    logger.info("Worker process initialized, cleaning up memory")
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    except ImportError:
        pass


# Task 실행 전 시그널
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """
    Task 실행 전 메모리 상태 로깅
    """
    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        logger.info(f"[Task {task_id}] Starting. Memory usage: {memory_gb:.2f} GB")
    except ImportError:
        pass


# Task 실행 후 시그널
@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    """
    Task 실행 후 메모리 정리
    """
    logger.info(f"[Task {task_id}] Completed. Cleaning up memory...")
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        logger.info(f"[Task {task_id}] Memory after cleanup: {memory_gb:.2f} GB")
    except ImportError:
        pass


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """
    디버깅용 테스트 태스크
    Celery 연결 테스트 시 사용:

    >>> from face_blur_web.celery import debug_task
    >>> debug_task.delay()
    """
    print(f'Request: {self.request!r}')
