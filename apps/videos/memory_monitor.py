# -*- coding: utf-8 -*-
"""
메모리 모니터링 및 관리 유틸리티
"""

import psutil
import logging
import gc
from typing import Optional
from django.core.cache import cache

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """메모리 사용량 모니터링 및 자동 정리"""

    def __init__(self, limit_gb: float = 7.0, warning_threshold: float = 0.85):
        """
        Args:
            limit_gb: 메모리 제한 (GB)
            warning_threshold: 경고 임계값 (limit의 비율)
        """
        self.limit_gb = limit_gb
        self.limit_bytes = limit_gb * 1024 ** 3
        self.warning_bytes = self.limit_bytes * warning_threshold
        self.process = psutil.Process()

    def get_current_usage(self) -> dict:
        """현재 메모리 사용량 반환"""
        mem_info = self.process.memory_info()
        rss_bytes = mem_info.rss
        rss_gb = rss_bytes / (1024 ** 3)

        return {
            'rss_bytes': rss_bytes,
            'rss_gb': round(rss_gb, 2),
            'limit_gb': self.limit_gb,
            'usage_percent': round((rss_bytes / self.limit_bytes) * 100, 1)
        }

    def check_and_cleanup(self, force: bool = False) -> dict:
        """메모리 체크 및 필요 시 정리"""
        usage = self.get_current_usage()

        # 경고 임계값 초과
        if usage['rss_bytes'] > self.warning_bytes or force:
            logger.warning(
                f"High memory usage: {usage['rss_gb']:.2f} GB "
                f"({usage['usage_percent']:.1f}% of limit)"
            )

            # Python GC 실행
            collected = gc.collect()
            logger.info(f"Garbage collected {collected} objects")

            # PyTorch CUDA 캐시 정리
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass

            # 정리 후 사용량
            new_usage = self.get_current_usage()
            freed_gb = usage['rss_gb'] - new_usage['rss_gb']

            logger.info(
                f"Memory after cleanup: {new_usage['rss_gb']:.2f} GB "
                f"(freed {freed_gb:.2f} GB)"
            )

            return {
                'cleaned': True,
                'before': usage,
                'after': new_usage,
                'freed_gb': round(freed_gb, 2)
            }

        return {
            'cleaned': False,
            'usage': usage
        }

    def is_limit_exceeded(self) -> bool:
        """메모리 제한 초과 여부"""
        usage = self.get_current_usage()
        return usage['rss_bytes'] > self.limit_bytes

    def log_usage(self, context: str = ""):
        """메모리 사용량 로깅"""
        usage = self.get_current_usage()
        logger.info(
            f"Memory usage [{context}]: {usage['rss_gb']:.2f} GB "
            f"({usage['usage_percent']:.1f}%)"
        )


class MemoryCache:
    """Redis 기반 진행 상황 캐시"""

    @staticmethod
    def save_progress(video_id: str, data: dict, timeout: int = 3600):
        """진행 상황 저장"""
        cache_key = f"video_progress:{video_id}"
        cache.set(cache_key, data, timeout)

    @staticmethod
    def get_progress(video_id: str) -> Optional[dict]:
        """진행 상황 조회"""
        cache_key = f"video_progress:{video_id}"
        return cache.get(cache_key)

    @staticmethod
    def clear_progress(video_id: str):
        """진행 상황 삭제"""
        cache_key = f"video_progress:{video_id}"
        cache.delete(cache_key)
