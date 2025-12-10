# ============================================================================
# Processing App Configuration
# ============================================================================
# 영상 처리 로직 앱 설정
# ============================================================================

from django.apps import AppConfig


class ProcessingConfig(AppConfig):
    """
    영상 처리 앱 설정

    이 앱은 다음 기능을 제공합니다:
    - FastAPI AI 서버와 통신
    - Celery 비동기 작업 (얼굴 분석, 영상 처리)
    - 영상 프레임 추출 및 블러 처리
    """

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.processing'
    verbose_name = '영상 처리'
