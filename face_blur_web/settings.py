# ============================================================================
# FaceBlur Project - Django Settings
# ============================================================================
# Django 프로젝트 설정 파일
#
# 이 파일은 개발 환경과 운영 환경을 모두 지원합니다.
# 환경 변수(.env 파일)를 통해 설정을 변경할 수 있습니다.
#
# 설정 우선순위:
# 1. 환경 변수 (.env 파일)
# 2. 이 파일의 기본값
# ============================================================================

from pathlib import Path
import os
from decouple import config, Csv  # python-decouple로 환경 변수 로드

# ============================================================================
# 기본 경로 설정
# ============================================================================
# BASE_DIR: 프로젝트 루트 디렉토리
# 예: /app 또는 C:\Users\이승복\Documents\face_blur_project
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================================
# 보안 설정
# ============================================================================
# SECRET_KEY: Django 암호화 키
# - 세션, 쿠키, CSRF 토큰 암호화에 사용
# - 운영 환경에서는 반드시 환경 변수로 설정!
# - 기본값: 개발용 키 (운영에서 사용 금지!)
SECRET_KEY = config(
    'SECRET_KEY',
    default='django-insecure-+28v39p5_v*)6$9vw)5h7_@*@7mdi$e26fyr#t^us7&p!yabso'
)

# DEBUG 모드
# - True: 개발 환경 (상세한 에러 메시지 표시)
# - False: 운영 환경 (에러 페이지 숨김)
DEBUG = config('DEBUG', default=True, cast=bool)

# 접속 허용 호스트
# - 개발: localhost, 127.0.0.1
# - ngrok: *.ngrok-free.app
# - 운영: yourdomain.com, www.yourdomain.com, EC2 IP
ALLOWED_HOSTS = config(
    'ALLOWED_HOSTS',
    default='localhost,127.0.0.1',
    cast=Csv()  # 쉼표로 구분된 문자열을 리스트로 변환
)

# ngrok 지원 (와일드카드 도메인 허용)
if DEBUG:
    ALLOWED_HOSTS += ['*.ngrok-free.app', '*.ngrok-free.dev', '.ngrok.io']

# CSRF Trusted Origins (ngrok 지원)
CSRF_TRUSTED_ORIGINS = config(
    'CSRF_TRUSTED_ORIGINS',
    default='',
    cast=Csv()
)
if DEBUG:
    # ngrok 도메인을 신뢰할 수 있는 origin으로 추가
    CSRF_TRUSTED_ORIGINS += [
        'https://*.ngrok-free.app',
        'https://*.ngrok-free.dev',
        'https://*.ngrok.io'
    ]

# ============================================================================
# 애플리케이션 정의
# ============================================================================
INSTALLED_APPS = [
    # Django 기본 앱
    'django.contrib.admin',           # Admin 페이지
    'django.contrib.auth',            # 인증 시스템
    'django.contrib.contenttypes',    # 컨텐츠 타입 프레임워크
    'django.contrib.sessions',        # 세션 프레임워크
    'django.contrib.messages',        # 메시징 프레임워크
    'django.contrib.staticfiles',     # Static 파일 관리

    # 서드파티 앱
    'rest_framework',                 # Django REST Framework
    'corsheaders',                    # CORS 헤더 지원
    # 'channels',                     # WebSocket 지원 (Phase 4에서 활성화)
    # 'django_celery_beat',           # Celery 스케줄러 (Phase 4에서 활성화)
    # 'storages',                     # AWS S3 스토리지 (운영 환경)

    # 우리 앱
    'apps.accounts.apps.AccountsConfig',      # 사용자 관리
    'apps.videos.apps.VideosConfig',          # 영상 관리
    'apps.processing.apps.ProcessingConfig',  # 영상 처리
]

# ============================================================================
# 미들웨어
# ============================================================================
# 요청/응답 처리 파이프라인
# 순서가 중요합니다!
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',      # 보안 헤더
    'corsheaders.middleware.CorsMiddleware',              # CORS (REST API를 위해 일찍 실행)
    'django.contrib.sessions.middleware.SessionMiddleware',  # 세션 관리
    'django.middleware.common.CommonMiddleware',           # 공통 기능
    'django.middleware.csrf.CsrfViewMiddleware',           # CSRF 보호
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # 인증
    'django.contrib.messages.middleware.MessageMiddleware',     # 메시지
    'django.middleware.clickjacking.XFrameOptionsMiddleware',   # Clickjacking 방지
]

# ============================================================================
# URL 설정
# ============================================================================
ROOT_URLCONF = 'face_blur_web.urls'

# ============================================================================
# 템플릿 설정
# ============================================================================
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        # DIRS: templates 폴더 경로 지정
        'DIRS': [BASE_DIR / 'templates'],  # stitch_ HTML을 여기로 이동 예정
        # APP_DIRS: 각 앱의 templates 폴더도 검색
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',  # request 객체를 템플릿에서 사용
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                # 커스텀 context processor 추가 가능
                # 'django.template.context_processors.media',  # MEDIA_URL을 템플릿에서 사용
            ],
        },
    },
]

# ============================================================================
# WSGI/ASGI 설정
# ============================================================================
# WSGI: HTTP 요청 처리 (Gunicorn)
WSGI_APPLICATION = 'face_blur_web.wsgi.application'

# ASGI: WebSocket 지원 (Daphne) - Phase 4에서 활성화
# ASGI_APPLICATION = 'face_blur_web.asgi.application'

# ============================================================================
# 데이터베이스 설정
# ============================================================================
# 환경 변수에 따라 SQLite 또는 PostgreSQL 선택

DB_ENGINE = config('DB_ENGINE', default='django.db.backends.sqlite3')

if DB_ENGINE == 'django.db.backends.postgresql':
    # PostgreSQL 설정 (운영 환경 또는 Docker)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': config('DB_NAME', default='faceblur_db'),
            'USER': config('DB_USER', default='postgres'),
            'PASSWORD': config('DB_PASSWORD', default='postgres123'),
            'HOST': config('DB_HOST', default='localhost'),
            'PORT': config('DB_PORT', default='5432'),
            # 연결 풀링 설정 (성능 향상)
            'CONN_MAX_AGE': 60,  # 60초 동안 연결 유지
        }
    }
else:
    # SQLite 설정 (개발 환경)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# ============================================================================
# 비밀번호 검증
# ============================================================================
# 비밀번호 강도 검사
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ============================================================================
# 국제화 설정
# ============================================================================
# 언어 및 시간대 설정
LANGUAGE_CODE = 'ko-kr'  # 한국어
TIME_ZONE = 'Asia/Seoul'  # 한국 시간대 (KST)
USE_I18N = True           # 국제화 활성화
USE_TZ = True             # 타임존 사용

# ============================================================================
# Static 파일 설정 (CSS, JavaScript, Images)
# ============================================================================
# Static 파일 URL
STATIC_URL = '/static/'

# Static 파일 디렉토리
# - 개발 중: BASE_DIR/static에서 파일 제공
# - 운영: collectstatic 명령으로 STATIC_ROOT에 수집
STATICFILES_DIRS = [
    BASE_DIR / 'static',  # 프로젝트 공통 static 파일
]

# collectstatic 명령으로 모든 static 파일을 모을 디렉토리
# python manage.py collectstatic
STATIC_ROOT = BASE_DIR / 'staticfiles'

# ============================================================================
# Media 파일 설정 (업로드된 파일)
# ============================================================================
# Media 파일 URL
MEDIA_URL = '/media/'

# Media 파일 저장 경로
# - 로컬 개발: BASE_DIR/media
# - 운영: AWS S3 (아래 S3 설정 참조)
MEDIA_ROOT = BASE_DIR / 'media'

# ============================================================================
# AI 모델 파일 설정 (Phase 3)
# ============================================================================
# AI 모델 파일 저장 디렉토리
MODELS_ROOT = BASE_DIR / 'models'

# YOLO Face Detection 모델 경로
YOLO_FACE_MODEL_PATH = MODELS_ROOT / 'yolov11s-face.pt'

# AdaFace 얼굴 인식 모델 경로
ADAFACE_MODEL_PATH = BASE_DIR / 'apps' / 'videos' / 'weights' / 'adaface_vit_base_kprpe_webface12m.pt'
ADAFACE_ARCHITECTURE = 'vit'  # 'ir_50', 'ir_101', 'vit'

# Face Recognition Model (사용할 모델 선택)
FACE_RECOGNITION_MODEL = 'adaface'  # 'arcface', 'adaface', 'facenet'

# ============================================================================
# AWS S3 설정 (운영 환경)
# ============================================================================
# USE_S3 환경 변수가 True면 S3 사용, False면 로컬 파일 시스템 사용
USE_S3 = config('USE_S3', default=False, cast=bool)

if USE_S3:
    # AWS 자격증명
    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = config('AWS_REGION', default='ap-northeast-2')

    # S3 URL 설정
    AWS_S3_CUSTOM_DOMAIN = f'{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com'

    # S3를 기본 파일 스토리지로 사용
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

    # MEDIA_URL을 S3 URL로 변경
    MEDIA_URL = f'https://{AWS_S3_CUSTOM_DOMAIN}/'

    # S3 추가 설정
    AWS_S3_FILE_OVERWRITE = False  # 같은 이름 파일 덮어쓰기 방지
    AWS_DEFAULT_ACL = 'private'    # 파일 기본 권한 (비공개)
    AWS_S3_SIGNATURE_VERSION = 's3v4'  # 서명 버전

# ============================================================================
# Django REST Framework 설정
# ============================================================================
REST_FRAMEWORK = {
    # 기본 인증 클래스
    # - SessionAuthentication: 세션 기반 인증 (브라우저)
    # - BasicAuthentication: HTTP Basic Auth (개발/테스트용)
    # Phase 3에서 JWT 추가 예정:
    # - 'rest_framework_simplejwt.authentication.JWTAuthentication'
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],

    # 기본 권한 클래스
    # - IsAuthenticated: 로그인 필요
    # - AllowAny: 누구나 접근 가능
    # - IsAdminUser: 관리자만
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],

    # 페이지네이션 설정
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,  # 한 페이지에 20개 항목

    # 기본 렌더러
    # - JSONRenderer: JSON 응답
    # - BrowsableAPIRenderer: DRF 웹 UI (개발 시 편리)
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',  # 운영에서는 제거 권장
    ],

    # 기본 파서
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',  # 파일 업로드 지원
    ],

    # 날짜/시간 형식
    'DATETIME_FORMAT': '%Y-%m-%d %H:%M:%S',
}

# ============================================================================
# CORS 설정 (Cross-Origin Resource Sharing)
# ============================================================================
# 다른 도메인에서 API 호출 허용 설정

# 모든 도메인 허용 (개발 환경에서만!)
# 운영에서는 False로 설정하고 CORS_ALLOWED_ORIGINS 사용
CORS_ALLOW_ALL_ORIGINS = DEBUG

# 허용할 도메인 리스트 (운영 환경)
CORS_ALLOWED_ORIGINS = config(
    'CORS_ALLOWED_ORIGINS',
    default='http://localhost:3000,http://127.0.0.1:3000',
    cast=Csv()
)

# 쿠키 전송 허용 (인증이 필요한 경우)
CORS_ALLOW_CREDENTIALS = True

# ============================================================================
# Redis 설정 (캐시 & Celery)
# ============================================================================
REDIS_URL = config('REDIS_URL', default='redis://localhost:6379/0')

# Django 캐시 백엔드로 Redis 사용
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'faceblur',  # 키 접두사
        'TIMEOUT': 300,             # 기본 캐시 시간 (5분)
    }
}

# 세션을 Redis에 저장 (선택사항 - 성능 향상)
# SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
# SESSION_CACHE_ALIAS = 'default'

# ============================================================================
# Celery 설정 (Phase 4에서 활성화)
# ============================================================================
# Celery 브로커 URL (Redis)
CELERY_BROKER_URL = config('CELERY_BROKER_URL', default=REDIS_URL)

# Celery 결과 백엔드 (Redis)
CELERY_RESULT_BACKEND = config('CELERY_RESULT_BACKEND', default=REDIS_URL)

# Celery 설정
CELERY_ACCEPT_CONTENT = ['json']         # 메시지 형식
CELERY_TASK_SERIALIZER = 'json'          # 작업 직렬화 형식
CELERY_RESULT_SERIALIZER = 'json'        # 결과 직렬화 형식
CELERY_TIMEZONE = TIME_ZONE              # 타임존

# 작업 시간 제한 (초) - 무제한 (사용자 요청)
CELERY_TASK_TIME_LIMIT = None
CELERY_TASK_SOFT_TIME_LIMIT = None

# Redis Visibility Timeout (중요: 작업 시간보다 길어야 함)
# 작업이 길어지면 Redis가 작업 실패로 간주하고 재할당하는 것을 방지
# 24시간으로 설정하여 사실상 무제한 지원
CELERY_BROKER_TRANSPORT_OPTIONS = {
    'visibility_timeout': 86400,  # 24시간
    'fanout_prefix': True,
    'fanout_patterns': True,
}

# 작업 상태 추적
CELERY_TASK_TRACK_STARTED = True

# Face Recognition Model Selection
# Options: 'arcface' (buffalo_l, Default), 'adaface' (ViT/IR50)
FACE_RECOGNITION_MODEL = 'adaface'

# Worker 설정
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1  # 작업 1개 처리 후 워커 재시작 (메모리 누수 방지)
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # 한 번에 1개 작업만 가져옴
CELERY_WORKER_MAX_MEMORY_PER_CHILD = 7168000  # 7GB (킬로바이트 단위)

# 기본 큐 설정 (Celery 기본값은 'celery'이지만 'default' 사용)
CELERY_TASK_DEFAULT_QUEUE = 'default'
CELERY_TASK_DEFAULT_EXCHANGE = 'default'
CELERY_TASK_DEFAULT_ROUTING_KEY = 'default'

# ============================================================================
# 로깅 설정
# ============================================================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': config('LOG_LEVEL', default='INFO'),
        },
        'apps': {  # 우리 앱의 로그
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if DEBUG else 'INFO',
        },
    },
}

# logs 디렉토리 생성
(BASE_DIR / 'logs').mkdir(exist_ok=True)

# ============================================================================
# FastAPI 서버 URL
# ============================================================================
# Django에서 FastAPI AI 서버로 요청을 보낼 때 사용
FASTAPI_BASE_URL = config('FASTAPI_BASE_URL', default='http://localhost:8001')

# ============================================================================
# 업로드 파일 제한
# ============================================================================
# 최대 업로드 크기 (bytes)
# 2GB = 2 * 1024 * 1024 * 1024
MAX_UPLOAD_SIZE = config('MAX_UPLOAD_SIZE', default=2147483648, cast=int)

# 허용 파일 형식
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv']

# ============================================================================
# 보안 설정 (운영 환경)
# ============================================================================
if not DEBUG:
    # HTTPS 강제
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

    # HSTS (HTTP Strict Transport Security)
    SECURE_HSTS_SECONDS = 31536000  # 1년
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

    # X-Frame-Options
    X_FRAME_OPTIONS = 'DENY'

# ============================================================================
# 기본 Auto Field
# ============================================================================
# Django 3.2+에서 권장되는 설정
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================================================
# 설정 완료
# ============================================================================
# 이 설정 파일로 다음 기능이 활성화됩니다:
#
# ✅ 3개 커스텀 앱 (accounts, videos, processing)
# ✅ Django REST Framework (API)
# ✅ CORS 지원 (프론트엔드 통신)
# ✅ PostgreSQL 또는 SQLite 선택 가능
# ✅ Redis 캐시
# ✅ AWS S3 스토리지 (선택적)
# ✅ Celery 준비 (Phase 4에서 활성화)
# ✅ 환경 변수 지원 (.env 파일)
# ✅ 로깅
#
# 다음 단계:
# 1. python manage.py makemigrations
# 2. python manage.py migrate
# 3. python manage.py createsuperuser
# 4. python manage.py runserver
# ============================================================================
