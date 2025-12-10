# ============================================================================
# Videos App - Models
# ============================================================================
# 영상 및 얼굴 데이터 모델 정의
#
# 모델 구조:
#   User (Django 기본) → Video (1:N) → Face (1:N)
#                      ↘ ProcessingJob (1:N)
# ============================================================================

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
import uuid


# ============================================================================
# Video 모델
# ============================================================================
class Video(models.Model):
    """
    업로드된 영상 정보를 저장하는 메인 모델

    워크플로우:
    1. 사용자가 영상 업로드 → status='uploaded'
    2. 얼굴 분석 시작 → status='analyzing'
    3. 분석 완료 (Face 객체 생성) → status='ready'
    4. 사용자가 얼굴 선택 후 처리 시작 → status='processing'
    5. 처리 완료 → status='completed'

    필드 설명:
        id: UUID 기본 키 (예측 불가능한 ID로 보안 강화)
        user: 업로드한 사용자
        title: 영상 제목
        original_file_url: 원본 영상 S3 URL
        processed_file_url: 처리된 영상 S3 URL
        duration: 영상 길이 (초)
        width, height: 해상도
        fps: 초당 프레임 수
        status: 처리 상태
        progress: 진행률 (0-100)
    """

    # ========================================================================
    # 처리 상태 선택지
    # ========================================================================
    STATUS_CHOICES = [
        ('uploaded', '업로드 완료'),       # 업로드만 완료, 분석 전
        ('analyzing', '얼굴 분석 중'),     # 1차 처리: 얼굴 추출 중
        ('ready', '처리 대기'),            # 얼굴 분석 완료, 사용자 선택 대기
        ('processing', '영상 처리 중'),    # 2차 처리: 블러 적용 중
        ('completed', '처리 완료'),        # 최종 영상 생성 완료
        ('failed', '실패'),                # 에러 발생
    ]

    # ========================================================================
    # 기본 정보 필드
    # ========================================================================
    # UUID 기본 키
    # - 순차적인 숫자 ID 대신 UUID 사용
    # - 보안: ID로 다른 사용자의 영상 URL 추측 방지
    # - 분산 시스템에서 ID 충돌 방지
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='영상 ID'
    )

    # 사용자 외래 키
    # - on_delete=models.CASCADE: 사용자 삭제 시 영상도 함께 삭제
    # - related_name='videos': user.videos.all()로 사용자의 모든 영상 조회 가능
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='videos',
        verbose_name='업로드 사용자'
    )

    # 영상 제목
    title = models.CharField(
        max_length=255,
        verbose_name='영상 제목',
        help_text='사용자가 입력한 영상 제목 또는 파일명'
    )

    # 원본 파일명
    original_filename = models.CharField(
        max_length=255,
        verbose_name='원본 파일명',
        help_text='업로드된 파일의 원래 이름'
    )

    # ========================================================================
    # 파일 URL 필드 (S3 또는 로컬)
    # ========================================================================
    # 원본 영상 URL
    # - S3: https://faceblur-videos.s3.ap-northeast-2.amazonaws.com/original/user123/video456.mp4
    # - 로컬: /media/videos/original/user123/video456.mp4
    original_file_url = models.URLField(
        max_length=500,
        verbose_name='원본 파일 URL',
        help_text='S3 또는 로컬 저장소의 원본 영상 경로'
    )

    # 처리된 영상 URL
    processed_file_url = models.URLField(
        max_length=500,
        blank=True,
        null=True,
        verbose_name='처리된 파일 URL',
        help_text='블러 처리가 완료된 영상 경로'
    )

    # ========================================================================
    # 영상 메타데이터
    # ========================================================================
    # 영상 길이 (초)
    # - FFmpeg 또는 MoviePy로 추출
    # - 예: 123.45 (2분 3.45초)
    duration = models.FloatField(
        verbose_name='영상 길이 (초)',
        help_text='영상 전체 재생 시간'
    )

    # 해상도 (가로 × 세로)
    width = models.IntegerField(
        verbose_name='가로 해상도',
        help_text='영상 가로 픽셀 수 (예: 1920)'
    )

    height = models.IntegerField(
        verbose_name='세로 해상도',
        help_text='영상 세로 픽셀 수 (예: 1080)'
    )

    # FPS (Frames Per Second)
    fps = models.FloatField(
        verbose_name='프레임 레이트',
        help_text='초당 프레임 수 (예: 30.0, 60.0)'
    )

    # 파일 크기 (bytes)
    file_size = models.BigIntegerField(
        verbose_name='파일 크기 (bytes)',
        help_text='원본 파일 크기 (바이트 단위)'
    )

    # ========================================================================
    # 처리 상태 필드
    # ========================================================================
    # 현재 상태
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='uploaded',
        verbose_name='처리 상태',
        db_index=True  # 조회 성능 향상을 위한 인덱스
    )

    # 진행률 (0~100)
    progress = models.IntegerField(
        default=0,
        verbose_name='처리 진행률 (%)',
        help_text='현재 작업 진행률 (0-100)'
    )

    # 에러 메시지
    error_message = models.TextField(
        blank=True,
        null=True,
        verbose_name='에러 메시지',
        help_text='처리 실패 시 에러 내용'
    )

    # ========================================================================
    # 타임스탬프 필드
    # ========================================================================
    # 생성 날짜
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='업로드 일시',
        db_index=True
    )

    # 수정 날짜
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name='수정 일시'
    )

    # 처리 완료 날짜
    completed_at = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='처리 완료 일시',
        help_text='영상 처리가 완료된 시각'
    )

    # 만료 날짜 (7일 후 자동 삭제)
    expires_at = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='파일 만료 일시',
        help_text='이 날짜 이후 파일 자동 삭제 (7일)'
    )

    # ========================================================================
    # 메타 정보
    # ========================================================================
    class Meta:
        db_table = 'videos'
        verbose_name = '영상'
        verbose_name_plural = '영상 목록'
        ordering = ['-created_at']  # 최신 순 정렬

        # 인덱스 설정 (조회 성능 향상)
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]

    # ========================================================================
    # 메서드
    # ========================================================================
    def __str__(self):
        """객체의 문자열 표현"""
        return f"{self.title} ({self.user.username}) - {self.get_status_display()}"

    def save(self, *args, **kwargs):
        """
        저장 시 자동 처리

        - 처리 완료 시 만료일 설정 (7일 후)
        - completed_at 자동 설정
        """
        # 상태가 completed로 변경되면
        if self.status == 'completed':
            # 완료 시각 설정 (아직 설정되지 않았다면)
            if not self.completed_at:
                self.completed_at = timezone.now()

            # 만료일 설정 (7일 후)
            if not self.expires_at:
                self.expires_at = timezone.now() + timedelta(days=7)

        super().save(*args, **kwargs)

    def get_file_size_mb(self):
        """파일 크기를 MB 단위로 반환"""
        return round(self.file_size / (1024 * 1024), 2)

    def get_resolution_str(self):
        """해상도를 문자열로 반환 (예: 1920x1080)"""
        return f"{self.width}x{self.height}"


# ============================================================================
# Face 모델
# ============================================================================
class Face(models.Model):
    """
    영상에서 발견된 고유 얼굴 Instance

    새로운 Instance 기반 시스템:
    1. YOLO Face v11s로 얼굴 감지
    2. BoTSORT로 tracking (track_id 부여)
    3. CVLface alignment
    4. AdaFace ViT-12M으로 임베딩 추출
    5. Two-Stage Re-ID로 instance_id 할당
       - 장면 전환이나 화면 밖 이탈 시에도 같은 사람은 같은 instance
    6. 각 instance의 최고 품질 썸네일만 저장

    필드:
        video: 영상 외래 키
        instance_id: Re-ID로 할당된 고유 instance 번호 (같은 사람은 항상 같은 ID)
        track_ids: 이 instance에 속한 모든 BoTSORT track ID들
        thumbnail_url: 최고 품질 얼굴 이미지 S3 URL
        embedding: 512차원 임베딩 벡터 (단일, 최고 품질)
        quality_score: Laplacian variance 품질 점수
        frame_index: 썸네일이 추출된 프레임 번호
        total_frames: 영상 내 등장한 총 프레임 수
        is_blurred: 블러 처리 여부 (사용자 선택)
    """

    # ========================================================================
    # 기본 정보
    # ========================================================================
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='얼굴 ID'
    )

    # 영상 외래 키
    # - on_delete=models.CASCADE: 영상 삭제 시 얼굴도 함께 삭제
    # - related_name='faces': video.faces.all()로 영상의 모든 얼굴 조회
    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='faces',
        verbose_name='영상'
    )

    # Instance ID (Re-ID 시스템이 할당)
    # - 같은 사람은 항상 같은 instance_id
    # - 장면 전환이나 화면 밖 이탈 후 재등장해도 유지됨
    # - 영상 내에서 고유 (unique_together)
    instance_id = models.IntegerField(
        verbose_name='Instance ID',
        help_text='Re-ID 시스템이 할당한 고유 instance 번호 (같은 사람은 항상 같은 ID)',
        db_index=True
    )

    # Track IDs (JSON 배열)
    # - 이 instance에 속한 모든 BoTSORT track_id들
    # - 예: [1, 5, 12] → 3번의 다른 track에서 등장했지만 같은 사람으로 인식됨
    track_ids = models.JSONField(
        default=list,
        verbose_name='Track ID 목록',
        help_text='이 instance에 속한 모든 BoTSORT track ID (JSON 배열)'
    )

    # ========================================================================
    # 얼굴 이미지 정보
    # ========================================================================
    # 최고 품질 썸네일 URL
    # - S3: https://faceblur-videos.s3.amazonaws.com/thumbnails/video123/instance_0.jpg
    # - 가장 선명한(Laplacian variance 최대) 얼굴 이미지
    thumbnail_url = models.URLField(
        max_length=500,
        verbose_name='썸네일 URL',
        help_text='최고 품질 얼굴 이미지 경로'
    )

    # 얼굴 임베딩 벡터 (512차원, 단일)
    # - AdaFace ViT-12M으로 추출한 특징 벡터
    # - JSON 형식으로 저장: [0.123, -0.456, ...]
    # - 블러 처리 시 이 임베딩과 비교 (코사인 유사도)
    # - 품질이 가장 좋은 프레임의 임베딩만 저장
    embedding = models.JSONField(
        verbose_name='임베딩 벡터',
        help_text='AdaFace 512차원 특징 벡터 (최고 품질, JSON 배열)'
    )

    # 품질 점수 (Laplacian variance)
    # - 높을수록 선명함
    # - 썸네일 선택 및 Re-ID 업데이트 기준
    quality_score = models.FloatField(
        verbose_name='품질 점수',
        help_text='Laplacian variance 선명도 점수 (높을수록 선명)',
        db_index=True
    )

    # 썸네일 프레임 번호
    # - 이 썸네일이 추출된 프레임의 위치
    frame_index = models.IntegerField(
        verbose_name='썸네일 프레임 번호',
        help_text='최고 품질 썸네일이 추출된 프레임 번호'
    )

    # Bounding Box (JSON)
    # - 썸네일 프레임에서의 얼굴 위치 [x1, y1, x2, y2]
    bbox = models.JSONField(
        default=list,
        verbose_name='Bounding Box',
        help_text='얼굴 위치 좌표 [x1, y1, x2, y2]'
    )

    # Frame-level tracking data (JSON)
    # - 각 프레임에서 이 instance의 bbox 위치
    # - 구조: {frame_idx: [x1, y1, x2, y2, confidence], ...}
    # - 예: {0: [100, 50, 200, 150, 0.95], 1: [102, 51, 201, 151, 0.96], ...}
    # - 블러 처리 시 YOLO 재실행 없이 이 데이터 사용
    frame_data = models.JSONField(
        default=dict,
        verbose_name='프레임별 Tracking 데이터',
        help_text='각 프레임에서의 bbox 위치 (JSON 객체)',
        blank=True
    )

    # ========================================================================
    # 통계 정보
    # ========================================================================
    # 등장 프레임 수
    # - 이 instance가 등장한 총 프레임 수
    # - Track ID가 여러 개여도 총합
    total_frames = models.IntegerField(
        default=0,
        verbose_name='등장 프레임 수',
        help_text='영상 내 이 instance가 등장한 총 프레임 수'
    )

    # ========================================================================
    # 사용자 선택 정보
    # ========================================================================
    # 블러 처리 여부
    # - True: 이 얼굴을 블러 처리 (기본값)
    # - False: 이 얼굴은 블러 처리하지 않음 (사용자 선택)
    is_blurred = models.BooleanField(
        default=True,
        verbose_name='블러 처리 여부',
        help_text='True면 블러 처리, False면 원본 유지'
    )

    # 생성 날짜
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='생성 일시'
    )

    # ========================================================================
    # 메타 정보
    # ========================================================================
    class Meta:
        db_table = 'faces'
        verbose_name = '얼굴 Instance'
        verbose_name_plural = '얼굴 Instance 목록'
        ordering = ['instance_id']

        # 복합 유니크 제약
        # - 같은 영상에서 같은 instance_id는 중복 불가
        unique_together = ['video', 'instance_id']

        # 인덱스
        indexes = [
            models.Index(fields=['video', 'instance_id']),
            models.Index(fields=['quality_score']),
        ]

    def __str__(self):
        """객체의 문자열 표현"""
        blur_status = "블러" if self.is_blurred else "원본"
        return f"Instance {self.instance_id} in {self.video.title} ({blur_status}, {len(self.track_ids)} tracks)"


# ============================================================================
# ProcessingJob 모델
# ============================================================================
class ProcessingJob(models.Model):
    """
    Celery 비동기 작업 추적 모델

    사용 목적:
    - Celery 작업의 상태를 데이터베이스에 저장
    - 프론트엔드에서 진행률 조회
    - 에러 발생 시 로그 저장

    작업 종류:
    1. face_analysis: 얼굴 분석 (1차 처리)
    2. video_processing: 영상 블러 처리 (2차 처리)
    """

    # ========================================================================
    # 작업 종류 선택지
    # ========================================================================
    JOB_TYPE_CHOICES = [
        ('face_analysis', '얼굴 분석'),
        ('video_processing', '영상 처리'),
    ]

    STATUS_CHOICES = [
        ('pending', '대기 중'),
        ('started', '실행 중'),
        ('success', '성공'),
        ('failure', '실패'),
        ('retry', '재시도 중'),
        ('revoked', '취소됨'),
    ]

    # ========================================================================
    # 기본 정보
    # ========================================================================
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name='작업 ID'
    )

    # 영상 외래 키
    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='processing_jobs',
        verbose_name='영상'
    )

    # 작업 종류
    job_type = models.CharField(
        max_length=50,
        choices=JOB_TYPE_CHOICES,
        verbose_name='작업 종류'
    )

    # ========================================================================
    # Celery 관련 정보
    # ========================================================================
    # Celery 작업 ID
    # - Celery가 반환하는 task_id를 저장
    # - 이 ID로 Celery 작업 상태 조회 가능
    celery_task_id = models.CharField(
        max_length=255,
        unique=True,
        verbose_name='Celery 작업 ID',
        help_text='Celery에서 생성한 고유 작업 ID'
    )

    # 작업 상태
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        verbose_name='작업 상태',
        db_index=True
    )

    # 진행률 (0~100)
    progress = models.IntegerField(
        default=0,
        verbose_name='진행률 (%)'
    )

    # ========================================================================
    # 결과 정보
    # ========================================================================
    # 작업 결과 데이터 (JSON)
    # - 성공 시: {"face_count": 3, "duration": 12.5}
    # - 실패 시: {"error": "FastAPI connection failed"}
    result_data = models.JSONField(
        blank=True,
        null=True,
        verbose_name='결과 데이터',
        help_text='작업 완료 후 결과 정보 (JSON)'
    )

    # 에러 메시지
    error_message = models.TextField(
        blank=True,
        null=True,
        verbose_name='에러 메시지'
    )

    # ========================================================================
    # 타임스탬프
    # ========================================================================
    started_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='시작 일시'
    )

    completed_at = models.DateTimeField(
        blank=True,
        null=True,
        verbose_name='완료 일시'
    )

    # ========================================================================
    # 메타 정보
    # ========================================================================
    class Meta:
        db_table = 'processing_jobs'
        verbose_name = '처리 작업'
        verbose_name_plural = '처리 작업 목록'
        ordering = ['-started_at']

        indexes = [
            models.Index(fields=['video', '-started_at']),
            models.Index(fields=['celery_task_id']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        """객체의 문자열 표현"""
        return f"{self.get_job_type_display()} - {self.video.title} ({self.get_status_display()})"


# ============================================================================
# 사용 예시:
#
# # 영상 생성
# video = Video.objects.create(
#     user=request.user,
#     title="내 영상",
#     original_filename="video.mp4",
#     original_file_url="https://s3.../video.mp4",
#     duration=120.5,
#     width=1920,
#     height=1080,
#     fps=30.0,
#     file_size=52428800  # 50MB
# )
#
# # 얼굴 생성
# face = Face.objects.create(
#     video=video,
#     face_index=1,
#     thumbnail_url="https://s3.../face_1.jpg",
#     embedding=[0.1, 0.2, ...],  # 512개 값
#     appearance_count=15,
#     first_frame=0,
#     last_frame=3600
# )
#
# # 조회
# user_videos = request.user.videos.filter(status='completed')
# video_faces = video.faces.filter(is_blurred=True)
# ============================================================================
