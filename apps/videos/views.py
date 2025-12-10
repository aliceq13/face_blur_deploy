# -*- coding: utf-8 -*-
"""
비디오 앱 Views
- API Views: Django REST Framework ViewSet (API 엔드포인트)
- Template Views: 일반 Django View (HTML 페이지 렌더링)
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.conf import settings
import uuid
import os
from datetime import timedelta
from django.utils import timezone

from .models import Video, Face, ProcessingJob
from .serializers import (
    VideoListSerializer,
    VideoDetailSerializer,
    VideoUploadSerializer,
    FaceSerializer,
    FaceUpdateSerializer,
    VideoProcessSerializer,
    ProcessingJobSerializer
)


# ============================================================================
# 템플릿 렌더링 Views (HTML 페이지)
# ============================================================================

def upload_page(request):
    """
    비디오 업로드 페이지
    GET / - 업로드 페이지 표시
    """
    return render(request, 'videos/upload.html')


def select_faces_page(request, pk):
    """
    얼굴 선택 페이지
    GET /video/<uuid>/select/ - 얼굴 선택 페이지 표시

    Args:
        pk: 비디오 UUID

    Returns:
        HTML 페이지 또는 리다이렉트
    """
    # 비디오 조회 (개발 환경에서는 인증 불필요)
    video = get_object_or_404(Video, id=pk)

    # 상태 확인 - 분석 완료 상태여야 함
    if video.status not in ['ready', 'analyzing']:
        # analyzing 상태면 조금 기다리면 ready가 될 수 있으므로 허용
        pass

    return render(request, 'videos/select_faces.html', {
        'video': video
    })


def preview_page(request, pk):
    """
    비디오 미리보기 및 처리 시작 페이지
    GET /video/<uuid>/preview/ - 미리보기 페이지 표시

    Args:
        pk: 비디오 UUID

    Returns:
        HTML 페이지
    """
    # 비디오 조회 (개발 환경에서는 인증 불필요)
    video = get_object_or_404(Video, id=pk)

    return render(request, 'videos/preview.html', {
        'video': video
    })


# ============================================================================
# API Views (REST Framework ViewSet)
# ============================================================================


class VideoViewSet(viewsets.ModelViewSet):
    """
    비디오 CRUD API ViewSet

    주요 엔드포인트:
    - GET /api/videos/ : 비디오 목록 조회
    - POST /api/videos/ : 비디오 업로드
    - GET /api/videos/{id}/ : 비디오 상세 조회
    - PUT/PATCH /api/videos/{id}/ : 비디오 정보 수정
    - DELETE /api/videos/{id}/ : 비디오 삭제
    - POST /api/videos/{id}/start_processing/ : 영상 처리 시작
    """
    permission_classes = [AllowAny]  # 개발 중: 로그인 없이 테스트 가능
    parser_classes = (MultiPartParser, FormParser, JSONParser)  # 파일 업로드 및 JSON 요청 지원

    def get_queryset(self):
        """
        비디오 목록 조회
        - 개발 모드: 로그인하지 않아도 모든 비디오 조회 가능
        - 운영 모드: 로그인한 사용자의 비디오만 조회
        """
        # 개발 중: 비로그인 사용자도 모든 비디오 조회 가능
        if self.request.user.is_authenticated:
            return Video.objects.filter(user=self.request.user).order_by('-created_at')
        else:
            # 비로그인 사용자는 모든 비디오 볼 수 있음 (개발 모드)
            return Video.objects.all().order_by('-created_at')

    def get_serializer_class(self):
        """
        액션에 따라 다른 Serializer 사용
        - 목록: VideoListSerializer (간단한 정보)
        - 상세: VideoDetailSerializer (전체 정보)
        - 업로드: VideoUploadSerializer
        """
        if self.action == 'list':
            return VideoListSerializer
        elif self.action == 'create':
            return VideoUploadSerializer
        return VideoDetailSerializer

    def create(self, request, *args, **kwargs):
        """
        비디오 업로드 API
        POST /api/videos/

        Request (multipart/form-data):
        - video_file: 영상 파일 (필수)
        - title: 제목 (필수)

        Response:
        - 201 Created: 업로드 성공
        - 400 Bad Request: 유효성 검사 실패
        """
        # 1. 파일 검증
        video_file = request.FILES.get('video_file')
        if not video_file:
            return Response(
                {'error': '비디오 파일이 필요합니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 파일 크기 제한 (예: 500MB)
        max_size = 500 * 1024 * 1024  # 500MB in bytes
        if video_file.size > max_size:
            return Response(
                {'error': f'파일 크기는 {max_size // (1024*1024)}MB를 초과할 수 없습니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 파일 확장자 검증
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        file_ext = os.path.splitext(video_file.name)[1].lower()
        if file_ext not in allowed_extensions:
            return Response(
                {'error': f'지원되는 파일 형식: {", ".join(allowed_extensions)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Serializer 유효성 검사 (title, description)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # 3. 파일 저장 (S3 또는 로컬)
        file_name = f"videos/{uuid.uuid4()}{file_ext}"
        file_path = default_storage.save(file_name, video_file)
        file_url = default_storage.url(file_path)

        # 4. 비디오 메타데이터 추출 (moviepy 사용)
        # TODO: Phase 2에서 구현 - 지금은 기본값 사용
        duration = 0.0
        width = 0
        height = 0
        fps = 0.0

        try:
            from moviepy.editor import VideoFileClip
            # 로컬 파일 경로 얻기
            if settings.USE_S3:
                # S3 사용 시 임시 다운로드 필요 (Phase 2에서 구현)
                pass
            else:
                local_path = default_storage.path(file_path)
                clip = VideoFileClip(local_path)
                duration = clip.duration
                width, height = clip.size
                fps = clip.fps
                clip.close()
        except Exception as e:
            # 메타데이터 추출 실패 시 경고만 하고 진행
            print(f"Warning: Failed to extract video metadata: {e}")

        # 5. Video 모델 생성
        # 로그인 안 한 경우 admin 사용자로 설정 (개발용)
        from django.contrib.auth.models import User
        user = request.user if request.user.is_authenticated else User.objects.get(username='admin')

        video = Video.objects.create(
            user=user,
            title=serializer.validated_data['title'],
            original_file_url=file_url,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            file_size=video_file.size,
            status='uploaded',  # 초기 상태
            expires_at=timezone.now() + timedelta(days=7)  # 7일 후 자동 삭제
        )

        # 6. Celery 얼굴 분석 태스크 트리거 (Phase 3)
        from .tasks import analyze_faces_task
        from .models import ProcessingJob

        task = analyze_faces_task.delay(str(video.id))

        # ProcessingJob 생성
        ProcessingJob.objects.create(
            video=video,
            job_type='face_analysis',
            celery_task_id=task.id,
            status='pending'
        )

        # 7. 응답
        response_serializer = VideoDetailSerializer(video)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        """
        비디오 삭제 API
        DELETE /api/videos/{id}/

        - 비디오와 연관된 파일도 함께 삭제 (원본, 처리본, 썸네일)
        - Face 모델은 CASCADE로 자동 삭제
        """
        video = self.get_object()

        # 파일 삭제
        try:
            if video.original_file_url:
                # URL에서 파일 경로 추출 후 삭제
                file_path = video.original_file_url.split('/')[-1]
                if default_storage.exists(file_path):
                    default_storage.delete(file_path)

            if video.processed_file_url:
                file_path = video.processed_file_url.split('/')[-1]
                if default_storage.exists(file_path):
                    default_storage.delete(file_path)

            if video.thumbnail_url:
                file_path = video.thumbnail_url.split('/')[-1]
                if default_storage.exists(file_path):
                    default_storage.delete(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete files: {e}")

        # DB에서 삭제
        video.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=['post'])
    def start_processing(self, request, pk=None):
        """
        영상 처리 시작 API
        POST /api/videos/{id}/start_processing/

        - 사용자가 얼굴 선택을 완료한 후 호출
        - Celery 비동기 작업으로 영상 처리 시작 (Phase 4에서 구현)
        - 현재는 상태만 변경
        """
        video = self.get_object()

        # Serializer 유효성 검사
        serializer = VideoProcessSerializer(data=request.data, context={'video': video})
        serializer.is_valid(raise_exception=True)

        # Celery 작업 시작
        from .tasks import process_video_blur_task
        task = process_video_blur_task.delay(str(video.id))

        # ProcessingJob 생성
        job = ProcessingJob.objects.create(
            video=video,
            job_type='video_processing',
            celery_task_id=task.id,
            status='pending'
        )

        return Response({
            'message': '영상 처리가 시작되었습니다.',
            'video_id': str(video.id),
            'job_id': str(job.id)
        }, status=status.HTTP_200_OK)


class FaceViewSet(viewsets.ModelViewSet):
    """
    얼굴 정보 API ViewSet

    주요 엔드포인트:
    - GET /api/faces/ : 얼굴 목록 조회
    - GET /api/faces/{id}/ : 얼굴 상세 조회
    - PATCH /api/faces/{id}/ : 블러 처리 여부 변경
    """
    permission_classes = [AllowAny]  # 개발 모드: 로그인 없이 테스트 가능
    serializer_class = FaceSerializer
    pagination_class = None  # 페이지네이션 비활성화 (비디오당 얼굴 수가 적음)

    def get_queryset(self):
        """
        얼굴 목록 조회
        - 개발 모드: 로그인하지 않아도 모든 얼굴 조회 가능
        - 쿼리 파라미터로 비디오 필터링 가능: ?video_id=xxx
        """
        # 개발 중: 비로그인 사용자도 모든 얼굴 조회 가능
        if self.request.user.is_authenticated:
            queryset = Face.objects.filter(video__user=self.request.user)
        else:
            queryset = Face.objects.all()

        # 특정 비디오의 얼굴만 필터링
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)

        return queryset.order_by('instance_id')

    def get_serializer_class(self):
        """
        액션에 따라 다른 Serializer 사용
        - 업데이트: FaceUpdateSerializer (is_blurred만 변경)
        - 기타: FaceSerializer
        """
        if self.action in ['update', 'partial_update']:
            return FaceUpdateSerializer
        return FaceSerializer


class ProcessingJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    처리 작업 조회 API ViewSet (읽기 전용)

    주요 엔드포인트:
    - GET /api/jobs/ : 작업 목록 조회
    - GET /api/jobs/{id}/ : 작업 상세 조회

    작업 생성은 Video API에서 자동으로 처리됨
    """
    permission_classes = [AllowAny]  # 개발 모드: 로그인 없이 테스트 가능
    serializer_class = ProcessingJobSerializer

    def get_queryset(self):
        """
        처리 작업 목록 조회
        - 개발 모드: 로그인하지 않아도 모든 작업 조회 가능
        - 쿼리 파라미터로 비디오 필터링 가능: ?video_id=xxx
        """
        # 개발 중: 비로그인 사용자도 모든 작업 조회 가능
        if self.request.user.is_authenticated:
            queryset = ProcessingJob.objects.filter(video__user=self.request.user)
        else:
            queryset = ProcessingJob.objects.all()

        # 특정 비디오의 작업만 필터링
        video_id = self.request.query_params.get('video_id')
        if video_id:
            queryset = queryset.filter(video_id=video_id)

        return queryset.order_by('-started_at')
