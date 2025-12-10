# -*- coding: utf-8 -*-
"""
비디오 앱 URL 라우팅
- API 라우팅: DRF Router 사용
- 템플릿 페이지 라우팅: 일반 path 사용
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# DRF Router 설정 (API 엔드포인트용)
router = DefaultRouter()
router.register(r'videos', views.VideoViewSet, basename='video')
router.register(r'faces', views.FaceViewSet, basename='face')
router.register(r'jobs', views.ProcessingJobViewSet, basename='processingjob')

# URL 패턴
urlpatterns = [
    # ========================================================================
    # 템플릿 페이지 (HTML 렌더링)
    # ========================================================================
    path('', views.upload_page, name='upload'),  # 업로드 페이지 (홈)
    path('video/<uuid:pk>/select/', views.select_faces_page, name='select_faces'),  # 얼굴 선택
    path('video/<uuid:pk>/preview/', views.preview_page, name='preview'),  # 미리보기

    # ========================================================================
    # API 엔드포인트 (DRF Router)
    # ========================================================================
    # /api/videos/ - 비디오 목록, 업로드
    # /api/videos/{id}/ - 비디오 상세, 수정, 삭제
    # /api/videos/{id}/start_processing/ - 영상 처리 시작
    # /api/faces/?video_id={id} - 얼굴 목록
    # /api/faces/{id}/ - 얼굴 상세, 블러 선택 변경
    # /api/jobs/?video_id={id} - 처리 작업 목록
    path('api/', include(router.urls)),
]
