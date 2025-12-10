# -*- coding: utf-8 -*-
"""
FaceBlur 프로젝트 메인 URL 설정

구조:
- /admin/ : Django Admin 페이지
- /api/ : REST API 엔드포인트
- /api/accounts/ : 사용자 인증 관련 API (Phase 2에서 구현)
- /api/ : 비디오, 얼굴, 작업 관련 API
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django Admin
    path('admin/', admin.site.urls),

    # DRF 기본 인증 (로그인/로그아웃 페이지)
    # 개발 중 테스트용 - 브라우저에서 API 테스트 가능
    path('api-auth/', include('rest_framework.urls')),

    # 비디오 앱 (템플릿 페이지 + API)
    # / - 업로드 페이지
    # /video/<id>/select/ - 얼굴 선택 페이지
    # /video/<id>/preview/ - 미리보기 페이지
    # /api/videos/, /api/faces/, /api/jobs/ - API 엔드포인트
    path('', include('apps.videos.urls')),

    # 사용자 인증 API (Phase 2에서 구현)
    # path('api/accounts/', include('apps.accounts.urls')),
]

# 개발 환경에서 미디어 파일 서빙
# 프로덕션에서는 Nginx 또는 S3가 처리
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
