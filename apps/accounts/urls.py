# ============================================================================
# Accounts App - URLs
# ============================================================================
# 사용자 관리 관련 URL 라우팅
# ============================================================================

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet

# ============================================================================
# DRF Router 설정
# ============================================================================
# DefaultRouter는 ViewSet을 위한 URL 패턴을 자동으로 생성합니다.
#
# UserViewSet 등록 시 자동 생성되는 URL:
# - GET    /users/          : 사용자 목록
# - POST   /users/          : 사용자 생성 (허용하지 않음 - ReadOnlyModelViewSet)
# - GET    /users/{id}/     : 특정 사용자 조회
# - PUT    /users/{id}/     : 사용자 수정 (허용하지 않음)
# - DELETE /users/{id}/     : 사용자 삭제 (허용하지 않음)
# - GET    /users/me/       : 현재 사용자 조회 (커스텀 action)

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

# ============================================================================
# URL Patterns
# ============================================================================
# 이 URL들은 face_blur_web/urls.py에서 include됩니다.
# 예: path('api/accounts/', include('apps.accounts.urls'))

urlpatterns = [
    # Router가 생성한 URL 포함
    path('', include(router.urls)),

    # 추가 인증 관련 URL (향후 구현)
    # path('register/', RegisterView.as_view(), name='register'),
    # path('login/', LoginView.as_view(), name='login'),
    # path('logout/', LogoutView.as_view(), name='logout'),
]

# ============================================================================
# 최종 URL 구조:
#
# /api/accounts/users/          (GET)  - 사용자 목록
# /api/accounts/users/{id}/     (GET)  - 특정 사용자
# /api/accounts/users/me/       (GET)  - 현재 로그인 사용자
# /api/accounts/register/       (POST) - 회원가입 (향후)
# /api/accounts/login/          (POST) - 로그인 (향후)
# /api/accounts/logout/         (POST) - 로그아웃 (향후)
# ============================================================================
