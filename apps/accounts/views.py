# ============================================================================
# Accounts App - Views
# ============================================================================
# 사용자 관리 관련 View 함수/클래스
# - 회원가입, 로그인, 로그아웃, 프로필 조회/수정
# ============================================================================

from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny


# ============================================================================
# User ViewSet (DRF)
# ============================================================================
# Django REST Framework의 ViewSet을 사용하여 RESTful API 구현
# ViewSet은 CRUD 작업을 자동으로 생성해줍니다.

class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    사용자 정보 조회 API

    제공하는 엔드포인트:
    - GET /api/users/          : 사용자 목록 조회 (관리자만)
    - GET /api/users/{id}/     : 특정 사용자 조회
    - GET /api/users/me/       : 현재 로그인한 사용자 정보 조회

    Permission:
    - 목록/상세 조회: 인증 필요
    - 자기 정보 조회: 인증 필요
    """

    queryset = User.objects.all()
    permission_classes = [IsAuthenticated]

    # Serializer는 apps/accounts/serializers.py에서 정의 (다음 단계에서 작성)
    # serializer_class = UserSerializer

    @action(detail=False, methods=['get'])
    def me(self, request):
        """
        현재 로그인한 사용자 정보 조회

        GET /api/users/me/

        Returns:
            {
                "id": 1,
                "username": "john",
                "email": "john@example.com",
                "profile": {
                    "phone_number": "010-1234-5678"
                }
            }
        """
        # serializer = self.get_serializer(request.user)
        # return Response(serializer.data)

        # Serializer 작성 전 임시 응답
        return Response({
            'id': request.user.id,
            'username': request.user.username,
            'email': request.user.email,
            'message': 'Serializer 작성 후 전체 데이터를 반환합니다.'
        })


# ============================================================================
# 회원가입 View (향후 구현)
# ============================================================================
# from rest_framework.views import APIView
#
# class RegisterView(APIView):
#     """
#     회원가입 API
#
#     POST /api/accounts/register/
#     Body:
#         {
#             "username": "newuser",
#             "email": "newuser@example.com",
#             "password": "securepassword123"
#         }
#     """
#     permission_classes = [AllowAny]
#
#     def post(self, request):
#         # 회원가입 로직 구현
#         pass


# ============================================================================
# 로그인/로그아웃 View
# ============================================================================
# Django REST Framework의 기본 인증을 사용하거나,
# djangorestframework-simplejwt를 사용하여 JWT 토큰 기반 인증 구현 가능

# 예시 (SimpleJWT 사용 시):
# from rest_framework_simplejwt.views import (
#     TokenObtainPairView,
#     TokenRefreshView,
# )
#
# urlpatterns = [
#     path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
#     path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
# ]


# ============================================================================
# 참고:
#
# 1. ViewSet의 장점:
#    - CRUD 작업을 자동으로 생성 (list, retrieve, create, update, destroy)
#    - Router와 함께 사용하여 URL을 자동 생성
#
# 2. Permission Classes:
#    - AllowAny: 누구나 접근 가능
#    - IsAuthenticated: 로그인한 사용자만
#    - IsAdminUser: 관리자만
#    - IsAuthenticatedOrReadOnly: 읽기는 누구나, 쓰기는 로그인 필요
#
# 3. Custom Actions:
#    - @action 데코레이터로 커스텀 엔드포인트 추가
#    - detail=True: /users/{id}/action_name/
#    - detail=False: /users/action_name/
# ============================================================================
