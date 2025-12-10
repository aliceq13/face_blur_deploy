# ============================================================================
# Accounts App - Admin
# ============================================================================
# Django Admin 페이지 설정
# - /admin URL에서 데이터베이스를 쉽게 관리할 수 있는 인터페이스 제공
# ============================================================================

from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import UserProfile


# ============================================================================
# UserProfile Inline Admin
# ============================================================================
# User 모델 편집 화면에 Profile 필드도 함께 표시

class UserProfileInline(admin.StackedInline):
    """
    User Admin 페이지에 Profile을 Inline으로 표시

    - StackedInline: 세로로 배치 (많은 필드가 있을 때)
    - TabularInline: 가로로 배치 (적은 필드가 있을 때)
    """
    model = UserProfile
    can_delete = False
    verbose_name_plural = '프로필 정보'

    # 표시할 필드
    fields = ('phone_number', 'profile_image', 'created_at', 'updated_at')

    # 읽기 전용 필드
    readonly_fields = ('created_at', 'updated_at')


# ============================================================================
# Custom User Admin
# ============================================================================
# Django 기본 User Admin을 확장하여 Profile 정보도 함께 관리

class CustomUserAdmin(BaseUserAdmin):
    """
    확장된 User Admin

    Django 기본 UserAdmin에 Profile Inline을 추가합니다.
    """
    inlines = (UserProfileInline,)

    # 목록 화면에 표시할 컬럼
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined')

    # 필터 옵션
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')

    # 검색 가능한 필드
    search_fields = ('username', 'email', 'first_name', 'last_name')

    # 정렬 순서
    ordering = ('-date_joined',)


# ============================================================================
# UserProfile Admin
# ============================================================================
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """
    UserProfile 모델의 Admin 페이지 설정
    """

    # 목록 화면에 표시할 컬럼
    list_display = ('user', 'phone_number', 'created_at', 'updated_at')

    # 필터 옵션
    list_filter = ('created_at',)

    # 검색 가능한 필드
    search_fields = ('user__username', 'user__email', 'phone_number')

    # 읽기 전용 필드
    readonly_fields = ('created_at', 'updated_at')

    # 정렬 순서
    ordering = ('-created_at',)

    # 상세 화면 필드 그룹핑
    fieldsets = (
        ('사용자 정보', {
            'fields': ('user',)
        }),
        ('추가 정보', {
            'fields': ('phone_number', 'profile_image')
        }),
        ('타임스탬프', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)  # 접혀있는 상태로 표시
        }),
    )


# ============================================================================
# User Admin 재등록
# ============================================================================
# Django는 기본적으로 User Admin을 등록하므로, 먼저 해제하고 커스텀 Admin을 등록합니다.

# 기존 User Admin 해제
admin.site.unregister(User)

# 커스텀 User Admin 등록
admin.site.register(User, CustomUserAdmin)


# ============================================================================
# Admin 사이트 커스터마이징
# ============================================================================
# Admin 페이지 제목 변경

admin.site.site_header = 'FaceBlur 관리자 페이지'
admin.site.site_title = 'FaceBlur Admin'
admin.site.index_title = '관리 대시보드'


# ============================================================================
# 사용 방법:
#
# 1. 슈퍼유저 생성:
#    python manage.py createsuperuser
#    또는
#    docker-compose exec django python manage.py createsuperuser
#
# 2. 개발 서버 실행:
#    python manage.py runserver
#
# 3. 브라우저에서 접속:
#    http://localhost:8000/admin
#
# 4. 로그인 후 사용자 관리 가능
# ============================================================================
