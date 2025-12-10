# ============================================================================
# Accounts App Configuration
# ============================================================================
# Django 앱 설정 파일
# - 앱의 이름과 기본 설정을 정의합니다
# - settings.py의 INSTALLED_APPS에 등록할 때 사용됩니다
# ============================================================================

from django.apps import AppConfig


class AccountsConfig(AppConfig):
    """
    사용자 관리 앱 설정

    이 앱은 다음 기능을 제공합니다:
    - 사용자 회원가입, 로그인, 로그아웃
    - 프로필 관리
    - JWT 토큰 기반 인증 (선택사항)
    """

    # BigAutoField: 기본 키(Primary Key)의 타입 설정
    # - Django 3.2부터 권장되는 설정
    # - 32비트 int 대신 64비트 bigint 사용 (더 많은 레코드 저장 가능)
    default_auto_field = 'django.db.models.BigAutoField'

    # 앱 이름 (apps.accounts)
    # - settings.py의 INSTALLED_APPS에 'apps.accounts' 또는
    #   'apps.accounts.apps.AccountsConfig'로 등록
    name = 'apps.accounts'

    # 앱의 사람이 읽기 쉬운 이름 (Admin 페이지에 표시됨)
    verbose_name = '사용자 관리'
