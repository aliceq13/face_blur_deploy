# ============================================================================
# Accounts App - Models
# ============================================================================
# 사용자 관리 관련 데이터베이스 모델
#
# Django는 기본적으로 User 모델을 제공하지만, 필요에 따라 확장 가능합니다.
# 여기서는 기본 User 모델을 사용하고, 향후 확장을 위한 준비를 합니다.
# ============================================================================

from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


# ============================================================================
# UserProfile 모델 (선택사항 - 향후 확장용)
# ============================================================================
# Django 기본 User 모델에 추가 정보를 저장하기 위한 프로필 모델
# User 모델과 1:1 관계를 맺습니다

class UserProfile(models.Model):
    """
    사용자 프로필 확장 모델

    Django 기본 User 모델은 username, email, password 등 기본 정보만 제공합니다.
    추가 정보(프로필 사진, 전화번호 등)를 저장하려면 이 모델을 사용합니다.

    필드:
        user: User 모델과의 1:1 관계
        phone_number: 전화번호
        created_at: 생성 날짜
        updated_at: 수정 날짜
    """

    # User 모델과 1:1 관계
    # - on_delete=models.CASCADE: User 삭제 시 프로필도 함께 삭제
    # - related_name='profile': user.profile로 접근 가능
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name='사용자'
    )

    # 전화번호 (선택사항)
    phone_number = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        verbose_name='전화번호',
        help_text='사용자 연락처'
    )

    # 프로필 이미지 (선택사항)
    # - upload_to: 업로드 경로 지정
    # - blank=True: Form에서 필수 아님
    # - null=True: 데이터베이스에서 NULL 허용
    profile_image = models.ImageField(
        upload_to='profiles/%Y/%m/%d/',
        blank=True,
        null=True,
        verbose_name='프로필 이미지'
    )

    # 타임스탬프
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name='생성일시'
    )

    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name='수정일시'
    )

    class Meta:
        """
        모델 메타데이터
        - db_table: 데이터베이스 테이블 이름
        - verbose_name: Admin 페이지에 표시될 이름
        - ordering: 기본 정렬 순서
        """
        db_table = 'user_profiles'
        verbose_name = '사용자 프로필'
        verbose_name_plural = '사용자 프로필 목록'
        ordering = ['-created_at']

    def __str__(self):
        """
        객체의 문자열 표현
        Admin 페이지나 쉘에서 객체를 출력할 때 보이는 내용
        """
        return f"{self.user.username}의 프로필"


# ============================================================================
# Signal: User 생성 시 자동으로 Profile 생성
# ============================================================================
# Django Signal을 사용하여 User 객체가 생성될 때 자동으로 프로필을 생성합니다.

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    User 생성 시 자동으로 UserProfile 생성

    Args:
        sender: 신호를 보낸 모델 클래스 (User)
        instance: 생성/수정된 User 인스턴스
        created: 새로 생성되었는지 여부 (True/False)
    """
    if created:
        # 새로운 User가 생성되면 프로필도 자동 생성
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    User 저장 시 프로필도 함께 저장
    """
    # User에 profile이 없으면 생성 (안전장치)
    if not hasattr(instance, 'profile'):
        UserProfile.objects.create(user=instance)
    else:
        instance.profile.save()


# ============================================================================
# 참고:
#
# 1. User 모델 사용 예시:
#    from django.contrib.auth.models import User
#    user = User.objects.create_user('john', 'john@example.com', 'password123')
#    print(user.profile.phone_number)  # 프로필 접근
#
# 2. 커스텀 User 모델:
#    더 복잡한 요구사항이 있다면 AbstractUser를 상속받아 커스텀 User 모델을 만듭니다.
#
# 3. 인증:
#    Django REST Framework의 TokenAuthentication 또는
#    SimpleJWT를 사용한 JWT 인증을 추가할 수 있습니다.
# ============================================================================
