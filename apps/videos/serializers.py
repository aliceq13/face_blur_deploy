# -*- coding: utf-8 -*-
"""
비디오 앱 Serializers
Django REST Framework의 Serializer는 모델 인스턴스를 JSON 형식으로 변환하거나,
JSON 데이터를 모델 인스턴스로 변환하는 역할을 합니다.
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Video, Face, ProcessingJob


class UserSerializer(serializers.ModelSerializer):
    """
    사용자 정보 Serializer
    - 비디오 소유자 정보를 표시할 때 사용
    - 민감한 정보(비밀번호 등)는 제외
    """
    class Meta:
        model = User
        fields = ['id', 'username', 'email']  # 비밀번호는 제외
        read_only_fields = ['id']


class FaceSerializer(serializers.ModelSerializer):
    """
    얼굴 Instance 정보 Serializer
    - Two-Stage Re-ID로 감지된 얼굴 Instance 정보를 직렬화
    - Instance ID, 썸네일 URL, Track IDs, 품질 점수, 등장 프레임 수 등
    """
    track_count = serializers.SerializerMethodField()  # Track ID 개수

    def get_track_count(self, obj):
        """Track IDs 배열의 길이를 반환"""
        return len(obj.track_ids) if obj.track_ids else 0

    class Meta:
        model = Face
        fields = [
            'id',
            'instance_id',         # Re-ID 시스템이 할당한 고유 Instance ID
            'thumbnail_url',       # 최고 품질 썸네일 이미지 URL
            'track_ids',          # 이 instance에 속한 모든 BoTSORT track ID 목록
            'track_count',        # Track ID 개수 (편의성)
            'quality_score',      # Laplacian variance 선명도 점수
            'frame_index',        # 썸네일이 추출된 프레임 번호
            'bbox',               # 얼굴 bounding box [x1, y1, x2, y2]
            'total_frames',       # 영상 내 등장한 총 프레임 수
            'is_blurred',         # 블러 처리 선택 여부 (사용자가 선택)
            'created_at',
        ]
        read_only_fields = [
            'id', 'instance_id', 'thumbnail_url', 'track_ids', 'track_count',
            'quality_score', 'frame_index', 'bbox', 'total_frames', 'created_at'
        ]


class ProcessingJobSerializer(serializers.ModelSerializer):
    """
    처리 작업 Serializer
    - Celery 비동기 작업의 상태를 추적
    - 얼굴 분석 작업, 영상 처리 작업 등
    """
    job_type_display = serializers.CharField(source='get_job_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = ProcessingJob
        fields = [
            'id',
            'job_type',
            'job_type_display',     # 한글 표시명 (예: "얼굴 분석")
            'status',
            'status_display',       # 한글 표시명 (예: "처리 중")
            'progress',             # 진행률 (0-100)
            'celery_task_id',       # Celery 태스크 ID (디버깅용)
            'started_at',
            'completed_at',
            'error_message',
            'result_data',          # JSON 형식의 결과 데이터
        ]
        read_only_fields = ['id', 'celery_task_id', 'started_at', 'completed_at']


class VideoListSerializer(serializers.ModelSerializer):
    """
    비디오 목록용 Serializer (간단한 정보만)
    - 목록 조회 시 성능을 위해 필수 정보만 포함
    - 상세 정보는 VideoDetailSerializer 사용
    """
    user = UserSerializer(read_only=True, allow_null=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    face_count = serializers.SerializerMethodField()  # 감지된 얼굴 수

    def get_face_count(self, obj):
        """감지된 얼굴의 총 개수를 반환"""
        return obj.faces.count()

    class Meta:
        model = Video
        fields = [
            'id',
            'title',
            'status',
            'status_display',
            'progress',
            'duration',
            'face_count',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'status', 'progress', 'created_at', 'updated_at']


class VideoDetailSerializer(serializers.ModelSerializer):
    """
    비디오 상세 정보 Serializer
    - 비디오의 모든 정보 포함 (메타데이터, 얼굴 목록, 처리 작업 등)
    - 개별 비디오 조회 시 사용
    """
    user = UserSerializer(read_only=True, allow_null=True)
    faces = FaceSerializer(many=True, read_only=True)  # 감지된 얼굴 목록
    processing_jobs = ProcessingJobSerializer(
        source='processingjob_set',  # Video 모델의 역참조
        many=True,
        read_only=True
    )
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = Video
        fields = [
            'id',
            'user',
            'title',
            'original_file_url',    # 원본 영상 URL
            'processed_file_url',   # 처리된 영상 URL

            # 메타데이터
            'duration',             # 영상 길이 (초)
            'width',                # 가로 해상도
            'height',               # 세로 해상도
            'fps',                  # 프레임률
            'file_size',            # 파일 크기 (bytes)

            # 상태 정보
            'status',
            'status_display',
            'progress',             # 처리 진행률 (0-100)

            # 관계 데이터
            'faces',                # 감지된 얼굴 목록
            'processing_jobs',      # 처리 작업 목록

            # 타임스탬프
            'created_at',
            'updated_at',
            'expires_at',           # 자동 삭제 예정일 (7일 후)
        ]
        read_only_fields = [
            'id', 'user', 'original_file_url', 'processed_file_url',
            'duration', 'width', 'height', 'fps',
            'file_size', 'status', 'progress', 'created_at', 'updated_at', 'expires_at'
        ]


class VideoUploadSerializer(serializers.ModelSerializer):
    """
    비디오 업로드용 Serializer
    - 사용자가 새 비디오를 업로드할 때 사용
    - 파일 업로드와 함께 제목만 받음
    """
    # 실제 파일 업로드는 View에서 처리하고, 여기서는 메타데이터만 검증

    class Meta:
        model = Video
        fields = ['title']

    def validate_title(self, value):
        """제목 유효성 검사"""
        if len(value.strip()) < 2:
            raise serializers.ValidationError("제목은 최소 2자 이상이어야 합니다.")
        return value.strip()


class FaceUpdateSerializer(serializers.ModelSerializer):
    """
    얼굴 블러 처리 선택 업데이트용 Serializer
    - 사용자가 특정 얼굴에 대해 블러 처리 여부를 선택할 때 사용
    """
    class Meta:
        model = Face
        fields = ['is_blurred']

    def validate_is_blurred(self, value):
        """
        블러 처리 선택 유효성 검사
        - 비디오가 'ready' 상태일 때만 변경 가능
        """
        video = self.instance.video
        if video.status not in ['uploaded', 'ready']:
            raise serializers.ValidationError(
                f"비디오가 '{video.get_status_display()}' 상태일 때는 얼굴 선택을 변경할 수 없습니다."
            )
        return value


class VideoProcessSerializer(serializers.Serializer):
    """
    비디오 처리 시작 요청용 Serializer
    - 사용자가 얼굴 선택을 완료하고 영상 처리를 시작할 때 사용
    - 모델과 연결되지 않은 일반 Serializer
    """
    # 추가 옵션이 필요하면 여기에 필드 추가
    # 예: blur_intensity = serializers.IntegerField(min_value=1, max_value=10, default=5)

    def validate(self, data):
        """
        처리 시작 전 유효성 검사
        - 비디오가 'ready' 상태인지 확인
        - 최소 1개 이상의 얼굴이 블러 처리 대상으로 선택되었는지 확인
        """
        # View에서 video 인스턴스를 context로 전달받음
        video = self.context.get('video')

        if not video:
            raise serializers.ValidationError("비디오 정보를 찾을 수 없습니다.")

        if video.status != 'ready':
            raise serializers.ValidationError(
                f"비디오가 '{video.get_status_display()}' 상태일 때만 처리를 시작할 수 있습니다."
            )

        # 블러 처리 대상 얼굴이 있는지 확인
        blurred_faces_count = video.faces.filter(is_blurred=True).count()
        if blurred_faces_count == 0:
            raise serializers.ValidationError(
                "블러 처리할 얼굴을 최소 1개 이상 선택해주세요."
            )

        return data
