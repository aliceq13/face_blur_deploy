# ============================================================================
# Videos App - Admin
# ============================================================================
# Django Admin 페이지에서 영상 및 얼굴 데이터 관리
# ============================================================================

from django.contrib import admin
from django.utils.html import format_html
from .models import Video, Face, ProcessingJob


# ============================================================================
# Face Inline Admin
# ============================================================================
# Video Admin 화면에 Face를 Inline으로 표시

class FaceInline(admin.TabularInline):
    """
    Video Admin에 얼굴 Instance 목록을 Inline으로 표시

    TabularInline: 테이블 형태로 표시
    """
    model = Face
    extra = 0  # 빈 추가 폼 표시하지 않음
    readonly_fields = ('id', 'instance_id', 'thumbnail_preview', 'quality_score',
                      'total_frames', 'track_count', 'created_at')
    fields = ('instance_id', 'thumbnail_preview', 'is_blurred',
              'quality_score', 'total_frames', 'track_count')

    def thumbnail_preview(self, obj):
        """썸네일 이미지 미리보기"""
        if obj.thumbnail_url:
            return format_html(
                '<img src="{}" width="50" height="50" style="border-radius: 4px;" />',
                obj.thumbnail_url
            )
        return "-"
    thumbnail_preview.short_description = '썸네일'

    def track_count(self, obj):
        """Track ID 개수"""
        return len(obj.track_ids) if obj.track_ids else 0
    track_count.short_description = 'Track 수'


# ============================================================================
# ProcessingJob Inline Admin
# ============================================================================
class ProcessingJobInline(admin.TabularInline):
    """Video Admin에 처리 작업 목록을 Inline으로 표시"""
    model = ProcessingJob
    extra = 0
    readonly_fields = ('id', 'job_type', 'status', 'progress',
                      'started_at', 'completed_at')
    fields = ('job_type', 'status', 'progress', 'started_at', 'completed_at')
    can_delete = False  # 작업은 삭제하지 못하게 함


# ============================================================================
# Video Admin
# ============================================================================
@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    """
    영상 모델 Admin 설정
    """

    # 목록 화면 설정
    list_display = ('title', 'user', 'status_badge', 'progress_bar',
                    'resolution', 'duration_display', 'file_size_mb',
                    'created_at')

    list_filter = ('status', 'created_at', 'user')

    search_fields = ('title', 'original_filename', 'user__username')

    readonly_fields = ('id', 'created_at', 'updated_at', 'completed_at',
                      'video_preview', 'file_size_mb')

    # Inline 추가
    inlines = [FaceInline, ProcessingJobInline]

    # 상세 화면 필드 그룹핑
    fieldsets = (
        ('기본 정보', {
            'fields': ('id', 'user', 'title', 'original_filename')
        }),
        ('파일 정보', {
            'fields': ('original_file_url', 'processed_file_url', 'video_preview')
        }),
        ('영상 메타데이터', {
            'fields': ('duration', 'width', 'height', 'fps', 'file_size', 'file_size_mb')
        }),
        ('처리 상태', {
            'fields': ('status', 'progress', 'error_message')
        }),
        ('타임스탬프', {
            'fields': ('created_at', 'updated_at', 'completed_at', 'expires_at'),
            'classes': ('collapse',)
        }),
    )

    # 정렬
    ordering = ('-created_at',)

    # 페이지당 표시 개수
    list_per_page = 25

    # ========================================================================
    # 커스텀 메서드
    # ========================================================================
    def status_badge(self, obj):
        """상태를 색상 뱃지로 표시"""
        colors = {
            'uploaded': '#808080',      # 회색
            'analyzing': '#3498db',     # 파랑
            'ready': '#2ecc71',         # 초록
            'processing': '#f39c12',    # 주황
            'completed': '#27ae60',     # 진한 초록
            'failed': '#e74c3c',        # 빨강
        }
        color = colors.get(obj.status, '#000000')
        return format_html(
            '<span style="background-color:{}; color:white; padding:3px 8px; border-radius:3px; font-weight:bold;">{}</span>',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = '상태'

    def progress_bar(self, obj):
        """진행률을 프로그레스 바로 표시"""
        if obj.progress == 0:
            return "-"

        # 진행률에 따라 색상 변경
        if obj.progress < 30:
            color = '#e74c3c'  # 빨강
        elif obj.progress < 70:
            color = '#f39c12'  # 주황
        else:
            color = '#2ecc71'  # 초록

        return format_html(
            '''
            <div style="width:100px; background-color:#ecf0f1; border-radius:3px; overflow:hidden;">
                <div style="width:{}%; background-color:{}; color:white; text-align:center; padding:2px 0; font-size:11px; font-weight:bold;">
                    {}%
                </div>
            </div>
            ''',
            obj.progress,
            color,
            obj.progress
        )
    progress_bar.short_description = '진행률'

    def resolution(self, obj):
        """해상도 표시"""
        return obj.get_resolution_str()
    resolution.short_description = '해상도'

    def duration_display(self, obj):
        """영상 길이를 분:초 형식으로 표시"""
        minutes = int(obj.duration // 60)
        seconds = int(obj.duration % 60)
        return f"{minutes:02d}:{seconds:02d}"
    duration_display.short_description = '길이'

    def file_size_mb(self, obj):
        """파일 크기를 MB로 표시"""
        return f"{obj.get_file_size_mb()} MB"
    file_size_mb.short_description = '파일 크기'

    def video_preview(self, obj):
        """영상 미리보기 (처리된 영상이 있으면 표시)"""
        if obj.processed_file_url:
            return format_html(
                '''
                <video width="320" height="240" controls>
                    <source src="{}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                ''',
                obj.processed_file_url
            )
        elif obj.original_file_url:
            return format_html(
                '''
                <video width="320" height="240" controls>
                    <source src="{}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                ''',
                obj.original_file_url
            )
        return "-"
    video_preview.short_description = '미리보기'


# ============================================================================
# Face Admin
# ============================================================================
@admin.register(Face)
class FaceAdmin(admin.ModelAdmin):
    """얼굴 Instance 모델 Admin 설정"""

    list_display = ('instance_id', 'video', 'thumbnail_preview',
                    'quality_score', 'total_frames', 'track_count', 'blur_status', 'created_at')

    list_filter = ('is_blurred', 'created_at')

    search_fields = ('video__title', 'video__user__username', 'instance_id')

    readonly_fields = ('id', 'created_at', 'thumbnail_large',
                      'track_ids_display', 'quality_score', 'frame_index', 'bbox')

    fieldsets = (
        ('기본 정보', {
            'fields': ('id', 'video', 'instance_id')
        }),
        ('이미지', {
            'fields': ('thumbnail_url', 'thumbnail_large')
        }),
        ('임베딩', {
            'fields': ('embedding',),
            'classes': ('collapse',)  # 접혀있는 상태
        }),
        ('Re-ID 정보', {
            'fields': ('track_ids', 'track_ids_display', 'quality_score', 'frame_index', 'bbox')
        }),
        ('통계', {
            'fields': ('total_frames',)
        }),
        ('처리 옵션', {
            'fields': ('is_blurred',)
        }),
        ('타임스탬프', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

    ordering = ('instance_id',)

    def track_count(self, obj):
        """Track ID 개수"""
        return len(obj.track_ids) if obj.track_ids else 0
    track_count.short_description = 'Track 수'

    def track_ids_display(self, obj):
        """Track IDs를 보기 좋게 표시"""
        if obj.track_ids:
            return format_html(
                '<span style="font-family:monospace;">{}</span>',
                ', '.join(map(str, sorted(obj.track_ids)))
            )
        return "-"
    track_ids_display.short_description = 'Track IDs (정렬됨)'

    def thumbnail_preview(self, obj):
        """작은 썸네일 미리보기 (목록용)"""
        if obj.thumbnail_url:
            return format_html(
                '<img src="{}" width="50" height="50" style="border-radius: 4px;" />',
                obj.thumbnail_url
            )
        return "-"
    thumbnail_preview.short_description = '썸네일'

    def thumbnail_large(self, obj):
        """큰 썸네일 미리보기 (상세 페이지용)"""
        if obj.thumbnail_url:
            return format_html(
                '<img src="{}" width="200" height="200" style="border-radius: 8px; border: 2px solid #ddd;" />',
                obj.thumbnail_url
            )
        return "-"
    thumbnail_large.short_description = '썸네일 (큰 이미지)'

    def blur_status(self, obj):
        """블러 처리 여부를 아이콘으로 표시"""
        if obj.is_blurred:
            return format_html(
                '<span style="color:#e74c3c; font-weight:bold;">● 블러 처리</span>'
            )
        else:
            return format_html(
                '<span style="color:#2ecc71; font-weight:bold;">○ 원본 유지</span>'
            )
    blur_status.short_description = '블러 상태'


# ============================================================================
# ProcessingJob Admin
# ============================================================================
@admin.register(ProcessingJob)
class ProcessingJobAdmin(admin.ModelAdmin):
    """처리 작업 Admin 설정"""

    list_display = ('video', 'job_type', 'status_badge', 'progress_bar',
                    'started_at', 'completed_at')

    list_filter = ('job_type', 'status', 'started_at')

    search_fields = ('video__title', 'celery_task_id')

    readonly_fields = ('id', 'celery_task_id', 'started_at', 'completed_at',
                      'result_data_display', 'error_message')

    fieldsets = (
        ('기본 정보', {
            'fields': ('id', 'video', 'job_type')
        }),
        ('Celery 정보', {
            'fields': ('celery_task_id', 'status', 'progress')
        }),
        ('결과', {
            'fields': ('result_data', 'result_data_display', 'error_message')
        }),
        ('타임스탬프', {
            'fields': ('started_at', 'completed_at')
        }),
    )

    ordering = ('-started_at',)

    def status_badge(self, obj):
        """상태 뱃지"""
        colors = {
            'pending': '#95a5a6',
            'started': '#3498db',
            'success': '#27ae60',
            'failure': '#e74c3c',
            'retry': '#f39c12',
            'revoked': '#7f8c8d',
        }
        color = colors.get(obj.status, '#000000')
        return format_html(
            '<span style="background-color:{}; color:white; padding:3px 8px; border-radius:3px; font-weight:bold;">{}</span>',
            color,
            obj.get_status_display()
        )
    status_badge.short_description = '상태'

    def progress_bar(self, obj):
        """진행률 바"""
        if obj.progress == 0:
            return "-"
        return format_html(
            '''
            <div style="width:100px; background-color:#ecf0f1; border-radius:3px; overflow:hidden;">
                <div style="width:{}%; background-color:#3498db; color:white; text-align:center; padding:2px 0; font-size:11px;">
                    {}%
                </div>
            </div>
            ''',
            obj.progress,
            obj.progress
        )
    progress_bar.short_description = '진행률'

    def result_data_display(self, obj):
        """결과 데이터를 보기 좋게 표시"""
        if obj.result_data:
            import json
            return format_html(
                '<pre style="background:#f8f9fa; padding:10px; border-radius:4px;">{}</pre>',
                json.dumps(obj.result_data, indent=2, ensure_ascii=False)
            )
        return "-"
    result_data_display.short_description = '결과 데이터 (포맷팅)'


# ============================================================================
# Admin 사이트 제목 커스터마이징
# ============================================================================
admin.site.site_header = 'FaceBlur 관리자 페이지'
admin.site.site_title = 'FaceBlur Admin'
admin.site.index_title = '영상 및 얼굴 관리'
