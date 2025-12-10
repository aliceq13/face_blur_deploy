# ============================================================================
# Processing App - Views
# ============================================================================
# 영상 처리 관련 API 엔드포인트
# Phase 4(Celery)에서 본격적으로 구현 예정
# ============================================================================

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


# ============================================================================
# 영상 처리 시작 API (향후 구현)
# ============================================================================
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_video_processing(request, video_id):
    """
    영상 블러 처리 시작

    POST /api/processing/videos/{video_id}/start/

    Body:
        {
            "face_selections": [1, 2, 3]  // 블러 처리할 얼굴 인덱스
        }

    Returns:
        {
            "job_id": "uuid-string",
            "status": "started",
            "message": "영상 처리가 시작되었습니다."
        }

    Phase 4에서 Celery 작업을 실행하도록 구현 예정:
    - face_analysis_task.delay(video_id)
    - video_processing_task.delay(video_id, face_selections)
    """

    # 임시 응답
    return Response({
        'message': 'Phase 4에서 Celery 작업으로 구현 예정입니다.',
        'video_id': video_id,
        'status': 'not_implemented'
    }, status=status.HTTP_501_NOT_IMPLEMENTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_processing_status(request, job_id):
    """
    처리 작업 상태 조회

    GET /api/processing/jobs/{job_id}/status/

    Returns:
        {
            "job_id": "uuid-string",
            "status": "processing",
            "progress": 75,
            "message": "프레임 처리 중..."
        }

    Phase 4에서 ProcessingJob 모델을 조회하도록 구현 예정
    """

    return Response({
        'message': 'Phase 4에서 구현 예정입니다.',
        'job_id': job_id,
        'status': 'not_implemented'
    }, status=status.HTTP_501_NOT_IMPLEMENTED)


# ============================================================================
# FastAPI 통신 클라이언트 (향후 구현)
# ============================================================================
# class FastAPIClient:
#     """
#     FastAPI AI 서버와 통신하는 클라이언트
#
#     메서드:
#     - detect_faces(image_bytes): 얼굴 검출
#     - detect_faces_batch(images_bytes_list): 배치 얼굴 검출
#     - extract_embedding(face_image_bytes): 임베딩 추출
#     """
#     pass


# ============================================================================
# 참고:
# Phase 4에서 다음 기능들을 구현합니다:
#
# 1. Celery Tasks (tasks.py 생성)
#    - analyze_faces_task: 얼굴 분석
#    - process_video_task: 영상 블러 처리
#
# 2. FastAPI 클라이언트 (services.py 생성)
#    - httpx를 사용한 비동기 HTTP 통신
#
# 3. 유틸리티 함수 (utils.py 생성)
#    - 영상 메타데이터 추출 (MoviePy)
#    - 프레임 추출 (OpenCV)
#    - 블러 처리 (OpenCV)
# ============================================================================
