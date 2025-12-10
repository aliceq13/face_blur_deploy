# -*- coding: utf-8 -*-
"""
Face 데이터 확인 스크립트
모자이크가 적용되지 않는 문제를 진단하기 위해 데이터베이스의 Face 모델을 확인합니다.
"""

import os
import sys
import django

# Django 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_blur_web.settings')
django.setup()

from apps.videos.models import Video, Face

def check_face_data():
    """Face 데이터 확인"""
    print("=" * 80)
    print("Face 데이터 확인 시작")
    print("=" * 80)
    
    # 모든 비디오 확인
    videos = Video.objects.all()
    print(f"\n총 비디오 수: {videos.count()}")
    
    for video in videos:
        print(f"\n{'=' * 80}")
        print(f"Video ID: {video.id}")
        print(f"파일명: {video.original_file_url}")
        print(f"상태: {video.status}")
        print(f"{'-' * 80}")
        
        # 이 비디오의 얼굴들 확인
        faces = Face.objects.filter(video=video)
        print(f"감지된 얼굴 수: {faces.count()}")
        
        blur_count = 0
        unblur_count = 0
        
        for face in faces:
            is_blurred_str = "블러 O (is_blurred=True)" if face.is_blurred else "블러 X (is_blurred=False)"
            print(f"\n  Face {face.face_index}:")
            print(f"    - ID: {face.id}")
            print(f"    - 블러 설정: {is_blurred_str}")
            print(f"    - 등장 횟수: {face.appearance_count}")
            print(f"    - 프레임 범위: {face.first_frame} ~ {face.last_frame}")
            print(f"    - 썸네일: {face.thumbnail_url}")
            print(f"    - 임베딩 존재: {'O' if face.embedding else 'X'}")
            if face.embedding:
                print(f"    - 임베딩 차원: {len(face.embedding)}")
            
            if face.is_blurred:
                blur_count += 1
            else:
                unblur_count += 1
        
        print(f"\n{'-' * 80}")
        print(f"요약:")
        print(f"  - 블러 적용 대상 (is_blurred=True): {blur_count}개")
        print(f"  - 블러 해제 대상 (is_blurred=False): {unblur_count}개")
        
        if unblur_count == 0:
            print(f"\n⚠️  경고: 블러 해제 대상이 없습니다!")
            print(f"   모든 얼굴이 is_blurred=True로 설정되어 있습니다.")
            print(f"   이 경우 target_embeddings가 빈 리스트가 되어")
            print(f"   모든 트랙이 is_blurred=True로 결정됩니다.")
            print(f"   하지만 렌더링 로직에서는 is_blurred=True인 얼굴만 블러 처리하므로")
            print(f"   모든 얼굴이 블러 처리되어야 합니다.")

if __name__ == '__main__':
    check_face_data()
