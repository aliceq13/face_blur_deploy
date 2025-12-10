# FaceBlur Project - 빠른 시작 가이드

**이 가이드는 다음 세션에서 빠르게 프로젝트를 재시작할 수 있도록 돕습니다.**

---

## 🎯 현재 상태 (2025-11-09)

**Phase 1 진행도**: 80% 완료

✅ **완료된 작업**:
- Docker 환경 구성
- Django 앱 3개 생성 (accounts, videos, processing)
- 데이터베이스 모델 구현 (Video, Face, ProcessingJob)
- Admin 페이지 구현
- settings.py 완전 설정

❌ **남은 작업**:
- DRF Serializers 작성
- API Views 구현
- URL 라우팅 설정
- 템플릿 통합
- 마이그레이션 실행

---

## ⚡ 다음 세션 시작 (1분 안에)

### 1️⃣ 프로젝트 열기

```bash
# 프로젝트 디렉토리로 이동
cd c:\Users\이승복\Documents\face_blur_project
```

### 2️⃣ Docker 서비스 시작

```bash
# Docker Compose 실행 (처음이거나 코드 변경 시)
docker-compose up -d --build

# 또는 빠르게 시작 (변경사항 없을 때)
docker-compose up -d
```

### 3️⃣ 로그 확인 (서비스가 정상 작동하는지 확인)

```bash
# Django 로그 실시간 확인
docker-compose logs -f django

# Ctrl+C로 종료
```

### 4️⃣ 진행 상황 확인

```bash
# PROGRESS.md 읽기
cat PROGRESS.md

# 또는 에디터에서 열기
code PROGRESS.md
```

---

## 📝 다음에 할 일 (우선순위)

### 1. DRF Serializers 작성 (30분)

**파일**: `apps/videos/serializers.py`

**내용**:
```python
from rest_framework import serializers
from .models import Video, Face, ProcessingJob

class FaceSerializer(serializers.ModelSerializer):
    # Face 직렬화

class VideoSerializer(serializers.ModelSerializer):
    # Video 직렬화 (Face 포함)

class ProcessingJobSerializer(serializers.ModelSerializer):
    # ProcessingJob 직렬화
```

**완료 조건**: Serializer 클래스 3개 작성

---

### 2. API Views 구현 (1시간)

**파일**: `apps/videos/views.py`

**내용**:
```python
from rest_framework import viewsets

class VideoViewSet(viewsets.ModelViewSet):
    # 영상 CRUD API
    # 파일 업로드 처리
    # 메타데이터 추출
```

**완료 조건**:
- VideoViewSet 구현
- 파일 업로드 동작

---

### 3. URL 라우팅 설정 (20분)

**파일 1**: `apps/videos/urls.py` (생성)
**파일 2**: `face_blur_web/urls.py` (수정)

**완료 조건**:
- `/api/videos/` 엔드포인트 접근 가능
- DRF Browsable API 작동

---

### 4. 템플릿 통합 (1시간)

**작업**:
- `templates/` 디렉토리 생성
- `stitch_` HTML 파일 이동
- Django 템플릿 문법 적용

**완료 조건**:
- 업로드 페이지 접속 가능

---

### 5. 마이그레이션 & 테스트 (30분)

```bash
# 마이그레이션 파일 생성
docker-compose exec django python manage.py makemigrations

# 마이그레이션 적용
docker-compose exec django python manage.py migrate

# 슈퍼유저 생성
docker-compose exec django python manage.py createsuperuser

# Admin 페이지 접속 테스트
# http://localhost:8000/admin
```

---

## 🐳 Docker 명령어 치트시트

### 기본 명령어

```bash
# 서비스 시작
docker-compose up -d

# 서비스 중지
docker-compose down

# 서비스 재시작
docker-compose restart

# 로그 확인 (실시간)
docker-compose logs -f [서비스명]

# 서비스 상태 확인
docker-compose ps
```

### Django 관련 명령어

```bash
# Django 쉘 접속
docker-compose exec django python manage.py shell

# 마이그레이션 생성
docker-compose exec django python manage.py makemigrations

# 마이그레이션 적용
docker-compose exec django python manage.py migrate

# 슈퍼유저 생성
docker-compose exec django python manage.py createsuperuser

# Static 파일 수집
docker-compose exec django python manage.py collectstatic --noinput
```

### 데이터베이스 명령어

```bash
# PostgreSQL 쉘 접속
docker-compose exec db psql -U postgres -d faceblur_db

# 테이블 목록 확인
docker-compose exec db psql -U postgres -d faceblur_db -c "\dt"

# 데이터베이스 초기화 (주의!)
docker-compose down -v
```

### 디버깅 명령어

```bash
# 컨테이너 내부 접속 (Bash)
docker-compose exec django bash

# Django 서버 재시작
docker-compose restart django

# 빌드 캐시 무시하고 재빌드
docker-compose build --no-cache
docker-compose up -d
```

---

## 🌐 접속 URL

| 서비스 | URL | 용도 |
|--------|-----|------|
| Django Admin | http://localhost:8000/admin | 관리자 페이지 |
| DRF API Root | http://localhost:8000/api/ | API 목록 |
| Videos API | http://localhost:8000/api/videos/ | 영상 API |
| PostgreSQL | localhost:5432 | DB 연결 |
| Redis | localhost:6379 | 캐시/Celery |

---

## 🔧 문제 해결

### 문제 1: Docker 컨테이너가 시작되지 않음

**해결**:
```bash
# 로그 확인
docker-compose logs

# 모든 컨테이너 중지
docker-compose down

# 볼륨 삭제 후 재시작
docker-compose down -v
docker-compose up -d --build
```

---

### 문제 2: 마이그레이션 에러

**해결**:
```bash
# 마이그레이션 초기화
docker-compose exec django python manage.py migrate --fake

# 또는 데이터베이스 완전 초기화
docker-compose down -v
docker-compose up -d
docker-compose exec django python manage.py migrate
```

---

### 문제 3: `ModuleNotFoundError: No module named 'decouple'`

**원인**: requirements.txt 설치 안됨

**해결**:
```bash
# 컨테이너 재빌드
docker-compose build --no-cache
docker-compose up -d
```

---

### 문제 4: 포트가 이미 사용 중

**해결**:
```bash
# 사용 중인 프로세스 종료 (Windows)
netstat -ano | findstr :8000
taskkill /PID [프로세스_ID] /F

# 또는 docker-compose.yml에서 포트 변경
# ports: "8080:8000"
```

---

## 📖 참고 파일

| 파일 | 설명 |
|------|------|
| `SKILL.md` | 전체 아키텍처 가이드 (1,890줄) |
| `PROGRESS.md` | 상세 진행 상황 (이 문서) |
| `QUICKSTART.md` | 빠른 시작 가이드 (이 파일) |
| `.env.example` | 환경 변수 템플릿 |
| `requirements.txt` | Python 패키지 목록 |
| `docker-compose.yml` | Docker 서비스 정의 |

---

## 🎓 학습 팁

### 1. Admin 페이지 활용

Admin 페이지에서 모델 구조를 시각적으로 확인할 수 있습니다.

```bash
# 슈퍼유저 생성
docker-compose exec django python manage.py createsuperuser

# 브라우저에서 접속
# http://localhost:8000/admin
```

### 2. Django Shell 활용

```bash
# Shell 접속
docker-compose exec django python manage.py shell

# 모델 테스트
>>> from apps.videos.models import Video
>>> Video.objects.all()
>>> from django.contrib.auth.models import User
>>> User.objects.create_user('test', 'test@example.com', 'password123')
```

### 3. DRF Browsable API 활용

브라우저에서 직접 API를 테스트할 수 있습니다.

```
http://localhost:8000/api/videos/
http://localhost:8000/api/accounts/users/
```

---

## ✅ Phase 1 완료 체크리스트

- [ ] Serializers 작성
- [ ] Views 작성
- [ ] URLs 설정
- [ ] 템플릿 통합
- [ ] 마이그레이션 실행
- [ ] Admin 페이지 접속 확인
- [ ] API 엔드포인트 테스트
- [ ] 파일 업로드 테스트

**완료 후**: Phase 2 (파일 업로드 기능 구현)로 진행

---

## 💬 질문이 있다면

1. **SKILL.md** 읽기 (전체 아키텍처)
2. **PROGRESS.md** 읽기 (상세 진행 상황)
3. 각 파일의 주석 읽기 (모든 코드에 설명 포함)

---

**작성일**: 2025-11-09
**다음 작업**: Serializers 작성부터 시작!
