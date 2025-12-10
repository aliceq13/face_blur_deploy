# FaceBlur - AI 기반 영상 얼굴 블러 처리 시스템

**영상 속 특정 인물의 얼굴을 선택적으로 블러 처리하는 웹 서비스**

[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://www.djangoproject.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20EC2-orange.svg)](https://aws.amazon.com/)

---

## 📖 프로젝트 소개

FaceBlur는 **AI 얼굴 인식 기술**을 활용하여 영상 속 얼굴을 자동으로 감지하고, 사용자가 선택한 얼굴만 블러 처리하는 웹 서비스입니다.
모든 AI 처리는 Celery 워커를 통해 비동기적으로 수행되며, 대용량 영상도 안정적으로 처리할 수 있습니다.

### 🎯 주요 기능

1. **영상 업로드**: 대용량 영상 파일 업로드 및 비동기 처리
2. **자동 얼굴 분석**: YOLOv11 + BoTSORT + AdaFace 기반 정밀 얼굴 분석
3. **인물별 클러스터링**: BoTSORT가 부여한 Track id를 Instance ID로 통합 --> 동일 인물을 자동으로 클러스터링(Re-ID) 
4. **선택적 블러 처리**: 사용자가 선택한 특정 인물만 자동으로 추적하여 블러링
5. **실시간 진행률**: WebSocket/Polling을 통한 처리 상태 실시간 확인

---

## 🏗️ 시스템 아키텍처

이 프로젝트는 **Django**를 메인 웹 프레임워크로 사용하며, **Celery**와 **Redis**를 통해 AI 작업을 비동기 처리하는 구조로 설계되었습니다.

"추후 작성"

### 기술 스택

**Backend**:
- **Django 4.2**: 웹 서버 및 API
- **Django REST Framework**: REST API
- **Celery**: 비동기 작업 큐 (AI 모델 실행 담당)
- **Redis**: 메시지 브로커 및 캐시

**AI/ML (Local Execution)**:
- **YOLOv11**: 고성능 얼굴 탐지
- **AdaFace**: 얼굴 특징 추출 (임베딩)
- **BoTSORT**: 다중 객체 추적 (MOT)
- **OpenCV**: 영상 처리 및 블러링 적용

**Infrastructure**:
- **Docker & Docker Compose**: 컨테이너 기반 배포
- **Nginx**: 리버스 프록시 (배포 시)
- **Ngrok**: local host 배포방식

---

## 🚀 설치 및 실행 (Quick Start)

이 프로젝트는 Docker Compose를 통해 한 번에 실행할 수 있습니다.

### 1️⃣ 요구 사항

- Docker & Docker Compose
- (권장) NVIDIA GPU + NVIDIA Container Toolkit (AI 모델 가속용)
- Git LFS (대용량 모델 파일 다운로드용)

### 2️⃣ 설치

```bash
# Git LFS 설치 (필요시)
git lfs install

# 저장소 클론
git clone https://github.com/aliceq13/face_blur_deploy.git
cd face_blur_project

# 대용량 모델 파일 pull
git lfs pull
```

### 3️⃣ 실행

```bash
# .env 설정 (기본값 사용 가능)
cp .env.example .env

# Docker 컨테이너 빌드 및 실행
docker-compose up -d --build

# 로그 확인 (선택)
docker-compose logs -f django celery
```

### 4️⃣ 초기 설정

```bash
# 데이터베이스 마이그레이션
docker-compose exec django python manage.py migrate

# 관리자 계정 생성
docker-compose exec django python manage.py createsuperuser
```

- 웹 접속: `http://localhost:8000`

---

## 📁 프로젝트 구조

```
face_blur_project/
├── apps/
│   ├── videos/             # 핵심 로직 (모델, 뷰, 시리얼라이저)
│   │   ├── tasks.py        # [중요] Celery AI 작업 정의 (분석, 블러링)
│   │   ├── video_blurring.py # 영상 처리/블러링 구현체
│   │   └── ...
│   ├── processing/         # 처리 상태 관리
│   └── accounts/           # 사용자 인증
├── models/                 # AI 모델 가중치 파일 저장소
├── face_blur_web/          # Django 설정
├── media/                  # (Git 제외) 업로드 및 결과 파일
├── docker-compose.yml      # 서비스 구성
└── Dockerfile.django       # Django/Celery 이미지 정의
```

---

## 🔐 데이터 및 보안

- **비밀번호/키 관리**: `.env` 파일을 통해 관리하며 Git에 포함되지 않습니다.
- **모델 파일**: Git LFS를 통해 대용량 모델 가중치(`*.pt`)를 관리합니다.
- **데이터베이스**: 개발 환경은 SQLite, 운영 환경은 PostgreSQL을 지원합니다.

---

## 👥 Contributor

- **aliceq13** (Project Lead)

## 🌐 demo website
- **web url**: https://unexceptional-lynwood-cognominally.ngrok-free.dev
