/**
 * API 클라이언트 유틸리티
 * Django REST Framework API와 통신하는 함수들
 */

// CSRF 토큰 가져오기 (Django에서 필요)
function getCSRFToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value || csrftoken;
}

/**
 * 비디오 업로드 API
 * @param {File} file - 업로드할 비디오 파일
 * @param {string} title - 비디오 제목
 * @param {Function} onProgress - 진행률 콜백 (percent) => void
 * @returns {Promise<Object>} 생성된 비디오 객체
 */
async function uploadVideo(file, title, onProgress) {
    const formData = new FormData();
    formData.append('video_file', file);
    formData.append('title', title);

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        // 진행률 이벤트
        if (onProgress) {
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    onProgress(percent);
                }
            });
        }

        // 완료 이벤트
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    resolve(response);
                } catch (e) {
                    reject(new Error('응답 파싱 실패'));
                }
            } else {
                try {
                    const error = JSON.parse(xhr.responseText);
                    reject(new Error(error.error || error.detail || '업로드 실패'));
                } catch (e) {
                    reject(new Error(`업로드 실패 (${xhr.status})`));
                }
            }
        });

        // 에러 이벤트
        xhr.addEventListener('error', () => {
            reject(new Error('네트워크 오류'));
        });

        // 취소 이벤트
        xhr.addEventListener('abort', () => {
            reject(new Error('업로드 취소됨'));
        });

        // 요청 시작
        xhr.open('POST', '/api/videos/');
        xhr.setRequestHeader('X-CSRFToken', getCSRFToken());
        xhr.send(formData);

        // XMLHttpRequest 객체를 반환하여 외부에서 abort() 가능하도록
        xhr.uploadPromise = { resolve, reject, xhr };
    });
}

/**
 * 비디오 목록 조회 API
 * @returns {Promise<Array>} 비디오 목록
 */
async function getVideoList() {
    const response = await fetch('/api/videos/', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin'
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '비디오 목록 조회 실패');
    }

    return await response.json();
}

/**
 * 비디오 상세 조회 API
 * @param {string} videoId - 비디오 UUID
 * @returns {Promise<Object>} 비디오 상세 정보
 */
async function getVideoDetail(videoId) {
    const response = await fetch(`/api/videos/${videoId}/`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin'
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '비디오 조회 실패');
    }

    return await response.json();
}

/**
 * 얼굴 목록 조회 API
 * @param {string} videoId - 비디오 UUID
 * @returns {Promise<Array>} 얼굴 목록
 */
async function getFaceList(videoId) {
    const response = await fetch(`/api/faces/?video_id=${videoId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin'
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '얼굴 목록 조회 실패');
    }

    return await response.json();
}

/**
 * 얼굴 블러 선택 업데이트 API
 * @param {string} faceId - 얼굴 UUID
 * @param {boolean} isBlurred - 블러 처리 여부
 * @returns {Promise<Object>} 업데이트된 얼굴 객체
 */
async function updateFaceBlur(faceId, isBlurred) {
    const response = await fetch(`/api/faces/${faceId}/`, {
        method: 'PATCH',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({ is_blurred: isBlurred })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || error.is_blurred?.[0] || '얼굴 선택 업데이트 실패');
    }

    return await response.json();
}

/**
 * 비디오 처리 시작 API
 * @param {string} videoId - 비디오 UUID
 * @returns {Promise<Object>} 처리 작업 정보
 */
async function startVideoProcessing(videoId) {
    const response = await fetch(`/api/videos/${videoId}/start_processing/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin',
        body: JSON.stringify({})
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || error.error || '처리 시작 실패');
    }

    return await response.json();
}

/**
 * 처리 작업 상태 조회 API
 * @param {string} videoId - 비디오 UUID
 * @returns {Promise<Array>} 처리 작업 목록
 */
async function getProcessingJobs(videoId) {
    const response = await fetch(`/api/jobs/?video_id=${videoId}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin'
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || '작업 상태 조회 실패');
    }

    return await response.json();
}

/**
 * 비디오 삭제 API
 * @param {string} videoId - 비디오 UUID
 * @returns {Promise<void>}
 */
async function deleteVideo(videoId) {
    const response = await fetch(`/api/videos/${videoId}/`, {
        method: 'DELETE',
        headers: {
            'X-CSRFToken': getCSRFToken(),
        },
        credentials: 'same-origin'
    });

    if (!response.ok && response.status !== 204) {
        const error = await response.json();
        throw new Error(error.detail || '비디오 삭제 실패');
    }
}

/**
 * 파일 크기 포맷팅 유틸리티
 * @param {number} bytes - 바이트 크기
 * @returns {string} 포맷된 문자열 (예: "150MB")
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * 시간 포맷팅 유틸리티
 * @param {number} seconds - 초 단위 시간
 * @returns {string} 포맷된 문자열 (예: "2:35")
 */
function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}
