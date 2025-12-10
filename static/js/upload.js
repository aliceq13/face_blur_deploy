/**
 * 비디오 업로드 페이지 JavaScript
 * 파일 업로드, 드래그앤드롭, 진행률 표시 기능
 */

// 업로드 중인 파일들의 상태 관리
const uploadingFiles = new Map(); // key: fileId, value: { file, xhr, status, videoId }

// DOM 요소
let dropZone, fileInput, selectFileBtn, progressSection;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    dropZone = document.getElementById('drop-zone');
    fileInput = document.getElementById('video-file-input');
    selectFileBtn = document.getElementById('select-file-btn');
    progressSection = document.getElementById('upload-progress-section');

    initializeEventListeners();
});

/**
 * 이벤트 리스너 초기화
 */
function initializeEventListeners() {
    // 파일 선택 버튼 클릭
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // 파일 입력 변경
    fileInput.addEventListener('change', handleFileSelect);

    // 드래그앤드롭 영역 클릭
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // 드래그앤드롭 이벤트
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    // 페이지 전체에서 드래그앤드롭 기본 동작 방지
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

/**
 * 파일 선택 핸들러
 */
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFiles(Array.from(files));
    }
    // input 초기화 (같은 파일 다시 선택 가능하도록)
    event.target.value = '';
}

/**
 * 드래그 오버 핸들러
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('border-primary', 'bg-primary/10');
}

/**
 * 드래그 리브 핸들러
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('border-primary', 'bg-primary/10');
}

/**
 * 드롭 핸들러
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('border-primary', 'bg-primary/10');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(Array.from(files));
    }
}

/**
 * 파일 처리 메인 함수
 */
function handleFiles(files) {
    files.forEach(file => {
        // 파일 유효성 검사
        if (!validateFile(file)) {
            return;
        }

        // 고유 ID 생성
        const fileId = Date.now() + '-' + Math.random().toString(36).substr(2, 9);

        // 업로드 시작
        startUpload(fileId, file);
    });
}

/**
 * 파일 유효성 검사
 */
function validateFile(file) {
    // 파일 형식 검사
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
    const allowedExtensions = ['.mp4', '.mov', '.avi', '.mkv'];

    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    const isValidType = allowedTypes.includes(file.type) || allowedExtensions.includes(fileExtension);

    if (!isValidType) {
        showError(file.name, '지원하지 않는 파일 형식입니다. (MP4, MOV, AVI, MKV만 지원)');
        return false;
    }

    // 파일 크기 검사 (500MB)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(file.name, `파일 크기가 너무 큽니다. (최대 ${formatFileSize(maxSize)})`);
        return false;
    }

    return true;
}

/**
 * 업로드 시작
 */
async function startUpload(fileId, file) {
    // 제목 생성 (파일명에서 확장자 제거)
    const title = file.name.replace(/\.[^/.]+$/, '');

    // UI에 업로드 항목 추가
    addUploadItem(fileId, file);

    // 업로드 섹션 표시
    progressSection.style.display = 'flex';

    try {
        // API 호출
        const video = await uploadVideo(file, title, (percent) => {
            updateProgress(fileId, percent);
        });

        // 업로드 완료
        uploadingFiles.get(fileId).status = 'completed';
        uploadingFiles.get(fileId).videoId = video.id;
        showCompleted(fileId, video);

    } catch (error) {
        // 업로드 실패
        uploadingFiles.get(fileId).status = 'error';
        showUploadError(fileId, error.message);
    }
}

/**
 * 업로드 항목 UI 추가
 */
function addUploadItem(fileId, file) {
    const itemDiv = document.createElement('div');
    itemDiv.id = `upload-${fileId}`;
    itemDiv.className = 'flex flex-col gap-3';
    itemDiv.innerHTML = `
        <div class="flex flex-wrap gap-2 justify-between">
            <p class="text-base font-medium">${file.name} (${formatFileSize(file.size)})</p>
            <p class="text-sm font-normal text-secondary-light dark:text-secondary-dark" data-progress>0%</p>
        </div>
        <div class="w-full rounded-full bg-slate-200 dark:bg-border-dark">
            <div class="h-2 rounded-full bg-primary transition-all duration-300" style="width: 0%;" data-progress-bar></div>
        </div>
        <div class="flex items-center justify-between">
            <p class="text-sm font-normal text-secondary-light dark:text-secondary-dark" data-status>업로드 중...</p>
            <button class="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-8 px-3 bg-transparent text-secondary-light dark:text-secondary-dark gap-1 text-sm font-bold transition-colors hover:bg-slate-100 dark:hover:bg-surface-dark" data-cancel>
                <span class="material-symbols-outlined text-base">cancel</span>
                <span class="truncate">취소</span>
            </button>
        </div>
    `;

    progressSection.appendChild(itemDiv);

    // 취소 버튼 이벤트
    const cancelBtn = itemDiv.querySelector('[data-cancel]');
    cancelBtn.addEventListener('click', () => cancelUpload(fileId));

    // 상태 저장
    uploadingFiles.set(fileId, {
        file,
        status: 'uploading',
        element: itemDiv
    });
}

/**
 * 진행률 업데이트
 */
function updateProgress(fileId, percent) {
    const item = uploadingFiles.get(fileId);
    if (!item || !item.element) return;

    const progressText = item.element.querySelector('[data-progress]');
    const progressBar = item.element.querySelector('[data-progress-bar]');

    if (progressText) progressText.textContent = `${percent}%`;
    if (progressBar) progressBar.style.width = `${percent}%`;
}

/**
 * 업로드 완료 표시
 */
function showCompleted(fileId, video) {
    const item = uploadingFiles.get(fileId);
    if (!item || !item.element) return;

    const statusContainer = item.element.querySelector('.flex.items-center.justify-between');
    statusContainer.innerHTML = `
        <p class="text-sm font-normal text-success">업로드 완료! 얼굴 분석 중...</p>
    `;

    // 진행률 100%로 설정
    updateProgress(fileId, 100);

    // 얼굴 분석 상태 폴링 시작
    startAnalysisPolling(fileId, video.id);
}

/**
 * 얼굴 분석 상태 폴링
 * 비디오 상태를 주기적으로 확인하여 analyzing -> ready 변화를 감지
 */
async function startAnalysisPolling(fileId, videoId) {
    const item = uploadingFiles.get(fileId);
    if (!item || !item.element) return;

    let pollCount = 0;
    // 시간 제한 제거: 분석이 완료될 때까지 무한 대기

    const pollInterval = setInterval(async () => {
        pollCount++;

        // 시간 제한 제거됨 - 분석 완료될 때까지 계속 폴링

        try {
            // 비디오 상태 확인
            const video = await getVideoDetail(videoId);

            if (video.status === 'ready') {
                // 분석 완료!
                clearInterval(pollInterval);

                const statusContainer = item.element.querySelector('.flex.items-center.justify-between');
                statusContainer.innerHTML = `
                    <p class="text-sm font-normal text-success">분석 완료! 얼굴 선택 단계로 이동하세요.</p>
                    <a href="/video/${videoId}/select/" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg h-8 px-4 bg-primary text-white text-sm font-bold shadow-sm transition-colors hover:bg-primary/90">
                        <span class="truncate">다음 단계로</span>
                    </a>
                `;
            } else if (video.status === 'failed') {
                // 분석 실패
                clearInterval(pollInterval);

                const statusContainer = item.element.querySelector('.flex.items-center.justify-between');
                statusContainer.innerHTML = `
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-error">error</span>
                        <p class="text-sm font-normal text-error">얼굴 분석 실패. 다시 시도해주세요.</p>
                    </div>
                `;
            } else {
                // 아직 진행 중 - 진행률 업데이트
                const statusText = item.element.querySelector('[data-status]');
                if (statusText) {
                    statusText.textContent = `얼굴 분석 중... (${video.progress || 0}%)`;
                }
            }
        } catch (error) {
            console.error('분석 상태 확인 실패:', error);
            // 에러가 나도 계속 폴링 (일시적인 네트워크 에러일 수 있음)
        }
    }, 2000); // 2초마다 체크
}

/**
 * 업로드 에러 표시
 */
function showUploadError(fileId, errorMessage) {
    const item = uploadingFiles.get(fileId);
    if (!item || !item.element) return;

    const statusContainer = item.element.querySelector('.flex.items-center.justify-between');
    statusContainer.innerHTML = `
        <div class="flex items-center gap-2">
            <span class="material-symbols-outlined text-error">error</span>
            <p class="text-sm font-normal text-error">${errorMessage}</p>
        </div>
    `;
}

/**
 * 에러 알림 표시 (독립적인 에러 항목)
 */
function showError(fileName, errorMessage) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'flex flex-col gap-3 p-4 border border-error/30 rounded-lg bg-error/5';
    errorDiv.innerHTML = `
        <div class="flex items-center justify-between">
            <p class="text-base font-medium">${fileName}</p>
            <span class="material-symbols-outlined text-error">error</span>
        </div>
        <p class="text-sm font-normal text-error">${errorMessage}</p>
    `;

    progressSection.style.display = 'flex';
    progressSection.appendChild(errorDiv);

    // 5초 후 자동 제거
    setTimeout(() => {
        errorDiv.remove();
        if (progressSection.children.length === 0) {
            progressSection.style.display = 'none';
        }
    }, 5000);
}

/**
 * 업로드 취소
 */
function cancelUpload(fileId) {
    const item = uploadingFiles.get(fileId);
    if (!item) return;

    // XHR 요청이 있으면 취소
    if (item.xhr) {
        item.xhr.abort();
    }

    // UI에서 제거
    if (item.element) {
        item.element.remove();
    }

    // 상태에서 제거
    uploadingFiles.delete(fileId);

    // 모든 업로드 항목이 제거되면 섹션 숨김
    if (progressSection.children.length === 0) {
        progressSection.style.display = 'none';
    }
}
