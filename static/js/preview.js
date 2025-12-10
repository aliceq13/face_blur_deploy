/**
 * 비디오 미리보기 및 처리 시작 페이지 JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeProcessingButton();

    // processing 상태면 폴링 시작
    if (VIDEO_STATUS === 'processing') {
        startStatusPolling();
    }
});

/**
 * 처리 시작 버튼 초기화
 */
function initializeProcessingButton() {
    const startBtn = document.getElementById('start-processing-btn');
    if (!startBtn) return;

    startBtn.addEventListener('click', async () => {
        // 확인 다이얼로그
        const confirm = window.confirm('선택한 얼굴을 제외하고 나머지를 블러 처리합니다. 계속하시겠습니까?');
        if (!confirm) return;

        // 버튼 비활성화
        startBtn.disabled = true;
        startBtn.innerHTML = `
            <span class="material-symbols-outlined mr-2 animate-spin">progress_activity</span>
            <span class="truncate">처리 시작 중...</span>
        `;

        try {
            // API 호출
            const result = await startVideoProcessing(VIDEO_ID);

            // 페이지 새로고침하여 상태 업데이트 확인
            setTimeout(() => {
                location.reload();
            }, 1000);

        } catch (error) {
            console.error('처리 시작 실패:', error);
            alert(`처리 시작 실패: ${error.message}`);

            // 버튼 복원
            startBtn.disabled = false;
            startBtn.innerHTML = `
                <span class="material-symbols-outlined mr-2">play_arrow</span>
                <span class="truncate">비디오 처리 시작</span>
            `;
        }
    });
}

/**
 * 처리 상태 폴링 (processing 상태일 때)
 */
async function startStatusPolling() {
    const pollInterval = 3000; // 3초마다

    const poll = async () => {
        try {
            const video = await getVideoDetail(VIDEO_ID);

            // 상태가 변경되면 페이지 새로고침
            if (video.status !== VIDEO_STATUS) {
                location.reload();
                return;
            }

            // 진행률 업데이트 (있다면)
            updateProgressBar(video.progress);

            // 계속 폴링
            if (video.status === 'processing') {
                setTimeout(poll, pollInterval);
            }

        } catch (error) {
            console.error('상태 조회 실패:', error);
            // 에러 발생 시에도 계속 폴링 시도
            setTimeout(poll, pollInterval);
        }
    };

    // 첫 폴링 시작
    setTimeout(poll, pollInterval);
}

/**
 * 진행률 바 업데이트
 */
function updateProgressBar(progress) {
    const progressBar = document.querySelector('[style*="width:"]');
    const progressText = progressBar?.parentElement?.nextElementSibling;

    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
    if (progressText) {
        progressText.textContent = `${progress}%`;
    }
}
