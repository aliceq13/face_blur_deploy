/**
 * 얼굴 선택 페이지 JavaScript
 * 얼굴 썸네일 표시 및 선택/해제 토글 기능
 */

// 선택된 얼굴 ID 저장
const selectedFaces = new Set();
let allFaces = [];

// DOM 로드 시 초기화
document.addEventListener('DOMContentLoaded', async function () {
    await loadFaces();
    initializeSubmitButton();
});

/**
 * 얼굴 목록 로드
 */
async function loadFaces() {
    const grid = document.getElementById('faces-grid');

    try {
        // API 호출
        allFaces = await getFaceList(VIDEO_ID);

        if (allFaces.length === 0) {
            grid.innerHTML = `
                <div class="col-span-full flex flex-col items-center justify-center py-12 gap-4">
                    <span class="material-symbols-outlined text-6xl text-slate-500">sentiment_dissatisfied</span>
                    <p class="text-slate-500 dark:text-slate-400 text-lg">감지된 얼굴이 없습니다.</p>
                    <a href="/" class="flex items-center justify-center rounded-lg h-10 px-6 bg-primary text-white text-sm font-bold hover:bg-primary/90">
                        처음으로 돌아가기
                    </a>
                </div>
            `;
            return;
        }

        // 그리드 초기화
        grid.innerHTML = '';

        // 각 얼굴에 대해 썸네일 생성
        allFaces.forEach((face, index) => {
            const faceDiv = createFaceThumbnail(face, index);
            grid.appendChild(faceDiv);
        });

    } catch (error) {
        console.error('얼굴 목록 로드 실패:', error);
        grid.innerHTML = `
            <div class="col-span-full flex flex-col items-center justify-center py-12 gap-4">
                <span class="material-symbols-outlined text-6xl text-error">error</span>
                <p class="text-error text-lg">${error.message}</p>
                <button onclick="location.reload()" class="flex items-center justify-center rounded-lg h-10 px-6 bg-primary text-white text-sm font-bold hover:bg-primary/90">
                    다시 시도
                </button>
            </div>
        `;
    }
}

/**
 * 얼굴 썸네일 DOM 생성
 */
function createFaceThumbnail(face, index) {
    const div = document.createElement('div');
    div.className = 'group relative flex cursor-pointer flex-col gap-3 transition-transform duration-200 ease-in-out hover:scale-105';
    div.dataset.faceId = face.id;

    // 기본적으로 모든 얼굴은 블러 처리 (is_blurred: true)
    // 사용자가 클릭하면 블러 처리 안 함 (is_blurred: false)
    const isSelected = !face.is_blurred; // 블러 안 하는 것 = 선택됨

    if (isSelected) {
        selectedFaces.add(face.id);
    }

    div.innerHTML = `
        <div class="w-full bg-center bg-no-repeat aspect-video bg-cover rounded-lg ${isSelected ? 'ring-4 ring-primary shadow-lg' : ''}"
             style='background-image: url("${face.thumbnail_url}");'>
        </div>
        <div class="absolute top-2 right-2 flex size-6 items-center justify-center rounded-full bg-primary text-white ${isSelected ? '' : 'hidden'} selected-indicator">
            <span class="material-symbols-outlined text-sm">check</span>
        </div>
        <div class="flex flex-col gap-1">
            <p class="text-white text-sm font-medium">인물 ${face.instance_id}</p>
            <p class="text-slate-400 text-xs">${face.total_frames}프레임 출현</p>
        </div>
    `;

    // 클릭 이벤트
    div.addEventListener('click', () => toggleFaceSelection(div, face));

    return div;
}

/**
 * 얼굴 선택/해제 토글
 */
async function toggleFaceSelection(element, face) {
    const thumbnail = element.querySelector('div:first-child');
    const indicator = element.querySelector('.selected-indicator');

    const wasSelected = selectedFaces.has(face.id);
    const nowSelected = !wasSelected;

    try {
        // API 호출: is_blurred는 블러 처리 여부이므로, 선택됨 = false, 선택 안 됨 = true
        await updateFaceBlur(face.id, !nowSelected);

        // UI 업데이트
        if (nowSelected) {
            selectedFaces.add(face.id);
            thumbnail.classList.add('ring-4', 'ring-primary', 'shadow-lg');
            indicator.classList.remove('hidden');
        } else {
            selectedFaces.delete(face.id);
            thumbnail.classList.remove('ring-4', 'ring-primary', 'shadow-lg');
            indicator.classList.add('hidden');
        }

    } catch (error) {
        console.error('얼굴 선택 업데이트 실패:', error);
        alert(`선택 업데이트 실패: ${error.message}`);
    }
}

/**
 * 제출 버튼 초기화
 */
function initializeSubmitButton() {
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = false;

    submitBtn.addEventListener('click', async () => {
        // 선택 확인
        const blurCount = allFaces.length - selectedFaces.size;
        if (blurCount === 0) {
            const confirm = window.confirm('블러 처리할 얼굴이 없습니다. 계속하시겠습니까?');
            if (!confirm) return;
        }

        // 다음 단계로 이동
        window.location.href = `/video/${VIDEO_ID}/preview/`;
    });
}
