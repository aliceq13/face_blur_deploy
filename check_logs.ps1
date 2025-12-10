# Celery Worker 로그 확인 스크립트

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Celery Worker 로그 확인" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 최근 200줄의 로그를 가져옴
Write-Host "최근 로그를 가져오는 중..." -ForegroundColor Yellow
$logs = docker-compose logs --tail=200 celery_worker 2>&1

# 디버그 로그만 필터링
Write-Host ""
Write-Host "=== [DEBUG] 로그 ===" -ForegroundColor Green
$logs | Select-String -Pattern "\[DEBUG\]" | ForEach-Object { 
    Write-Host $_.Line 
}

Write-Host ""
Write-Host "=== Pass 2 렌더링 로그 ===" -ForegroundColor Green
$logs | Select-String -Pattern "Pass 2|Rendering" | ForEach-Object { 
    Write-Host $_.Line 
}

Write-Host ""
Write-Host "=== Analyzing tracks ===" -ForegroundColor Green
$logs | Select-String -Pattern "Analyzing.*tracks" | ForEach-Object { 
    Write-Host $_.Line 
}

Write-Host ""
Write-Host "로그 파일로도 저장합니다: celery_recent_logs.txt" -ForegroundColor Yellow
docker-compose logs --tail=500 celery_worker > celery_recent_logs.txt
Write-Host "완료!" -ForegroundColor Green
