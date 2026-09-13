@echo off
chcp 65001 >nul
cd /d %~dp0
echo ========================================
echo   毕设工坊 ThesisForge  http://127.0.0.1:8765
echo ========================================
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
