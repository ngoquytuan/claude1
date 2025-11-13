@echo off
echo ============================================================
echo GPU AVAILABILITY CHECK
echo ============================================================
echo.

call venv\Scripts\activate.bat

python check_gpu.py

echo.
pause
