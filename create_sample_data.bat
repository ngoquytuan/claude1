@echo off
echo ============================================================
echo CREATING SAMPLE DATA
echo ============================================================
echo.

call venv\Scripts\activate.bat

python create_sample_data.py

echo.
echo ============================================================
pause
