@echo off
echo ============================================================
echo KEYSTROKE GAN DEMO - SETUP (WINDOWS)
echo ============================================================
echo.

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please make sure Python is installed and in PATH
    pause
    exit /b 1
)
echo     Virtual environment created!

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo     Virtual environment activated!

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo     Pip upgraded!

echo.
echo [4/5] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo     Dependencies installed!

echo.
echo [5/5] Creating directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\models" mkdir "data\models"
if not exist "logs" mkdir "logs"
echo     Directories created!

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Run: create_sample_data.bat   (to generate test data)
echo   2. Run: train_models.bat         (to train AI models)
echo   3. Run: run_demo.bat             (to start demo)
echo.
pause
