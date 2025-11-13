@echo off
echo ============================================================
echo TRAINING MODELS (GPU OPTIMIZED)
echo ============================================================
echo.
echo This script uses GPU-optimized parameters:
echo   - Higher epochs (1500) for better quality
echo   - Larger batch size (64) for GPU efficiency
echo   - Estimated time: ~7-8 minutes with GPU
echo.

call venv\Scripts\activate.bat

echo Checking GPU availability...
python check_gpu.py
echo.

pause

echo.
echo [1/2] Training Authentication Model...
echo (This is fast, ~10 seconds)
echo.
python -m src.training.train_auth --raw-data data\raw\sample_data.json
if errorlevel 1 (
    echo ERROR: Authentication training failed
    pause
    exit /b 1
)

echo.
echo [2/2] Training GAN Model (GPU Optimized)...
echo (This will take ~7-8 minutes with GPU)
echo.
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 1500 --batch-size 64
if errorlevel 1 (
    echo ERROR: GAN training failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
echo Models saved to: data\models\
echo   - auth_model.pkl
echo   - gan_model_*.h5
echo.
echo Training used GPU optimization for best quality!
echo.
echo You can now run the demo:
echo   run_demo.bat
echo.
pause
