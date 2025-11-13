@echo off
echo ============================================================
echo TRAINING MODELS
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo [1/2] Training Authentication Model...
echo.
python -m src.training.train_auth --raw-data data\raw\sample_data.json
if errorlevel 1 (
    echo ERROR: Authentication training failed
    pause
    exit /b 1
)

echo.
echo [2/2] Training GAN Model (this may take 5-15 minutes)...
echo.
python -m src.training.train_gan --raw-data data\raw\sample_data.json --epochs 500
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
echo You can now run the demo:
echo   run_demo.bat
echo.
pause
