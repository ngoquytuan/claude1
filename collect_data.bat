@echo off
echo ============================================================
echo KEYSTROKE DATA COLLECTION
echo ============================================================
echo.
echo This will open a GUI to collect your keystroke data.
echo You need to type "security2025" correctly 100 times.
echo.
pause

call venv\Scripts\activate.bat

python -m src.data_collection.collector

echo.
echo Data collection complete!
echo Data saved to: data\raw\
echo.
pause
