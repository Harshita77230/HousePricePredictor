@echo off
title House Price Predictor — Setup ^& Train
cd /d "%~dp0"
echo.
echo ============================================
echo   STEP 1: Installing required packages...
echo ============================================
pip install numpy pandas scikit-learn matplotlib seaborn joblib
echo.
echo ============================================
echo   STEP 2: Training the ML model...
echo ============================================
python train_model.py
echo.
echo ============================================
echo   Setup complete! You can now run predict.bat
echo ============================================
pause
