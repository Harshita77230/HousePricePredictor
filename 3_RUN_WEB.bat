@echo off
title House Price Predictor — Web UI
echo.
echo  Installing / updating dependencies...
pip install -r requirements.txt --quiet
echo.
echo  Starting web server...
echo  Open your browser at:  http://127.0.0.1:5000
echo.
python app.py
pause
