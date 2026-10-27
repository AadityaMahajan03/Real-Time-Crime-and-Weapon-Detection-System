@echo off
title FAST-TRACK AI Security Command Center
cd /d "%~dp0"
echo =====================================================================
echo  FAST-TRACK AI: REAL-TIME CRIME AND WEAPON DETECTION SYSTEM
echo =====================================================================
echo [*] Starting Streamlit Command Center...
echo [*] Opening browser to http://localhost:8501
echo.
start "" http://localhost:8501
python run.py ui
pause
