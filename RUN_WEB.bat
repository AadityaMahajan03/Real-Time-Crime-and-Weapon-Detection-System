@echo off
title FAST-TRACK AI - Modern Next.js Command Center
cd /d "%~dp0\web"
echo =====================================================================
echo  FAST-TRACK AI: NEXT.JS 14 MODERN SECURITY COMMAND CENTER
echo =====================================================================
echo [*] Starting Next.js development server on port 3000...
echo [*] Opening browser to http://localhost:3000
echo.
start "" http://localhost:3000
npm run dev
pause
