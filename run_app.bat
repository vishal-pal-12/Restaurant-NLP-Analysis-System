@echo off
title Restaurant NLP Analysis System
echo ===================================================
echo Starting Restaurant NLP Analysis System...
echo ===================================================
cd /d "%~dp0"
streamlit run app.py
pause
