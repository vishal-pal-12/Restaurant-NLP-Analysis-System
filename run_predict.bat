@echo off
title Restaurant NLP Interactive Console
echo ===================================================
echo Starting Restaurant NLP Prediction Console...
echo ===================================================
cd /d "%~dp0"
python predict.py
pause
