@echo off
title Restaurant NLP Master Notebook
echo ===================================================
echo Opening Restaurant NLP Master Jupyter Notebook...
echo ===================================================
cd /d "%~dp0"
jupyter notebook restaurant_nlp_mini_project.ipynb
if %errorlevel% neq 0 (
    python -m notebook restaurant_nlp_mini_project.ipynb
)
pause
