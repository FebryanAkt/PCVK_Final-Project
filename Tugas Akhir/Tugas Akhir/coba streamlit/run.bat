@echo off
echo ========================================================
echo   Menjalankan Aplikasi Streamlit VisionLab Studio
echo ========================================================
cd /d "%~dp0"
py -m streamlit run app.py
pause
