@echo off
echo ========================================================
echo   Menjalankan Aplikasi Face Shape & Vision Studio
echo ========================================================
cd /d "%~dp0"
py -m streamlit run app.py
pause
