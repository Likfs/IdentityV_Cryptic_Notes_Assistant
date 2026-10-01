@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" -X utf8 "tools\collector\run_fixed_roi_collector.py" %*
if errorlevel 1 pause
