@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0..\..\"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
set "DEMO_SCRIPT=%PROJECT_DIR%tools\demos\run_layout_recognition_demo.py"

if not exist "%PYTHON_EXE%" goto missing_python
"%PYTHON_EXE%" -X utf8 "%DEMO_SCRIPT%" %*
set "DEMO_EXIT=%ERRORLEVEL%"
if not "%DEMO_EXIT%"=="0" pause
exit /b %DEMO_EXIT%

:missing_python
echo 未找到项目虚拟环境：%PYTHON_EXE%
pause
exit /b 1
