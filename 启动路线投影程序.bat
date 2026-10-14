@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch.py %*
) else (
  python launch.py %*
)
set "launch_exit=%errorlevel%"
if not "%launch_exit%"=="0" (
  echo.
  echo Startup failed. See runtime\logs\launcher.log and README.md.
  pause
)
endlocal & exit /b %launch_exit%
