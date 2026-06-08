@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title Futa-Vision Setup

echo.
echo ============================================================
echo   Futa-Vision Setup - friendly Windows launcher
echo ============================================================
echo.
echo This setup will check Python, install required packages,
echo run the Futa-Vision installer, and optionally launch the app.
echo Recommended local preset: RTX 4070 8GB / low VRAM safe mode.
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH.
    echo.
    echo Please install Python 3.11 or newer from https://www.python.org/
    echo IMPORTANT: check "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

if not exist requirements.txt (
    echo [ERROR] requirements.txt was not found in this folder.
    echo Please run setup.bat from the Futa-Vision project folder.
    echo.
    pause
    exit /b 1
)

echo Checking Python package requirements...
python -m pip show gradio >nul 2>nul
if errorlevel 1 (
    echo Installing required packages. This may take several minutes...
    python -m pip install --upgrade pip
    if errorlevel 1 goto :pip_failed
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :pip_failed
) else (
    echo [OK] Core packages already look installed.
    echo If you recently updated Futa-Vision, setup will still refresh requirements now.
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :pip_failed
)

echo.
echo Launching the Futa-Vision installer...
echo Please read and answer any prompts in this window.
echo.
python installer.py
if errorlevel 1 goto :installer_failed

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
set /p LAUNCH_APP="Launch Futa-Vision now? (Y/n): "
if /I "%LAUNCH_APP%"=="N" goto :done
if /I "%LAUNCH_APP%"=="NO" goto :done

echo.
echo Starting Futa-Vision. When it opens, visit http://127.0.0.1:7860
echo Press Ctrl+C in this window to stop the app.
echo.
python main.py
goto :done

:pip_failed
echo.
echo [ERROR] Package installation failed.
echo Try running this command manually:
echo   python -m pip install -r requirements.txt
echo.
pause
exit /b 1

:installer_failed
echo.
echo [ERROR] The installer reported a problem.
echo Please check logs\installer.log, then try:
echo   python installer.py repair --all
echo.
pause
exit /b 1

:done
echo.
echo You can launch Futa-Vision any time with:
echo   python main.py
echo.
pause
endlocal
