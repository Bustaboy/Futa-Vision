@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
title Futa-Vision Setup

cls
echo ============================================================
echo   Futa-Vision Setup - Windows Friendly Installer
echo ============================================================
echo.
echo This will check Python, install required packages, and run the
echo Futa-Vision installer. RTX 4070 8GB users will get safe defaults.
echo.

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PYTHON_CMD=python"
    ) else (
        echo [ERROR] Python was not found.
        echo.
        echo Please install Python 3.12 or newer from:
        echo   https://www.python.org/downloads/windows/
        echo.
        echo During installation, check "Add python.exe to PATH".
        echo Then close this window and run setup.bat again.
        echo.
        pause
        exit /b 1
    )
)

echo [OK] Python command found: !PYTHON_CMD!
!PYTHON_CMD! --version
if errorlevel 1 (
    echo [ERROR] Python exists but could not be started.
    pause
    exit /b 1
)

echo.
echo [1/3] Upgrading pip...
!PYTHON_CMD! -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo [WARNING] Pip upgrade failed. Continuing with the existing pip version.
)

echo.
echo [2/3] Installing Futa-Vision requirements...
echo This can take several minutes, especially for PyTorch/CUDA packages.
!PYTHON_CMD! -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo Please check your internet connection and available disk space, then run setup.bat again.
    echo If PyTorch fails, install the CUDA wheel recommended for your NVIDIA driver and retry.
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] Running the Futa-Vision installer...
echo The installer will create folders, detect ComfyUI/Ostris/Pinokio, and save setup status.
!PYTHON_CMD! installer.py
if errorlevel 1 (
    echo.
    echo [ERROR] The installer reported a problem.
    echo Please review logs\installer.log, fix the issue shown above, and run setup.bat again.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo You can start Futa-Vision now or launch it later with:
echo   !PYTHON_CMD! main.py
echo.
choice /C YN /N /M "Launch Futa-Vision now? [Y/N]: "
if errorlevel 2 goto done

echo.
echo Starting Futa-Vision. When Gradio prints a local URL, open it in your browser.
!PYTHON_CMD! main.py

goto end

:done
echo.
echo Launch skipped. You can run setup.bat again any time for repair/setup checks.

:end
echo.
pause
endlocal
