@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
title Futa-Vision Setup

cls
echo ============================================================
echo   Futa-Vision Setup - Guided Windows Installer
echo ============================================================
echo.
echo Welcome! This setup will prepare Futa-Vision on this PC.
echo.
echo What it will do:
echo   1. Find a working Python 3 installation
echo   2. Install or refresh required Python packages
echo   3. Run the Phase 5 installer / repair checks
echo   4. Run a quick sample verification command
echo   5. Offer to launch the Gradio app in your browser
echo.
echo RTX 4070 8GB note: setup uses safe local defaults: 720p,
echo batch size 1, VRAM safety, and RunPod prompts for heavy jobs.
echo.
echo Tip: If Windows asks for network access, allow Python so Gradio can open locally.
echo.

set "PYTHON_CMD="
echo [Step 1/5] Looking for Python...
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON_CMD=py -3"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo.
    echo [ERROR] Python was not found on PATH.
    echo.
    echo Please install Python 3.12 or newer from:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo IMPORTANT: During installation, check "Add python.exe to PATH".
    echo After installing Python, close this window and run setup.bat again.
    echo.
    pause
    exit /b 1
)

echo [OK] Python command found: !PYTHON_CMD!
!PYTHON_CMD! --version
if errorlevel 1 (
    echo.
    echo [ERROR] Python was found but could not be started.
    echo Try reinstalling Python and make sure it is added to PATH.
    echo.
    pause
    exit /b 1
)

echo.
echo [Step 2/5] Preparing Python package installer...
echo This is usually quick. If it fails, setup will continue with your current pip.
!PYTHON_CMD! -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed. Continuing with the existing pip version.
) else (
    echo [OK] Pip is ready.
)

echo.
echo [Step 3/5] Installing Futa-Vision requirements...
echo This can take several minutes because AI/video packages are large.
echo Please keep this window open.
!PYTHON_CMD! -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo.
    echo What to try next:
    echo   - Check your internet connection and free disk space.
    echo   - Update your NVIDIA driver if PyTorch/CUDA packages fail.
    echo   - Run setup.bat again after fixing the issue.
    echo.
    pause
    exit /b 1
)
echo [OK] Requirements installed or already satisfied.

echo.
echo [Step 4/5] Running Futa-Vision installer / repair checks...
echo The installer will create folders, detect ComfyUI/Ostris/Pinokio,
echo run safe sample checks, and update settings\installer_manifest.json.
!PYTHON_CMD! installer.py
if errorlevel 1 (
    echo.
    echo [ERROR] The installer reported a problem.
    echo.
    echo Please review:
    echo   logs\installer.log
    echo.
    echo You can run setup.bat again after following the repair suggestion above.
    echo.
    pause
    exit /b 1
)

echo.
echo [Step 5/5] Running quick verification sample test...
echo Command: !PYTHON_CMD! installer.py test-samples
!PYTHON_CMD! installer.py test-samples
if errorlevel 1 (
    echo.
    echo [WARNING] Verification sample test did not complete successfully.
    echo Review logs\installer.log, then try this command later:
    echo   !PYTHON_CMD! installer.py test-samples
) else (
    echo [OK] Verification sample image/clip test completed.
)

echo.
echo ============================================================
echo   Installation completed successfully!
echo ============================================================
echo.
echo Next steps:
echo   1. Launch Futa-Vision and open the Settings tab.
echo   2. Confirm the Phase 5 Installer Status looks green/ready.
echo   3. If ComfyUI or Ostris paths are missing, install or set them and run Repair.
echo   4. Start with RTX 4070 8GB Safe presets before trying heavier jobs.
echo.
echo Launch command for later:
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
echo Launch skipped. You can start later with: !PYTHON_CMD! main.py
echo You can also run setup.bat again any time to refresh dependencies and repair setup.

:end
echo.
pause
endlocal
