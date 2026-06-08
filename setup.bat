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
echo   4. Offer to launch the Gradio app in your browser
echo   5. Print a simple sample verification command for later
echo.
echo RTX 4070 8GB note: setup uses safe local defaults: 720p,
echo batch size 1, VRAM safety, and RunPod prompts for heavy jobs.
echo.
echo Tip: If Windows asks for network access, allow Python so Gradio can open locally.
echo.

set "PYTHON_CMD="
echo [Step 1/4] Looking for a supported Python 3.12 interpreter...
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    for %%V in (3.12) do (
        if not defined PYTHON_CMD (
            py -%%V -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
            if not errorlevel 1 set "PYTHON_CMD=py -%%V"
        )
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        python -c "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo.
    echo [ERROR] No supported Python interpreter was found on PATH.
    echo.
    echo Please install Python 3.12 from:
    echo   https://www.python.org/downloads/windows/
    echo.
    echo Python 3.13+ is intentionally blocked because the pinned NumPy 1.26.x,
    echo MoviePy, and OpenCV compatibility stack targets Python 3.12. During installation, check
    echo "Add python.exe to PATH".
    echo After installing Python, close this window and run setup.bat again.
    echo.
    pause
    exit /b 1
)

echo [OK] Supported Python command found: !PYTHON_CMD!
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
echo [Step 2/4] Preparing Python package installer...
echo This is usually quick. If it fails, setup will continue with your current pip.
!PYTHON_CMD! -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed. Continuing with the existing pip version.
) else (
    echo [OK] Pip is ready.
)

echo.
echo [Step 3/4] Installing Futa-Vision requirements...
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
echo [Step 4/4] Running Futa-Vision installer / repair checks...
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
echo [Verification] Running a quick sample media verification...
!PYTHON_CMD! installer.py test-samples
if errorlevel 1 (
    echo.
    echo [WARNING] Sample verification reported a problem.
    echo The app can still open, but please review logs\installer.log
    echo and use the Settings tab Repair button before generation.
) else (
    echo [OK] Sample verification completed.
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
echo Quick verification command for later:
echo   !PYTHON_CMD! installer.py test-samples
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
