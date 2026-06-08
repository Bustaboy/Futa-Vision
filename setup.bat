@echo off
setlocal EnableExtensions

 title Futa-Vision Setup
 cd /d "%~dp0"

 echo.
 echo ============================================================
 echo   Futa-Vision Director - Friendly Windows Setup
 echo ============================================================
 echo.
 echo This setup is optimized for a local RTX 4070 8GB PC.
 echo It will check Python, install required packages, and run the installer.
 echo.

 where python >nul 2>nul
 if errorlevel 1 (
   echo [ERROR] Python was not found on PATH.
   echo.
   echo Please install Python 3.12 or newer from https://www.python.org/downloads/windows/
   echo IMPORTANT: tick "Add python.exe to PATH" during installation.
   echo Then close this window and run setup.bat again.
   echo.
   pause
   exit /b 1
 )

 echo [1/4] Python found:
 python --version
 if errorlevel 1 (
   echo [ERROR] Python is installed but could not run correctly.
   pause
   exit /b 1
 )
 echo.

 echo [2/4] Checking pip...
 python -m pip --version >nul 2>nul
 if errorlevel 1 (
   echo pip was not available. Trying to enable it with ensurepip...
   python -m ensurepip --upgrade
   if errorlevel 1 (
     echo [ERROR] pip could not be enabled. Please repair your Python install.
     pause
     exit /b 1
   )
 )
 echo pip is ready.
 echo.

 echo [3/4] Installing or updating Futa-Vision requirements...
 echo This can take a while the first time, especially PyTorch.
 python -m pip install --upgrade pip
 python -m pip install -r requirements.txt
 if errorlevel 1 (
   echo.
   echo [ERROR] Dependency installation failed.
   echo Please review the message above. If PyTorch failed, install the CUDA wheel recommended for your driver and rerun setup.bat.
   echo.
   pause
   exit /b 1
 )
 echo Requirements are ready.
 echo.

 echo [4/4] Running the Futa-Vision installer...
 echo The installer will ask a few simple questions and write logs to logs\installer.log.
 python installer.py
 if errorlevel 1 (
   echo.
   echo [ERROR] The installer did not complete successfully.
   echo Open logs\installer.log for details, then run setup.bat again or use Repair Mode in the app Settings tab.
   echo.
   pause
   exit /b 1
 )
 echo.
 echo ============================================================
 echo   Setup complete!
 echo ============================================================
 echo.
 choice /C YN /N /M "Launch Futa-Vision now? [Y/N] "
 if errorlevel 2 goto done
 echo.
 echo Starting the Gradio app. When ready, open http://127.0.0.1:7860
 python main.py

:done
 echo.
 echo You can start Futa-Vision later by double-clicking setup.bat or running: python main.py
 pause
 endlocal
