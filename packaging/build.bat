@echo off
setlocal
echo ========================================================
echo Building Standalone ARGUSShield.exe
echo ========================================================

cd /d "%~dp0"
pyinstaller build_exe.spec --noconfirm

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo Build Successful: packaging\dist\ARGUSShield.exe
    echo ========================================================
) else (
    echo.
    echo Build failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
