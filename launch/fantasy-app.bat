@echo off
rem FANTASY app launcher for Windows (double-click or run in a terminal).
rem Runs the app with the recommended Python version and the tested package
rem versions (uv.lock), in an environment of its own.
setlocal
set "PYTHON_VERSION=3.12"
set "HERE=%~dp0.."
if "%FANTASY_ENV%"=="" set "FANTASY_ENV=%USERPROFILE%\.fantasy_agn\env-py%PYTHON_VERSION%"

where uv >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto run

echo FANTASY uses uv (https://docs.astral.sh/uv/) to install Python %PYTHON_VERSION% and its packages.
set /p "ANSWER=Install uv now with its official installer? [y/N] "
if /i not "%ANSWER%"=="y" (
  echo Install uv ^(https://docs.astral.sh/uv/getting-started/installation/^) and run this again.
  pause
  exit /b 1
)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:run
cd /d "%HERE%"
set "UV_PROJECT_ENVIRONMENT=%FANTASY_ENV%"
uv run --locked --python %PYTHON_VERSION% fantasy-app %*
if errorlevel 1 pause
