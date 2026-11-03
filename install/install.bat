@echo off
rem Install FANTASY on Windows and check that it works (double-click or run in a terminal).
rem   install.bat                       the wheel next to this file (tester bundle)
rem   install.bat path\to\fantasy_agn-X.whl   or a git URL
setlocal
set "PYTHON_VERSION=3.12"
set "HERE=%~dp0"
set "SOURCE=%~1"
if "%SOURCE%"=="" for %%f in ("%HERE%fantasy_agn-*.whl") do set "SOURCE=%%~ff"
if "%SOURCE%"=="" (
  echo No FANTASY wheel found next to this file; give one ^(or a git URL^) as argument.
  pause
  exit /b 1
)

where uv >nul 2>nul
if errorlevel 1 if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if not errorlevel 1 goto install
echo FANTASY is installed with uv (https://docs.astral.sh/uv/), which also provides Python %PYTHON_VERSION%.
set /p "ANSWER=Install uv now with its official installer? [y/N] "
if /i not "%ANSWER%"=="y" (
  echo Install uv ^(https://docs.astral.sh/uv/getting-started/installation/^) and run this again.
  pause
  exit /b 1
)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:install
set "CONSTRAINTS="
if exist "%HERE%constraints.txt" set "CONSTRAINTS=--constraints "%HERE%constraints.txt""
echo Installing FANTASY (Python %PYTHON_VERSION%) ...
uv tool install --force --reinstall-package fantasy_agn --python %PYTHON_VERSION% %CONSTRAINTS% "%SOURCE%"
if errorlevel 1 ( pause & exit /b 1 )
for /f "delims=" %%b in ('uv tool dir --bin') do set "PATH=%%b;%PATH%"
uv tool update-shell >nul 2>nul

echo.
echo Checking the installation (downloads the stellar templates, 13 MB, once) ...
fantasy test --full
if errorlevel 1 (
  echo.
  echo Some checks failed. Please send the output above to the FANTASY authors.
  pause
  exit /b 1
)
echo.
echo Installed. In a new terminal:  fantasy app   (or  fantasy --help)
set /p "ANSWER=Start the FANTASY app now? [Y/n] "
if /i "%ANSWER%"=="n" exit /b 0
fantasy app
