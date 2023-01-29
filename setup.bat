@echo off


rem conda env create -f environment-installer.yaml
rem conda activate ai-pixel


set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=
set PYSIDE_DESIGNER_PLUGINS = '.'

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set VENV_DIR=venv)

set ERROR_REPORTING=FALSE

mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :install
echo Couldn't launch python
goto :show_stdout_stderr

:install
%PYTHON% install.py %*
pause
exit /b

:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo install unsuccessful. Exiting.
pause
