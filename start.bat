@echo off

rem Check for Anaconda in the system profile
if exist "%ProgramData%\Anaconda3\Scripts\activate.bat" (
  call "%ProgramData%\Anaconda3\Scripts\activate.bat" activate ai-nodes
  goto end
)

rem Check for Anaconda in the user profile
if exist "%userprofile%\Anaconda3\Scripts\activate.bat" (
  call "%userprofile%\Anaconda3\Scripts\activate.bat" activate ai-nodes
  goto end
)

rem Check for Miniconda in the system profile
if exist "%ProgramData%\Miniconda3\Scripts\activate.bat" (
  call "%ProgramData%\Miniconda3\Scripts\activate.bat" activate ai-nodes
  goto end
)

rem Check for Miniconda in the user profile
if exist "%userprofile%\Miniconda3\Scripts\activate.bat" (
  call "%userprofile%\Miniconda3\Scripts\activate.bat" activate ai-nodes
  goto end
)
echo Error: Conda was not found in either the system or user profile.
pause
goto end

:end


set PYSIDE_DESIGNER_PLUGINS = '.'

call activate ai-nodes
call python start.py
