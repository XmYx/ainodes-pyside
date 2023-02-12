@echo off

rem Check for Anaconda in the system profile
if exist "%ProgramData%\Anaconda3\Scripts\conda.bat" (
  call "%ProgramData%\Anaconda3\Scripts\conda.bat" activate base
  goto end
)

rem Check for Anaconda in the user profile
if exist "%userprofile%\Anaconda3\Scripts\conda.bat" (
  call "%userprofile%\Anaconda3\Scripts\conda.bat" activate base
  goto end
)

rem Check for Miniconda in the system profile
if exist "%ProgramData%\Miniconda3\Scripts\activate.bat" (
  call "%ProgramData%\Miniconda3\Scripts\activate.bat" activate base
  goto end
)

rem Check for Miniconda in the user profile
if exist "%userprofile%\Miniconda3\Scripts\activate.bat" (
  call "%userprofile%\Miniconda3\Scripts\activate.bat" activate base
  goto end
)
echo Error: Conda was not found in either the system or user profile.
pause
goto end

:end


call conda update -n base -c defaults conda -y
call conda activate base
call conda env remove -n ai-nodes
call conda env create -f environment-installer_debug.yaml
call conda activate ai-nodes
setup.bat
