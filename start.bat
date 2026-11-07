@echo off
rem aiNodes for Windows: creates .venv on first run, then starts the app.
rem Options: --install-only, --reinstall, --test, --cpu (see launcher.py)
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if %ERRORLEVEL% == 0 (
    uv run --no-project --python 3.12 python launcher.py %*
    goto end
)

for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "" >nul 2>nul
    if not errorlevel 1 (
        py -%%V launcher.py %*
        goto end
    )
)

python -c "import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))" >nul 2>nul
if not errorlevel 1 (
    python launcher.py %*
    goto end
)

echo aiNodes needs Python 3.10-3.12 or uv (https://docs.astral.sh/uv/).
pause

:end
endlocal
