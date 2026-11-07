#!/usr/bin/env bash
# aiNodes for Linux and macOS: creates .venv on first run, then starts the app.
# Options: --install-only, --reinstall, --test, --cpu (see launcher.py)
set -e
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    exec uv run --no-project --python 3.12 python launcher.py "$@"
fi

for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null; then
        exec "$candidate" launcher.py "$@"
    fi
done

echo "aiNodes needs Python 3.10-3.12 or uv (https://docs.astral.sh/uv/)." >&2
exit 1
