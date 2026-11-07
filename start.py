"""Start the aiNodes UI. Use start.sh / start.bat, which prepare .venv first."""
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # configs/ and data/ are relative paths
os.environ["PYSIDE_DESIGNER_PLUGINS"] = '.'

import backend.paths  # noqa: E402  sets sys.path and library shims before any model code
import frontend.startup  # noqa: E402

print("Launching aiNodes UI")
frontend.startup.run_app()
