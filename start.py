print(f"Launching aiNodes UI")
import frontend.startup
import backend.paths
import os
os.environ["PYSIDE_DESIGNER_PLUGINS"] = '.'

frontend.startup.run_app()
