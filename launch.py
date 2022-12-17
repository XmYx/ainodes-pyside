import os
import subprocess


def create_venv(venv_path):
    subprocess.run(["python", "-m", "virtualenv", "--python=python3.10", venv_path])


def activate_venv(venv_path):
    activate_this = os.path.join(venv_path, "Scripts", "activate.bat")
    subprocess.run([activate_this])
    #exec(open(activate_this).read(), {'__file__': activate_this})

if __name__ == "__main__":

    python = "test_venv/Scripts/python.exe"
    activate_this = "test_venv/Scripts/activate_this.py"
    exec(open(activate_this).read(), {'__file__': activate_this})
    subprocess.run([python, "installer.py"])
