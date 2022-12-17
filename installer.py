import importlib
import os
import subprocess
import pkg_resources
subprocess.run(["pip", "install", "pyside6"])
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel


subprocess.run(["pip", "install", "virtualenv"])



from PySide6 import QtWidgets, QtGui


def run(command, desc=None, errdesc=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")

def create_venv(venv_path):
    subprocess.run(["python", "-m", "virtualenv", "--python=python3.10", venv_path])


def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def activate_venv(venv_path):
    activate_this = os.path.join(venv_path, "Scripts", "activate.bat")
    subprocess.run([activate_this])
    print(is_package_installed("k-diffusion"))
    python = "test_venv/Scripts/python.exe"
    print(subprocess.run([python, "--version"]))

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.installButton = QtWidgets.QPushButton('Install All', self)
        self.installButton.clicked.connect(self.installPackages)

        self.runButton = QtWidgets.QPushButton('Run aiNodes', self)
        self.runButton.clicked.connect(self.run_aiNodes)
        # Create a layout to hold the list widget and button
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.packageList)
        layout.addWidget(self.installButton)
        layout.addWidget(self.runButton)

    def initUI(self):
        # Create a list widget to display the packages
        self.packageList = QtWidgets.QListWidget(self)
        self.install_buttons = {}
        # Read the requirements.txt file and add each package to the list widget
        with open('requirements_versions.txt', 'r') as f:
            for line in f:
                package = line.strip()
                item = QtWidgets.QListWidgetItem(package)
                print(package.split('==')[0])
                # Check if the package is installed and set the item's color accordingly
                #if self.isPackageInstalled(package.split('==')[0]):
                #    item.setForeground(QtGui.QColor('green'))
                #else:
                #    item.setForeground(QtGui.QColor('red'))
                if is_package_installed(package.split('==')[0]):
                    item.setForeground(QtGui.QColor('green'))
                else:
                    item.setForeground(QtGui.QColor('red'))


                widget = QWidget()
                layout = QHBoxLayout()
                widget.setLayout(layout)
                label = QLabel(package)
                button = QPushButton("Install")
                button.setMaximumWidth(200)
                button.clicked.connect(self.install_package)
                self.install_buttons[button] = package
                layout.addWidget(button)
                layout.addWidget(label)
                item.setSizeHint(widget.sizeHint())



                self.packageList.addItem(item)
                self.packageList.setItemWidget(item, widget)

        # Create a button to install all the packages
    def install_package(self):
        python = "python"
        button = self.sender()
        requirement = self.install_buttons[button]
        if 'torch' in requirement:
            torch_command = os.environ.get('TORCH_COMMAND',
                                           "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")

            run(f'{torch_command}', "Installing torch and torchvision", "Couldn't install torch")
        elif 'xformers' in requirement:
            subprocess.run(["pip", "install", "xformers-0.0.15.dev0+4601d9d.d20221216-cp310-cp310-win_amd64.whl"])
        else:
            subprocess.run(["pip", "install", requirement])
        #reinitUI()
    def run_aiNodes(self):
        print(f"Launching SD UI")
        #launch = 'frontend/main_app.py'
        #exec(open(launch).read(), {'__file__': launch})

        import frontend.startup
        frontend.startup.run_app()

    def isPackageInstalled(self, package):
        """Returns True if the given package is installed, False otherwise."""
        installed = subprocess.run(["pip", "freeze"], capture_output=True)
        return installed
    def installPackages(self):
        """Installs all the packages listed in the requirements.txt file."""
        subprocess.run(['pip', 'install', '-r', "--extra-index-url", "https://download.pytorch.org/whl/cu117", 'requirements_versions.txt'])
def reinitUI():
    global window
    #window.destroy()
    #window = MainWindow()
    #window.show()
    window.layout().removeWidget(window.packageList)
    window.initUI()
    window.layout().addWidget(window.packageList)

if __name__ == '__main__':
    global window
    #create_venv('test_venv')
    #activate_venv('test_venv')
    app = QtWidgets.QApplication()
    window = MainWindow()
    window.show()
    app.exec_()


'''import os
import subprocess
subprocess.run(["pip", "install", "pyside6"])
from PySide6 import QtWidgets, QtCore, QtGui
def create_venv(venv_path):
    subprocess.run(["python", "-m", "venv", venv_path])



def activate_venv():
    venv_path = os.path.join(os.getcwd(), "venv")
    activate_this = os.path.join(venv_path, "Scripts", "activate.bat")
    subprocess.run(["cmd.exe", "/C", activate_this])

class ListView(QtWidgets.QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        venv_path = 'venv'
        create_venv(venv_path)

        activate_venv()

        # Set up the model
        self.model = QtGui.QStandardItemModel(self)

        # Add items to the model
        with open("requirements_versions.txt") as f:
            items = f.read().splitlines()

            # Add items to the model
        for item in items:
            list_item = QtGui.QStandardItem(item)
            button = QtWidgets.QPushButton("Install")
            list_item.setSizeHint(button.sizeHint())
            self.model.appendRow(list_item)

        # Set the model
        self.setModel(self.model)

        # Set the delegate
        self.setItemDelegate(ButtonDelegate(self))

class ButtonDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        button = QtWidgets.QPushButton(parent)
        button.setText(index.data())
        button.clicked.connect(self.handleButtonClick)
        return button

    def setEditorData(self, editor, index):
        editor.setText(index.data())

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text())

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def handleButtonClick(self):
        index = self.parent().currentIndex()
        package_name = index.data()

        installed = subprocess.run(["pip", "freeze"], capture_output=True)
        if package_name in installed.stdout.decode():
            # Set the button color to green
            button = self.sender()
            button.setStyleSheet("background-color: green")
            print("Package already installed: {}".format(package_name))
        else:
            # Install the package using pip
            subprocess.run(["pip", "install", package_name])
            print("Installed package: {}".format(package_name))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    view = ListView()
    view.show()
    app.exec()'''




'''import os
import sys
import subprocess

from PySide6.QtCore import Qt

subprocess.run(["pip", "install", "pyside6"])

from PySide6 import QtWidgets


def create_venv(venv_path):
    subprocess.run(["python", "-m", "venv", venv_path])



def activate_venv():
    venv_path = os.path.join(os.getcwd(), "test_venv")
    activate_this = os.path.join(venv_path, "Scripts", "activate.bat")
    subprocess.run(["cmd.exe", "/C", activate_this])
class TestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the QListWidget
        self.list_widget = QtWidgets.QListWidget()

        # Add items to the QListWidget
        items = ["Item 1", "Item 2", "Item 3"]
        for item in items:
            list_item = QtWidgets.QListWidgetItem(item)
            list_item.setFlags(Qt.ItemIsEnabled)
            button = QtWidgets.QPushButton("Button")
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

            self.list_widget.setItemWidget(list_item, button)
            self.list_widget.addItem(list_item)

        # Set the QListWidget as the central widget
        self.list_widget.layout().setStretchFactor(0, 1)
        self.setCentralWidget(self.list_widget)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        venv_path = 'test_venv'
        create_venv(venv_path)

        activate_venv()

        self.install_list = QtWidgets.QListWidget()
        self.install_buttons = {}
        with open("requirements_versions.txt") as f:
            for requirement in f:
                requirement = requirement.strip()
                if not requirement:
                    continue
                install_button = QtWidgets.QPushButton("Install")
                install_button.clicked.connect(self.install_package)
                self.install_buttons[requirement] = install_button
                item = QtWidgets.QListWidgetItem(requirement)
                item.clicked.connect(self.install_package)
                self.install_list.setItemWidget(item, install_button)
                self.install_list.addItem(item)


        self.setCentralWidget(self.install_list)

    def install_package(self):
        button = self.sender()
        requirement = self.install_buttons[button]
        subprocess.run(["pip", "install", requirement])

if __name__ == "__main__":
    subprocess.run(["pip", "install", "pyside6"])

    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())'''
