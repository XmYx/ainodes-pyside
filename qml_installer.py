import os
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QLabel, QPushButton, QVBoxLayout, \
    QHBoxLayout, QFileDialog, QWidget, QListView
import subprocess
import os
from PySide6 import QtWidgets, QtGui, QtCore

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Package Manager")
        self.setStyleSheet("QMainWindow {background-color: #2c2c2c;}")
        self.list_view = QListView()
        self.packageList = QListWidget(self.list_view)
        self.packageList.setStyleSheet("color: white; background-color: #424242;")
        self.populatePackageList()

        self.installButton = QPushButton("Install All")
        self.installButton.setStyleSheet("color: white; background-color: #424242;")
        self.installButton.clicked.connect(self.installAllPackages)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.list_view)
        mainLayout.addWidget(self.installButton)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def populatePackageList(self):
        with open("requirements_versions.txt", "r") as f:
            packages = [line.strip() for line in f.readlines()]

        installed_packages = set(
            subprocess.run(
                ["pip", "freeze"],
                stdout=subprocess.PIPE,
                encoding="utf-8",
            ).stdout.strip().split("\n")
        )

        self.packageList.clear()
        for package in packages:
            if package not in installed_packages:
                item = QListWidgetItem()

                widget = QWidget()
                layout = QHBoxLayout()
                layout.setContentsMargins(0, 0, 0, 0)
                widget.setLayout(layout)
                widget.setMinimumSize(100, 50)
                label = QLabel(package)
                layout.addWidget(label)

                installButton = QPushButton("Install")
                installButton.setStyleSheet("color: white; background-color: #424242;")
                installButton.clicked.connect(lambda: self.installPackage(package))
                layout.addWidget(installButton)
                item.setSizeHint(widget.sizeHint())
                self.packageList.setItemWidget(item, widget)
                self.packageList.addItem(item)


    def installPackage(self, package):
        subprocess.run(["pip", "install", package])
        self.populatePackageList()

    def installAllPackages(self):
        with open("requirements_versions.txt", "r") as f:
            packages = [line.strip() for line in f.readlines()]

        for package in packages:
            subprocess.run(["pip", "install", package])
        self.populatePackageList()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
