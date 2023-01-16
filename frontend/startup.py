import sys

from PySide6 import QtCore, QtQuick
from PySide6.QtGui import QIcon, QAction, QPixmap

from PySide6.QtQuick import QSGRendererInterface
from PySide6.QtWidgets import QApplication, QSplashScreen
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_app():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QtQuick.QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGLRhi)
    app = QApplication(sys.argv)


    pixmap = QPixmap('frontend/main/splash_2.png')
    splash = QSplashScreen(pixmap)
    splash.show()
    icon = QIcon('frontend/main/splash_2.png')
    from frontend.mainwindow import MainWindow

    sshFile="frontend/style/elegantDark.stylesheet"

    mainWindow = MainWindow()

    mainWindow.setWindowTitle("aiNodes")
    mainWindow.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.setStyleSheet(fh.read())

    mainWindow.show()

    splash.finish(mainWindow)

    sys.exit(app.exec())


    mainWindow.w.setWindowTitle("aiNodes")
    mainWindow.w.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.w.setStyleSheet(fh.read())

    mainWindow.w.show()

    mainWindow.w.resize(1280, 720)
    splash.finish(mainWindow.w)

    sys.exit(app.exec())
