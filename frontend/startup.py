import os
import shutil
import sys

from PySide6 import QtCore, QtQuick
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtQuick import QSGRendererInterface
from PySide6.QtWidgets import QApplication, QSplashScreen

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# for safety as in some very seldom cases the pycache seems to harm the application
def clean_pycache():
    for root, dirs, files in os.walk(os.getcwd()):
        for dir in dirs:
            if dir == '__pycache__':
                shutil.rmtree(os.path.join(root, dir))
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))

def run_app():

    clean_pycache()

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

