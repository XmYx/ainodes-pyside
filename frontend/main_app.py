import platform
from PySide6 import QtCore, QtQuick
from PySide6.QtGui import QIcon, QAction, QPixmap
import sys
from PySide6.QtQuick import QSGRendererInterface
from PySide6.QtWidgets import QApplication, QSplashScreen
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

defaultdirs = [
    "data",
    "data/models",
    "data/models/custom",
]
sys.path.append('src/AdaBins')
sys.path.append('src/MiDaS')
sys.path.append('src/pytorch3d-lite')
sys.path.append('src/BLIP')
for i in defaultdirs:
    os.makedirs(i, exist_ok=True)


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    QtQuick.QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGLRhi)
    app = QApplication(sys.argv)


    pixmap = QPixmap('frontend/main/splash_2.png')
    splash = QSplashScreen(pixmap)
    splash.show()
    icon = QIcon('frontend/main/splash_2.png')
    from frontend.mainwindow import MainWindow

    sshFile="frontend/style/QTDark.stylesheet"

    mainWindow = MainWindow()

    mainWindow.setWindowTitle("aiNodes")
    mainWindow.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.setStyleSheet(fh.read())

    mainWindow.show()

    splash.finish(mainWindow)

    sys.exit(app.exec())
