
import os
import platform
import sys

from PySide6.QtGui import QIcon, QAction, QPixmap
from PySide6.QtWidgets import *
import sys, os, time

from pyparsing import unicode
from werkzeug._internal import _log

if (os.name == 'nt'):
    #This is needed to display the app icon on the taskbar on Windows 7
    import ctypes
    myappid = 'aiNodes' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from PySide6.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
app = QApplication(sys.argv)


from frontend import mainwindow
from frontend.mainwindow import gs

from frontend import paintwindow_func
#from ui_classes import *

if __name__ == "__main__":

    # Create the tray
    tray = QSystemTrayIcon()
    icon = QIcon('frontend/main/splash_2.png')
    tray.setIcon(icon)
    tray.setVisible(True)

    # Create the menu
    menu = QMenu()
    action = QAction("A menu item")
    menu.addAction(action)

    # Add a Quit option to the menu.
    quit = QAction("Quit")
    quit.triggered.connect(app.quit)
    menu.addAction(quit)

    # Add the menu to the tray
    tray.setContextMenu(menu)





    pixmap = QPixmap('frontend/main/splash_2.png')
    splash = QSplashScreen(pixmap)
    splash.show()
    icon = QIcon('frontend/main/splash_2.png')


    sshFile="frontend/style/QTDark.stylesheet"

    #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

    mainWindow = mainwindow.GenerateWindow()

    #mainWindow = paintwindow_func.MainWindow()
    if "macOS" in platform.platform():
        gs.platform = "macOS"
        mainWindow.prepare_loading()


    mainWindow.w.setWindowTitle("aiNodes")
    mainWindow.w.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.w.setStyleSheet(fh.read())
    #app.setIcon(QIcon('frontend/main/splash.png'))
    mainWindow.w.show()
    #mainWindow.nodeWindow.show()
    mainWindow.w.resize(1280, 720)
    splash.finish(mainWindow.w)
    #mainWindow.progress_thread()

    #mainWindow.thumbnails.setGeometry(680,0,800,600)
    #mainWindow.w.thumbnails.tileAction.triggered.connect(mainWindow.tileImageClicked)
    mainWindow.w.prompt.w.runButton.clicked.connect(mainWindow.taskSwitcher)


    #mainWindow.runner.runButton.clicked.connect(mainWindow.progress_thread)
    mainWindow.w.actionSoft_Restart.triggered.connect(restart_with_reloader)
    mainWindow.w.actionNodes.triggered.connect(mainWindow.nodeWindow.show)
    mainWindow.w.sampler.w.scale.valueChanged.connect(mainWindow.update_scaleNumber)
    mainWindow.w.sizer_count.w.gfpganSlider.valueChanged.connect(mainWindow.update_gfpganNumber)

    mainWindow.w.preview.w.zoomInButton.clicked.connect(mainWindow.zoom_IN)
    mainWindow.w.preview.w.zoomOutButton.clicked.connect(mainWindow.zoom_OUT)
    #mainWindow.timeline.timeline.start()
    #mainWindow.deforum_thread()


    sys.exit(app.exec())

