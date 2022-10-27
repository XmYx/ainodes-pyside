import platform

from PySide6.QtGui import QIcon, QAction, QPixmap
#from PySide6.QtWidgets import *
import sys, os

from PySide6.QtWidgets import QApplication, QSplashScreen

"""if (os.name == 'nt'):
    #This is needed to display the app icon on the taskbar on Windows 7
    import ctypes
    myappid = 'aiNodes' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

from PySide6.QtCore import QCoreApplication, Qt
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)"""




if __name__ == "__main__":
    app = QApplication(sys.argv)
    from frontend.mainwindow import GenerateWindow
    # Create the tray
    #tray = QSystemTrayIcon()
    #icon = QIcon('frontend/main/splash_2.png')
    #tray.setIcon(icon)
    #tray.setVisible(True)

    # Create the menu
    #menu = QMenu()
    #action = QAction("A menu item")
    #menu.addAction(action)

    # Add a Quit option to the menu.
    #quit = QAction("Quit")
    #quit.triggered.connect(app.quit)
    #menu.addAction(quit)

    # Add the menu to the tray
    #tray.setContextMenu(menu)

    pixmap = QPixmap('frontend/main/splash_2.png')
    splash = QSplashScreen(pixmap)
    splash.show()
    icon = QIcon('frontend/main/splash_2.png')


    sshFile="frontend/style/QTDark.stylesheet"

    mainWindow = GenerateWindow()
    mainWindow.defaultMode()

    #mainWindow = paintwindow_func.MainWindow()
    if "macOS" in platform.platform():
        #gs.platform = "macOS"
        mainWindow.prepare_loading()


    mainWindow.w.setWindowTitle("aiNodes")
    mainWindow.w.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.w.setStyleSheet(fh.read())
    #app.setIcon(QIcon('frontend/main/splash_2.png'))
    mainWindow.w.show()

    mainWindow.w.resize(1280, 720)
    splash.finish(mainWindow.w)

    #mainWindow.w.thumbnails.tileAction.triggered.connect(mainWindow.tileImageClicked)
    mainWindow.w.prompt.w.runButton.clicked.connect(mainWindow.taskSwitcher)

    #mainWindow.w.actionSoft_Restart.triggered.connect(restart_with_reloader)
    mainWindow.w.actionNodes.triggered.connect(mainWindow.show_nodes)
    mainWindow.w.sampler.w.scale.valueChanged.connect(mainWindow.update_scaleNumber)
    mainWindow.w.sizer_count.w.gfpganSlider.valueChanged.connect(mainWindow.update_gfpganNumber)

    mainWindow.w.preview.w.zoomInButton.clicked.connect(mainWindow.zoom_IN)
    mainWindow.w.preview.w.zoomOutButton.clicked.connect(mainWindow.zoom_OUT)


    mainWindow.outpaint.canvas.signals.outpaint_signal.connect(mainWindow.deforum_outpaint_thread)
    mainWindow.outpaint.canvas.signals.txt2img_signal.connect(mainWindow.deforum_txt2img_thread)
    #mainWindow.outpaint.canvas.signals.outpaint_signal_direct.connect(mainWindow.run_deforum_outpaint)
    mainWindow.outpaint.canvas.signals.txt2img_signal_direct.connect(mainWindow.run_deforum_txt2img)


    mainWindow.outpaint_controls.w.redoButton.clicked.connect(mainWindow.redo_current_outpaint)
    mainWindow.outpaint_controls.w.delButton.clicked.connect(mainWindow.delete_outpaint_frame)
    mainWindow.outpaint_controls.w.widthSlider.valueChanged.connect(mainWindow.update_outpaint_parameters)
    mainWindow.outpaint_controls.w.heightSlider.valueChanged.connect(mainWindow.update_outpaint_parameters)
    mainWindow.outpaint.canvas.signals.update_selected.connect(mainWindow.show_outpaint_details)
    mainWindow.outpaint_controls.w.offsetSlider.valueChanged.connect(mainWindow.outpaint_offset_signal)

    mainWindow.outpaint_controls.w.previewBatch.clicked.connect(mainWindow.preview_batch_outpaint)
    mainWindow.outpaint_controls.w.runBatch.clicked.connect(mainWindow.run_batch_outpaint_thread)

    mainWindow.w.actionOutpaint.triggered.connect(mainWindow.outpaintMode)
    mainWindow.w.actionDefault.triggered.connect(mainWindow.defaultMode_restore)

    sys.exit(app.exec())


#run_with_reloader(main())
