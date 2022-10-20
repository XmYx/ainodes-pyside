import platform
import subprocess
from concurrent.futures import thread
from itertools import chain


from PySide6.QtGui import QIcon, QAction, QPixmap
from PySide6.QtWidgets import *
import sys, os, time


import concurrent.futures

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


def reloader_loop(extra_files=None, interval=1):
    """When this function is run from the main thread, it will force other
    threads to exit when any modules currently loaded change.

    Copyright notice.  This function is based on the autoreload.py from
    the CherryPy trac which originated from WSGIKit which is now dead.

    :param extra_files: a list of additional files it should watch.
    """

    def iter_module_files():
        for module in sys.modules.values():
            filename = getattr(module, '__file__', None)
            if filename:
                old = None
                while not os.path.isfile(filename):
                    old = filename
                    filename = os.path.dirname(filename)
                    if filename == old:
                        break
                else:
                    if filename[-4:] in ('.pyc', '.pyo'):
                        filename = filename[:-1]
                    yield filename

    mtimes = {}
    while 1:
        print('sdscess')
        for filename in chain(iter_module_files(), extra_files or ()):

            try:
                mtime = os.stat(filename).st_mtime
            except OSError:
                continue

            old_time = mtimes.get(filename)
            if old_time is None:
                mtimes[filename] = mtime
                continue
            elif mtime > old_time:
                _log('info', ' * Detected change in %r, reloading' % filename)
                sys.exit(3)
        time.sleep(interval)


def restart_with_reloader():
    """Spawn a new Python interpreter with the same arguments as this one,
    but running the reloader thread.
    """
    while 1:
        _log('info', ' * Restarting with reloader...')
        args = [sys.executable] + sys.argv
        new_environ = os.environ.copy()
        new_environ['WERKZEUG_RUN_MAIN'] = 'true'

        # a weird bug on windows. sometimes unicode strings end up in the
        # environment and subprocess.call does not like this, encode them
        # to latin1 and continue.
        if os.name == 'nt':
            for key, value in new_environ.iteritems():
                if isinstance(value, unicode):
                    new_environ[key] = value.encode('iso-8859-1')

        exit_code = subprocess.call(args, env=new_environ)
        if exit_code != 3:
            return exit_code


def run_with_reloader(main_func, extra_files=None, interval=1):
    """Run the given function in an independent python interpreter."""

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        thread.start_new_thread(main_func, ())
        try:
            reloader_loop(extra_files, interval)
        except KeyboardInterrupt:
            return
    try:
        newdef()
        sys.exit(restart_with_reloader())
    except KeyboardInterrupt:
        pass

def newdef():

    pass



#def main():
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
    mainWindow.w.actionNodes.triggered.connect(mainWindow.show_nodes)
    mainWindow.w.sampler.w.scale.valueChanged.connect(mainWindow.update_scaleNumber)
    mainWindow.w.sizer_count.w.gfpganSlider.valueChanged.connect(mainWindow.update_gfpganNumber)

    mainWindow.w.preview.w.zoomInButton.clicked.connect(mainWindow.zoom_IN)
    mainWindow.w.preview.w.zoomOutButton.clicked.connect(mainWindow.zoom_OUT)
    #mainWindow.timeline.timeline.start()
    #mainWindow.deforum_thread()


    sys.exit(app.exec())


#run_with_reloader(main())


