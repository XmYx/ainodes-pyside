
from PySide6 import QtUiTools

from PySide6.QtWidgets import QApplication, QGraphicsView
from PySide6.QtWidgets import *
from PySide6.QtGui import *


import sys, os
app = QApplication(sys.argv)
pixmap = QPixmap('frontend/main/splash_2.png')
splash = QSplashScreen(pixmap)
splash.show()

icon = QIcon('frontend/main/splash_2.png')


if (os.name == 'nt'):
    #This is needed to display the app icon on the taskbar on Windows 7
    import ctypes
    myappid = 'aiNodes' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


# Create the tray
tray = QSystemTrayIcon()
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

#from PyQt6 import QtCore as qtc
#from PyQt6 import QtWidgets as qtw
#from PyQt6 import uic
#from PyQt6.Qt import *
from PySide6.QtCore import *
from PySide6 import QtCore
from PySide6.QtGui import QIcon, QPixmap
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertTokenizerFast
import warnings, random, traceback, time


from ldm.generate import Generate
from ui_classes import *
from backend.ui_func import getLatestGeneratedImagesFromPath



#Node Editor Functions - We have to make it a QWidget because its now a MainWindow object, which can only be created in a QApplication, which we already have.
#from nodeeditor.utils import loadStylesheet
#from nodeeditor.node_editor_window import NodeEditorWindow
#from frontend.example_calculator.calc_window import CalculatorWindow
#from qtpy.QtWidgets import QApplication as qapp


from PySide6.QtGui import QIcon, QKeySequence, QAction
from PySide6.QtWidgets import QMdiArea, QWidget, QDockWidget, QMessageBox, QFileDialog
from PySide6.QtCore import Qt, QSignalMapper

from nodeeditor.node_editor_window import NodeEditorWindow

from nodeeditor.node_editor_window import NodeEditorWindow
from frontend.example_calculator.calc_sub_window import CalculatorSubWindow
from frontend.example_calculator.calc_drag_listbox import QDMDragListbox
from nodeeditor.utils import dumpException, pp
from frontend.example_calculator.calc_conf import CALC_NODES

# Enabling edge validators
from nodeeditor.node_edge import Edge
from nodeeditor.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node
)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)


#from nodeeditor.utils import loadStylesheets
#from nodeeditor.node_editor_window import NodeEditorWindow


from backend.singleton import singleton

import backend.settings as settings
settings.load_settings_json()

gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

gs = singleton

gs.result = ""

gs.album = getLatestGeneratedImagesFromPath()

def prepare_loading():
    transformers.logging.set_verbosity_error()

    # this will preload the Bert tokenizer fles
    print('preloading bert tokenizer...')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print('...success')

    # this will download requirements for Kornia
    print('preloading Kornia requirements (ignore the deprecation warnings)...')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import kornia
    print('...success')

    version = 'openai/clip-vit-large-patch14'

    print('preloading CLIP model (Ignore the deprecation warnings)...')
    sys.stdout.flush()
    load_upscalers()
    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version)
    print('\n\n...success')

    # In the event that the user has installed GFPGAN and also elected to use
    # RealESRGAN, this will attempt to download the model needed by RealESRGANer


def load_upscalers():
    gfpgan = False
    try:
        from realesrgan import RealESRGANer

        gfpgan = True
    except ModuleNotFoundError:
        pass

    if gfpgan:
        print('Loading models from RealESRGAN and facexlib')
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper

            RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                ),
            )

            RealESRGANer(
                scale=4,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                model=RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                ),
            )

            FaceRestoreHelper(1, det_model='retinaface_resnet50')
            print('...success')
        except Exception:
            import traceback

            print('Error loading GFPGAN:')
            print(traceback.format_exc())


import platform

if "macOS" in platform.platform():
    gs.platform = "macOS"
    #prepare_loading()

load_upscalers()

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class NodeWindow(NodeEditorWindow):

    def initUI(self):
        self.name_company = 'Blenderfreak'
        self.name_product = 'Calculator NodeEditor'

        self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "qss/nodeeditor.qss")
        """loadStylesheets(
            os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )"""

        self.empty_icon = QIcon(".")

        """if DEBUG:
            print("Registered nodes:")
            pp(CALC_NODES)"""


        self.mdiArea = QMdiArea()
        self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setViewMode(QMdiArea.TabbedView)
        self.mdiArea.setDocumentMode(True)
        self.mdiArea.setTabsClosable(True)
        self.mdiArea.setTabsMovable(True)
        self.setCentralWidget(self.mdiArea)

        self.mdiArea.subWindowActivated.connect(self.updateMenus)
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mappedObject.connect(self.setActiveSubWindow)

        self.createNodesDock()

        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.updateMenus()

        self.readSettings()

        self.setWindowTitle("Calculator NodeEditor Example")

    def closeEvent(self, event):
        self.mdiArea.closeAllSubWindows()
        if self.mdiArea.currentSubWindow():
            event.ignore()
        else:
            self.writeSettings()
            event.accept()
            # hacky fix for PyQt 5.14.x
            import sys
            sys.exit(0)


    def createActions(self):
        super().createActions()

        self.actClose = QAction("Cl&ose", self, statusTip="Close the active window", triggered=self.mdiArea.closeActiveSubWindow)
        self.actCloseAll = QAction("Close &All", self, statusTip="Close all the windows", triggered=self.mdiArea.closeAllSubWindows)
        self.actTile = QAction("&Tile", self, statusTip="Tile the windows", triggered=self.mdiArea.tileSubWindows)
        self.actCascade = QAction("&Cascade", self, statusTip="Cascade the windows", triggered=self.mdiArea.cascadeSubWindows)
        self.actNext = QAction("Ne&xt", self, shortcut=QKeySequence.NextChild, statusTip="Move the focus to the next window", triggered=self.mdiArea.activateNextSubWindow)
        self.actPrevious = QAction("Pre&vious", self, shortcut=QKeySequence.PreviousChild, statusTip="Move the focus to the previous window", triggered=self.mdiArea.activatePreviousSubWindow)

        self.actSeparator = QAction(self)
        self.actSeparator.setSeparator(True)

        self.actAbout = QAction("&About", self, statusTip="Show the application's About box", triggered=self.about)

    def getCurrentNodeEditorWidget(self):
        """ we're returning NodeEditorWidget here... """
        activeSubWindow = self.mdiArea.activeSubWindow()
        if activeSubWindow:
            return activeSubWindow.widget()
        return None

    def onFileNew(self):
        try:
            subwnd = self.createMdiChild()
            subwnd.widget().fileNew()
            subwnd.show()
        except Exception as e: dumpException(e)


    def onFileOpen(self):
        fnames, filter = QFileDialog.getOpenFileNames(self, 'Open graph from file', self.getFileDialogDirectory(), self.getFileDialogFilter())

        try:
            for fname in fnames:
                if fname:
                    existing = self.findMdiChild(fname)
                    if existing:
                        self.mdiArea.setActiveSubWindow(existing)
                    else:
                        # we need to create new subWindow and open the file
                        nodeeditor = CalculatorSubWindow()
                        if nodeeditor.fileLoad(fname):
                            self.statusBar().showMessage("File %s loaded" % fname, 5000)
                            nodeeditor.setTitle()
                            subwnd = self.createMdiChild(nodeeditor)
                            subwnd.show()
                        else:
                            nodeeditor.close()
        except Exception as e: dumpException(e)


    def about(self):
        QMessageBox.about(self, "About Calculator NodeEditor Example",
                          "The <b>Calculator NodeEditor</b> example demonstrates how to write multiple "
                          "document interface applications using PyQt5 and NodeEditor. For more information visit: "
                          "<a href='https://www.blenderfreak.com/'>www.BlenderFreak.com</a>")

    def createMenus(self):
        super().createMenus()

        self.windowMenu = self.menuBar().addMenu("&Window")
        self.updateWindowMenu()
        self.windowMenu.aboutToShow.connect(self.updateWindowMenu)

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.actAbout)

        self.editMenu.aboutToShow.connect(self.updateEditMenu)

    def updateMenus(self):
        # print("update Menus")
        active = self.getCurrentNodeEditorWidget()
        hasMdiChild = (active is not None)

        self.actSave.setEnabled(hasMdiChild)
        self.actSaveAs.setEnabled(hasMdiChild)
        self.actClose.setEnabled(hasMdiChild)
        self.actCloseAll.setEnabled(hasMdiChild)
        self.actTile.setEnabled(hasMdiChild)
        self.actCascade.setEnabled(hasMdiChild)
        self.actNext.setEnabled(hasMdiChild)
        self.actPrevious.setEnabled(hasMdiChild)
        self.actSeparator.setVisible(hasMdiChild)

        self.updateEditMenu()

    def updateEditMenu(self):
        try:
            # print("update Edit Menu")
            active = self.getCurrentNodeEditorWidget()
            hasMdiChild = (active is not None)

            self.actPaste.setEnabled(hasMdiChild)

            self.actCut.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actCopy.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actDelete.setEnabled(hasMdiChild and active.hasSelectedItems())

            self.actUndo.setEnabled(hasMdiChild and active.canUndo())
            self.actRedo.setEnabled(hasMdiChild and active.canRedo())
        except Exception as e: dumpException(e)



    def updateWindowMenu(self):
        self.windowMenu.clear()

        toolbar_nodes = self.windowMenu.addAction("Nodes Toolbar")
        toolbar_nodes.setCheckable(True)
        toolbar_nodes.triggered.connect(self.onWindowNodesToolbar)
        toolbar_nodes.setChecked(self.nodesDock.isVisible())

        self.windowMenu.addSeparator()

        self.windowMenu.addAction(self.actClose)
        self.windowMenu.addAction(self.actCloseAll)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actTile)
        self.windowMenu.addAction(self.actCascade)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actNext)
        self.windowMenu.addAction(self.actPrevious)
        self.windowMenu.addAction(self.actSeparator)

        windows = self.mdiArea.subWindowList()
        self.actSeparator.setVisible(len(windows) != 0)

        for i, window in enumerate(windows):
            child = window.widget()

            text = "%d %s" % (i + 1, child.getUserFriendlyFilename())
            if i < 9:
                text = '&' + text

            action = self.windowMenu.addAction(text)
            action.setCheckable(True)
            action.setChecked(child is self.getCurrentNodeEditorWidget())
            action.triggered.connect(self.windowMapper.map)
            self.windowMapper.setMapping(action, window)

    def onWindowNodesToolbar(self):
        if self.nodesDock.isVisible():
            self.nodesDock.hide()
        else:
            self.nodesDock.show()

    def createToolBars(self):
        pass

    def createNodesDock(self):
        self.nodesListWidget = QDMDragListbox()

        self.nodesDock = QDockWidget("Nodes")
        self.nodesDock.setWidget(self.nodesListWidget)
        self.nodesDock.setFloating(False)

        self.addDockWidget(Qt.RightDockWidgetArea, self.nodesDock)

    def createStatusBar(self):
        self.statusBar().showMessage("Ready")

    def createMdiChild(self, child_widget=None):
        nodeeditor = child_widget if child_widget is not None else CalculatorSubWindow()
        subwnd = self.mdiArea.addSubWindow(nodeeditor)
        subwnd.setWindowIcon(self.empty_icon)
        # nodeeditor.scene.addItemSelectedListener(self.updateEditMenu)
        # nodeeditor.scene.addItemsDeselectedListener(self.updateEditMenu)
        nodeeditor.scene.history.addHistoryModifiedListener(self.updateEditMenu)
        nodeeditor.addCloseEventListener(self.onSubWndClose)
        return subwnd

    def onSubWndClose(self, widget, event):
        existing = self.findMdiChild(widget.filename)
        self.mdiArea.setActiveSubWindow(existing)

        if self.maybeSave():
            event.accept()
        else:
            event.ignore()


    def findMdiChild(self, filename):
        for window in self.mdiArea.subWindowList():
            if window.widget().filename == filename:
                return window
        return None


    def setActiveSubWindow(self, window):
        if window:
            self.mdiArea.setActiveSubWindow(window)
class GenerateWindow(QWidget):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/main/main_window.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



        self.image_path = ""

        #print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        #uic.loadUi("frontend/main/main_window.ui", self)



        self.home()
        self.nodeWindow = NodeWindow()
        self.load_history()
        #self.show_anim()

        self.w.actionAnim.triggered.connect(self.show_anim)
        self.w.actionPreview.triggered.connect(self.show_preview)
        self.w.actionPrompt.triggered.connect(self.show_prompt)
        #self.actionRunControl.triggered.connect(self.show_runner)
        self.w.actionSampler.triggered.connect(self.show_sampler)
        self.w.actionSliders.triggered.connect(self.show_sizer_count)
        self.w.actionThumbnails.triggered.connect(self.show_thumbnails)

        #self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        #self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        #self.pix_map_item = self.preview.scene.addPixmap(self.pix_map)
        """self.global_factor = 1
        self.pix_map_item = QGraphicsPixmapItem()

    def scaleImage(self, factor):
        _pixmap = self.pic.scaledToHeight(int(factor*self.viewport().geometry().height()), Qt.SmoothTransformation)
        self.pix_map_item.setPixmap(_pixmap)
        self.preview.scene.setSceneRect(QRectF(_pixmap.rect()))

    def wheelEvent(self, event):
        factor = 1.5

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            self.centerOn(scene_pos)

            if event.angleDelta().y() > 0 and self.global_factor < 20:
                self.global_factor *= factor
                self.scaleImage(self.global_factor)
            elif event.angleDelta().y() < 0 and self.global_factor > 0.2:
                self.global_factor /= factor
                self.scaleImage(self.global_factor)
        else:
            return super().wheelEvent(event)"""

    def home(self):

        self.w.preview = Preview()
        self.w.sizer_count = SizerCount()
        self.w.sampler = Sampler()
        #self.runner = Runner()
        self.w.anim = Anim()
        self.w.prompt = Prompt()

        self.w.thumbnails = Thumbnails()

        #app2  = qapp(sys.argv)
        #self.nodes = NodeEditorWindow()
        #self.nodes.nodeeditor.addNodes()

        #wnd.show()

        self.w.thumbnails.w.thumbs.itemClicked.connect(self.viewImageClicked)
        #self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon('frontend/main/splash.png'), "Earth"))



        self.w.sizer_count.w.heightNumber.display(str(self.w.sizer_count.w.heightSlider.value()))
        self.w.sizer_count.w.widthNumber.display(str(self.w.sizer_count.w.widthSlider.value()))
        self.w.sizer_count.w.samplesNumber.display(str(self.w.sizer_count.w.samplesSlider.value()))
        self.w.sizer_count.w.batchSizeNumber.display(str(self.w.sizer_count.w.batchSizeSlider.value()))
        self.w.sizer_count.w.stepsNumber.display(str(self.w.sizer_count.w.stepsSlider.value()))
        self.w.sizer_count.w.scaleNumber.display(str(self.w.sizer_count.w.scaleSlider.value()))



        self.w.setCentralWidget(self.w.preview.w)

        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sampler.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sizer_count.w.dockWidget)

        #self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.runner)
        #self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.anim.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.w.prompt.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.thumbnails.w.dockWidget)

        #self.resizeDocks({self.thumbnails}, {100}, QtWidgets.Horizontal);

        self.w.preview.w.scene = QGraphicsScene()
        self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)

        self.w.thumbnails.w.thumbsZoom.valueChanged.connect(self.updateThumbsZoom)
        self.w.thumbnails.w.refresh.clicked.connect(self.load_history)





    def updateThumbsZoom(self):
        size = self.w.thumbnails.w.thumbsZoom.value()
        self.w.thumbnails.w.thumbs.setGridSize(QSize(size, size))
        self.w.thumbnails.w.thumbs.setIconSize(QSize(size, size))
    def update_scaleNumber(self):
        float = self.w.sizer_count.w.scaleSlider.value() / 100
        self.w.sizer_count.w.scaleNumber.display(str(float))
    def update_gfpganNumber(self):
        float = self.w.sizer_count.w.gfpganSlider.value() / 10
        self.w.sizer_count.w.gfpganNumber.display(str(float))


    def show_anim(self):
        self.w.anim.w.show()
    def show_preview(self):
        self.w.preview.w.show()
    def show_prompt(self):
        self.w.prompt.w.show()
    #def show_runner(self):
        #self.runner.show()
    def show_sampler(self):
        self.w.sampler.w.show()
    def show_sizer_count(self):
        self.w.sizer_count.w.show()
    def show_thumbnails(self):
        self.w.thumbnails.w.show()

    def load_history(self):
        self.w.thumbnails.w.thumbs.clear()
        for image in gs.album:
            self.w.thumbnails.w.thumbs.addItem(QListWidgetItem(QIcon(image), str(image)))

    def viewImageClicked(self, item):
        #self.preview.setPixmap(item.image())
        #pic = QImage(item[0])
        imageSize = item.icon().actualSize(QSize(512, 512))

        self.w.preview.pic = QGraphicsPixmapItem()
        self.w.preview.pic.setPixmap(item.icon().pixmap(imageSize))

        self.w.preview.w.scene.clear()

        self.w.preview.w.scene.addItem(self.w.preview.pic)
        self.w.preview.w.graphicsView.fitInView(self.w.preview.pic, Qt.AspectRatioMode.KeepAspectRatio)
        #self.preview.graphicsView.setDragMode(QGraphicsView.RubberBandDrag)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        #self.preview.graphicsView.setPhoto(item.icon().pixmap(imageSize))

    def run_txt2img(self, progress_callback):

        width=self.w.sizer_count.w.widthSlider.value()
        height=self.w.sizer_count.w.heightSlider.value()
        scale=self.w.sizer_count.w.scaleSlider.value()
        steps=self.w.sizer_count.w.stepsSlider.value()
        samples=self.w.sizer_count.w.samplesSlider.value()
        batchsize=self.w.sizer_count.w.batchSizeSlider.value()
        seamless=self.w.sampler.w.seamless.isChecked()
        full_precision=self.w.sampler.w.fullPrecision.isChecked()
        sampler=self.w.sampler.w.comboBox.currentText()
        upscale=[self.w.sizer_count.w.upscaleSlider.value()]
        gfpgan_strength=self.w.sizer_count.w.gfpganSlider.value() / 100

        if self.w.sampler.w.seedEdit.text() != '':
            seed=int(self.w.sampler.w.seedEdit.text())
        else:
            seed=''


        if gs.defaults.general.default_path_mode == "subfolders":
            outdir = gs.defaults.general.outdir
        else:
            outdir = f'{gs.defaults.general.outdir}/_batch_images'




        """The full list of arguments to Generate() are:
        gr = Generate(
                  weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
                  config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
                  iterations  = <integer>     // how many times to run the sampling (1)
                  steps       = <integer>     // 50
                  seed        = <integer>     // current system time
                  sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
                  grid        = <boolean>     // false
                  width       = <integer>     // image width, multiple of 64 (512)
                  height      = <integer>     // image height, multiple of 64 (512)
                  cfg_scale   = <float>       // condition-free guidance scale (7.5)
                  )

"""

        all_images = []
        for i in range(batchsize):
            print(f"Full Precision {full_precision}")

            results = gr.prompt2image(prompt   = self.w.prompt.w.textEdit.toPlainText(),
                                      outdir   = outdir,
                                      cfg_scale = scale,
                                      width  = width,
                                      height = height,
                                      iterations = samples,
                                      steps = steps,
                                      seamless = seamless,
                                      sampler_name = sampler,
                                      seed = seed,
                                      upscale = upscale,
                                      gfpgan_strength = gfpgan_strength,
                                      strength = 0.0,
                                      full_precision = full_precision)
            for row in results:
                print(f'filename={row[0]}')
                print(f'seed    ={row[1]}')
                filename = random.randint(10000, 99999)
                output = f'outputs/{filename}.png'
                row[0].save(output)
                self.image_path = output
                self.get_pic()
                self.w.thumbnails.w.thumbs.addItem(QListWidgetItem(QIcon(output), str(self.w.prompt.w.textEdit.toPlainText())))
            #all_images.append(results)

            #return all_images






    def progressbar(self, progress_callback):
        a = 0
        b = 100
        while b > 0:
            a = a + 1
            self.preview.progressBar.setValue(int(a))
            time.sleep(1)
            if b == 1:
                b = 100
    def progress_thread(self):
        worker2 = Worker(self.progressbar)
        #worker.signals.result.connect(self.get_pic)
        # Execute
        threadpool.start(worker2)

    def txt2img_thread(self):
        # Pass the function to execute
        worker = Worker(self.run_txt2img)
        worker.signals.result.connect(self.get_pic)
        # Execute
        threadpool.start(worker)

        #progress bar test:
        #self.progress_thread()
    def get_pic(self): #from self.image_path
        print("ok")
        image_qt = QImage(self.image_path)

        self.w.preview.pic = QGraphicsPixmapItem()
        self.w.preview.pic.setPixmap(QPixmap.fromImage(image_qt))

        self.w.preview.w.scene.clear()
        self.w.preview.w.scene.addItem(self.w.preview.pic)

        self.w.preview.w.graphicsView.fitInView(self.w.preview.pic, Qt.AspectRatioMode.KeepAspectRatio)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
    def zoom_IN(self):
        self.w.preview.w.graphicsView.scale(1.25, 1.25)
    def zoom_OUT(self):
        self.w.preview.w.graphicsView.scale(0.75, 0.75)

"""class CalculatorWin(CalculatorWindow):
    def __init__(self, *args, **kwargs):




        app2 = qapp(sys.argv)
        nodes = CalculatorWindow()
        nodes.show()
        app2.exec()"""
"""def show_nodes():
    #in main thread:
    CalculatorWin()

    #in a separate thread
    #worker = Worker(CalculatorWin) # Any other args, kwargs are passed to the run function
    # Execute
    #threadpool.start(worker)"""

if __name__ == "__main__":

    mainWindow = GenerateWindow()
    threadpool = QThreadPool()
    mainWindow.w.setWindowTitle("aiNodes")
    mainWindow.w.setWindowIcon(QIcon('frontend/main/splash_2.png'))

    #app.setIcon(QIcon('frontend/main/splash.png'))
    mainWindow.w.show()
    #mainWindow.nodeWindow.show()
    mainWindow.w.resize(1280, 720)
    splash.finish(mainWindow)
    #mainWindow.progress_thread()

    #mainWindow.thumbnails.setGeometry(680,0,800,600)

    mainWindow.w.prompt.w.runButton.clicked.connect(mainWindow.txt2img_thread)
    #mainWindow.runner.runButton.clicked.connect(mainWindow.progress_thread)

    mainWindow.w.actionNodes.triggered.connect(mainWindow.nodeWindow.show)
    mainWindow.w.sizer_count.w.scaleSlider.valueChanged.connect(mainWindow.update_scaleNumber)
    mainWindow.w.sizer_count.w.gfpganSlider.valueChanged.connect(mainWindow.update_gfpganNumber)

    mainWindow.w.preview.w.zoomInButton.clicked.connect(mainWindow.zoom_IN)
    mainWindow.w.preview.w.zoomOutButton.clicked.connect(mainWindow.zoom_OUT)

    sys.exit(app.exec())
