import functools
import os

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from backend.singleton import singleton

gs = singleton
deep_signals = gs.Singleton()

class BrushSizeAction(QtWidgets.QWidgetAction):
    def __init__(self, size, parent=None):
        super().__init__(parent)
        self.size = size
        self.setCheckable(True)
        self.triggered.connect(self.brush_triggered)

    def brush_triggered(self):
        self.setChecked(True)
        size = self.size
        self.parent().brushChanged.emit(size)

    def createWidget(self, parent):
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtCore.Qt.black)
        ellipse = QtWidgets.QGraphicsEllipseItem(0, 0, self.size, self.size)
        ellipse.setBrush(brush)
        ellipse.setPen(QtGui.QPen(QtCore.Qt.black, 0.5))

        view = QtWidgets.QGraphicsView()
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(ellipse)
        view.setScene(scene)
        view.setFixedSize(60, self.size + 4)
        view.setRenderHint(QtGui.QPainter.Antialiasing)
        view.setAlignment(QtCore.Qt.AlignLeft)
        view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(view)

        label = QtWidgets.QLabel(f"{self.size} px")
        layout.addWidget(label)

        widget = QtWidgets.QWidget(parent)
        widget.setLayout(layout)
        return widget

class BrushMenu(QtWidgets.QMenu):
    brushChanged = QtCore.Signal(int)

    def __init__(self, parent=None, menu=None):
        super().__init__(None)
        self.parent = parent
        self.menu = menu
        self.setTitle("Brush Size")
        self.brush_actions = []
        self.current_brush_size = 22
        for size in range(2, 51, 5):
            brush_action = BrushSizeAction(size, self)
            self.addAction(brush_action)
            self.brush_actions.append(brush_action)
        self.brushChanged.connect(self.parent.set_brush_size)

        brush_size_action = self.menu.addAction("Brush Size")
        brush_size_action.setMenu(self)


class ModelMenu:

    def __init__(self, parent=None, menu=None):
        self.folder_path = gs.system.models_path
        self.deep_signals = gs.Singleton()
        self.parent = parent
        self.parent_menu = menu
        self.submenu = QMenu("Inpaint Model")
        self.parent_menu.addMenu(self.submenu)
        self.current_action = None
        self.file_actions = []
        self.add_actions()


    def add_actions(self):
        self.file_actions = []
        files = [f for f in os.listdir(self.folder_path) if "inpaint" in f and not '.yaml' in f]

        second_folder_path = os.path.join(self.folder_path, 'custom')
        files.extend([os.path.join('custom',f) for f in os.listdir(second_folder_path) if "inpaint" in f and not '.yaml' in f])

        for n, file in enumerate(files):
            file = file.replace('\\', '/')
            action = QAction(file, self.submenu)
            action.setCheckable(True)
            action.triggered.connect(functools.partial(self.handleFileSelected, action))
            self.file_actions.append(action)
            if self.current_action is None:
                if n == 0:
                    action.setChecked(True)
                    self.current_action = action
                    gs.selected_inpaint_model = action.text()
            self.submenu.addAction(action)


    def handleFileSelected(self, action):
        filename = action.text()
        if action != self.current_action and self.current_action is not None:
            self.current_action.setChecked(False)
        for action in self.file_actions:
            if action.text() == filename:
                action.setChecked(True)
                self.current_action = action
                gs.selected_inpaint_model = action.text()
                break
        self.parent.signals.selected_model_changed.emit(filename)

class SaveImage:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        self.deep_signals = gs.Singleton()
        action = self.parent_menu.addAction('Save Image as')
        action.triggered.connect(self.send_save_image_signal)

    def send_save_image_signal(self):
        self.parent.signals.save_image_triggered.emit()


class DoInpaint:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        self.deep_signals = gs.Singleton()
        action = self.parent_menu.addAction('Do inpaint')
        action.triggered.connect(self.send_do_inpaint_signal)

    def send_do_inpaint_signal(self):
        self.parent.signals.doInpaintTriggered.emit()


class InpaintMask:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Paint inpaint mask')
        action.triggered.connect(self.send_do_inpaint_signal)

    def send_do_inpaint_signal(self):
        self.parent.signals.paintInpaintMaskTriggered.emit()


class SelectRect:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Select Rect')
        action.triggered.connect(self.parent.parent.select_mode)


class DeleteRect:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Delete Rect')
        action.triggered.connect(self.parent.parent.delete_rect)


class DragCanvas:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Drag Canvas')
        action.triggered.connect(self.parent.parent.drag_mode)


class Outpaint:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Outpaint')
        action.triggered.connect(self.parent.parent.add_mode)

class InpaintCurrentFrame:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Inpaint current frame')
        action.triggered.connect(self.parent.parent.inpaint_current_frame)

class MoveRect:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Move rect')
        action.triggered.connect(self.parent.parent.move_mode)


class ResetCanvas:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Reset Canvas')
        action.triggered.connect(self.signal_reset_canvas)

    @Slot()
    def signal_reset_canvas(self):
        self.parent.parent.reset()
        #self.parent.signals.reset_canvas.emit()

class SaveAsJson:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Save canvas as json')
        action.triggered.connect(self.parent.parent.save_rects_as_json)

class SaveAsPng:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Save canvas as PNG')
        action.triggered.connect(self.parent.parent.save_canvas)

class LoadFromJson:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Load from json')
        action.triggered.connect(self.parent.parent.load_rects_from_json)

class LoadImage:
    def __init__(self, parent=None, menu=None):
        self.parent = parent
        self.parent_menu = menu
        action = self.parent_menu.addAction('Load Image')
        action.triggered.connect(self.parent.parent.load_img_into_rect)


