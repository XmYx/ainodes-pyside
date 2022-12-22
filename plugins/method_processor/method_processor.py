"""This is an example plugin file that you can use to build your own plugins for aiNodes.

Welcome to aiNodes. Please refer to PySide 6.4 documentation for UI functions.

Please also note the following features at your disposal by default:
DeforumSix
Hypernetworks
Singleton

at plugin loading time, the plugins initme function will be called automatically to make sure that
all defaults are set correctly, and that your new UI element is loaded, with its signals and slots connected.

Your plugin's parent is the MainWindow, and by default, it has a canvas loaded. You can access all of its functions,
such as addrect_atpos, and image_preview_func (make sure to set self.parent.image before doing so).

It is good to know, that if you are doing heavy lifting, you have to use its own QThreadPool, otherwise your gui freezes
while processing. To do so, just use the worker from backend.worker

        worker = Worker(self.parent.deforum_ui.run_deforum_six_txt2img)
        self.parent.threadpool.start(worker)

It is also worth mentioning, that ui should only be modified from the main thread, therefore when displaying an image,
set self.parent.image, then call self.parent.image_preview_signal, which will emit a signal to call
the image_preview_func from the main thread.
"""
import copy

import PySide6
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QListWidget, QDialog, QFormLayout, \
    QLineEdit, QHBoxLayout, QLabel, QMenu
from PySide6.QtCore import QObject, Signal, Slot, Qt
import types
import argparse

from backend.singleton import singleton
from backend.worker import Worker

gs = singleton


class aiNodesPlugin():
    def __init__(self, parent):
        self.parent = parent

    def initme(self):
        print("Initializing Method Processor")
        self.widget = MethodProcessorWidget(self.parent)
        self.widget.widget.show()


class MethodProcessorWidget():
    def __init__(self, parent):
        self.parent = parent
        self.widget = QWidget()
        #super().__init__(parent)
        self.methods = {}  # dictionary to store added methods
        self.parameters = {}  # dictionary to store parameters for each method

        # create layout
        layout = QVBoxLayout()

        # create combo box to select method
        self.method_combo_box = QComboBox()
        self.method_combo_box.addItem("example_method")
        self.method_combo_box.addItem("txt2img")
        self.method_combo_box.addItem("restart_loop")
        layout.addWidget(self.method_combo_box)

        # create button to add selected method to list
        self.add_button = QPushButton("Add Method")
        self.add_button.clicked.connect(self.add_method)
        layout.addWidget(self.add_button)

        # create list widget to display added methods
        self.method_list = QListWidget()
        #self.method_list.itemClicked.connect(self.show_parameter_widget)
        self.method_list.itemDoubleClicked.connect(self.delete_method)

        layout.addWidget(self.method_list)

        # create button to start processing methods
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.process_methods_thread)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)

        # set layout
        self.widget.setLayout(layout)

        self.method_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.method_list.customContextMenuRequested.connect(self.show_context_menu)
    @Slot()
    def show_parameter_widget(self, item):
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis
        params = self.parameters[(method_name, method_id)]

        # create parameter widget
        parameter_widget = QDialog(self.widget)
        parameter_widget.setWindowTitle(f"{method_name} ({method_id}) Parameters")

        # create horizontal layout to contain column layouts
        h_layout = QHBoxLayout()
        parameter_widget.setLayout(h_layout)

        # create column layouts
        column_layouts = []
        for i in range(15):
            column_layout = QVBoxLayout()
            h_layout.addLayout(column_layout)
            column_layouts.append(column_layout)

        self.line_edits = {}
        for i, param_name in enumerate(vars(params)):
            line_edit = QLineEdit(str(getattr(params, param_name)))
            self.line_edits[param_name] = line_edit
            column_layouts[i % 15].addWidget(QLabel(param_name))
            column_layouts[i % 15].addWidget(line_edit)

        # create OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(parameter_widget.accept)
        h_layout.addStretch()
        h_layout.addWidget(ok_button)

        # show widget and update parameters if OK is clicked
        # show widget and update parameters if OK is clicked
        if parameter_widget.exec_() == QDialog.Accepted:
            for param_name in vars(params):
                line_edit = self.line_edits[param_name]
                value = line_edit.text()
                if value.isnumeric():
                    value = int(value)
                elif "." in value and ":" not in value:
                    value = float(value)
                elif value == "True":
                    value = True
                elif value == "False":
                    value = False
                else:
                    value = value
                setattr(params, param_name, value)

    def show_context_menu(self, position):
        # get current item
        item = self.method_list.itemAt(position)
        if not item:
            return

        # create context menu
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        parameters_action = menu.addAction("Show Parameters")

        # show context menu and handle actions
        action = menu.exec_(self.method_list.mapToGlobal(position))
        if action == delete_action:
            self.delete_method(item)
        elif action == parameters_action:
            self.show_parameter_widget(item)
    @Slot()
    def delete_method(self, item):
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis
        del self.methods[method_name][method_id]
        del self.parameters[(method_name, method_id)]
        self.method_list.takeItem(self.method_list.row(item))
    @Slot()
    def add_method(self):
        # get selected method from combo box
        method_name = self.method_combo_box.currentText()
        if method_name in self.methods and self.methods[method_name]:
            method_id = max(self.methods[method_name]) + 1  # generate new id for this instance of the method
        else:
            method_id = 0  # first instance of this method

        # create SimpleNamespace object to store parameters for this method
        params = copy.deepcopy(self.parent.sessionparams.params)
        params.param1 = "test"
        params.param2 = "test"

        self.parameters[(method_name, method_id)] = params

        # add method to list
        if method_name not in self.methods:
            self.methods[method_name] = {}
        self.methods[method_name][method_id] = getattr(self, method_name)
        self.method_list.addItem(f"{method_name} ({method_id})")
    def process_methods_thread(self):
        self.stop = False
        worker = Worker(self.process_methods)
        self.parent.threadpool.start(worker)
    def stop_processing(self):
        self.stop = True
    @Slot()
    def process_methods(self, progress_callback=None):
        if self.stop == False:
            for method_name in self.methods:
                for method_id in self.methods[method_name]:
                    method = self.methods[method_name][method_id]
                    params = self.parameters[(method_name, method_id)]
                    method(params)

    # example method
    def example_method(self, params):
        print(params)
        print(params.param2)
    def txt2img(self, params):
        self.parent.deforum_ui.run_deforum_six_txt2img(params=params)

        print(params)
        print(params.param2)
    def restart_loop(self, params=None):
        self.process_methods()