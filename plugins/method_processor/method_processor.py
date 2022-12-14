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
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QListWidget, QDialog, QFormLayout, \
    QLineEdit, QHBoxLayout, QLabel, QMenu, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QAbstractItemView, \
    QDialogButtonBox
from PySide6.QtCore import QObject, Signal, Slot, Qt
import types
import argparse

from backend.singleton import singleton
from backend.worker import Worker
from frontend.session_params import translate_sampler

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
        self.method_combo_box.addItem("txt2img")
        self.method_combo_box.addItem("switch_model")
        self.method_combo_box.addItem("restart_loop")
        self.method_combo_box.addItem("use_prev_as_init")
        self.method_combo_box.addItem("param_editor")
        self.method_combo_box.addItem("space_engineer_example")
        layout.addWidget(self.method_combo_box)

        # create button to add selected method to list
        self.add_button = QPushButton("Add Method")
        self.add_button.clicked.connect(self.add_method)
        layout.addWidget(self.add_button)

        # create list widget to display added methods
        self.method_list = QListWidget()
        #self.method_list.itemClicked.connect(self.show_parameter_widget)
        self.method_list.itemClicked.connect(self.item_clicked)
        self.method_list.itemDoubleClicked.connect(self.delete_method)

        layout.addWidget(self.method_list)

        # create button to start processing methods
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.process_methods_thread)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.play_button)
        layout.addWidget(self.stop_button)
        self.param_modifier = None
        # set layout
        self.widget.setLayout(layout)
        self.method_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.method_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.method_list.customContextMenuRequested.connect(self.show_context_menu)

    def item_clicked(self, item):

        #runs the function names [method_name]_ui_func on click
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis
        func_name = method_name + "_ui_func"
        getattr(self, func_name)(item)
    @Slot()
    def show_parameter_widget(self, item):

        #shows the full parameter widget on right click show parameters
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
        for i in range(20):
            column_layout = QVBoxLayout()
            column_layout.setContentsMargins(3,5,5,2)
            column_layout.setSpacing(3)
            h_layout.addLayout(column_layout)
            column_layouts.append(column_layout)

        self.line_edits = {}

        widget_types_and_values = self.get_widget_types(params=params, current_widget=self.parent.widgets[self.parent.current_widget].w)

        widget_types_and_values = sorted(widget_types_and_values,
                                         key=lambda x: (list(x.keys())[0][1], list(x.keys())[0][0]))
        print(widget_types_and_values)
        self.line_edits = {}
        for i, widget_type_and_value in enumerate(widget_types_and_values):
            for name, widget_type in widget_type_and_value.keys():
                value = widget_type_and_value[(name, widget_type)]
                if widget_type == 'QSpinBox':
                    widget = QSpinBox()
                    widget.setMaximum(4096)
                    widget.setValue(value)
                if widget_type == 'QDoubleSpinBox':
                    widget = QDoubleSpinBox()
                    widget.setMaximum(4096.0)
                    widget.setValue(value)
                elif widget_type == 'QLineEdit':
                    widget = QLineEdit()
                    widget.setText(str(value))
                elif widget_type == 'QTextEdit':
                    widget = QTextEdit()
                    widget.setText(str(value))
                elif widget_type == 'QCheckBox':
                    widget = QCheckBox()
                    widget.setChecked(value)
                elif widget_type == 'QComboBox':
                    widget = QComboBox()
                    widget.addItems(value)
                    widget.setCurrentText(str(params.__dict__[name]))
                self.line_edits[name] = widget
                column_layouts[i % 15].addWidget(QLabel(name))
                column_layouts[i % 15].addWidget(widget)


        #input_widgets_dict = self.create_input_widgets_dict()
        #for i, param_name in enumerate(vars(params)):
        #    value = str(getattr(params, param_name))
        #    widget_type = input_widgets_dict.get(param_name, QLineEdit)
        #    input_widget = widget_type(value)
        #    self.line_edits[param_name] = input_widget
        #    column_layouts[i % 15].addWidget(QLabel(param_name))
        #    column_layouts[i % 15].addWidget(input_widget)
        # create OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(parameter_widget.accept)
        h_layout.addStretch()
        h_layout.addWidget(ok_button)

        # show widget and update parameters if OK is clicked
        # show widget and update parameters if OK is clicked
        if parameter_widget.exec() == QDialog.Accepted:
            # update params with new values
            for widget_type_and_value in widget_types_and_values:
                for name, widget_type in widget_type_and_value.keys():
                    widget = self.line_edits[name]
                    value = widget_type_and_value[(name, widget_type)]
                    if widget_type == 'QSpinBox' or widget_type == 'QDoubleSpinBox':
                        value = widget.value()
                    elif widget_type == 'QLineEdit':
                        value = widget.text()
                    elif widget_type == 'QTextEdit':
                        value = widget.toPlainText()
                    elif widget_type == 'QCheckBox':
                        value = widget.isChecked()
                    elif widget_type == 'QComboBox':
                        value = widget.currentText()
                    string = False
                    #if any(c.isalpha() for c in value) or "(" in value:
                    #    value = str(value)
                    #    string = True
                    #if value.isnumeric():
                    #    value = int(value)
                    #elif "." in value and string == False:
                    #    value = float(value)
                    #elif value == "True":
                    #    value = True
                    #elif value == "False":
                    #    value = False
                    # check for non-numeric characters
                    #else:
                    #    value = value
                    setattr(params, name, value)
    def show_context_menu(self, position):
        # get current item
        item = self.method_list.itemAt(position)
        if not item:
            return

        # create context menu
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        parameters_action = menu.addAction("Show Parameters")
        # add duplicate in place action
        duplicate_in_place_action = QAction("Duplicate in place", self.widget)
        duplicate_in_place_action.triggered.connect(lambda: self.duplicate_in_place(item))
        menu.addAction(duplicate_in_place_action)

        # add duplicate to end action
        duplicate_to_end_action = QAction("Duplicate to end", self.widget)
        #duplicate_to_end_action.triggered.connect(lambda: self.duplicate_to_end(item))
        menu.addAction(duplicate_to_end_action)

        # add copy action
        copy_action = QAction("Copy", self.widget)
        #copy_action.triggered.connect(lambda: self.copy(item))
        menu.addAction(copy_action)

        # add paste action
        paste_action = QAction("Paste", self.widget)
        #paste_action.triggered.connect(lambda: self.paste())
        menu.addAction(paste_action)

        # show context menu and handle actions
        action = menu.exec(self.method_list.mapToGlobal(position))
        if action == delete_action:
            self.delete_method(item)
        elif action == parameters_action:
            self.show_parameter_widget(item)
        elif action == duplicate_in_place_action:
            self.duplicate_in_place(item)
        elif action == duplicate_to_end_action:
            self.duplicate_to_end(item)
        elif action == copy_action:
            self.copy(item)
        elif action == paste_action:
            self.paste()



    def duplicate_in_place(self, item):
        # get method name and id
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis

        # create new id for duplicate
        new_id = max(self.methods[method_name].keys()) + 1

        # add duplicate method
        self.methods[method_name][new_id] = self.methods[method_name][method_id]
        self.parameters[(method_name, new_id)] = self.parameters[(method_name, method_id)]

        # add duplicate item to list
        item_text = f"{method_name} ({new_id})"
        self.method_list.insertItem(self.method_list.row(item), item_text)

    def duplicate_to_end(self, item):
        # get method name and id
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis

        # create new id for duplicate
        new_id = max(self.methods[method_name].keys()) + 1

        # add duplicate method
        self.methods[method_name][new_id] = self.methods[method_name][method_id]
        self.parameters[(method_name, new_id)] = self.parameters[(method_name, method_id)]

        # add duplicate item to list
        item_text = f"{method_name} ({new_id})"
        self.method_list.addItem(item_text)

    def copy(self, item):
        # get method name and id
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis

        # store copied method and id
        self.copied_method = (method_name, method_id)
    def paste(self):
        if self.copied_method:
            # get method name and id
            method_name, method_id = self.copied_method[0], self.copied_method[1]

            # create new id for pasted method
            new_id = max(self.methods[method_name].keys()) + 1

            # paste method and parameters
            self.methods[method_name][new_id] = self.methods[method_name][method_id]
            self.parameters[(method_name, new_id)] = self.parameters[(method_name, method_id)]

            # add pasted item to list
            item_text = f"{method_name} ({new_id})"
            self.method_list.addItem(item_text)

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
        print(method_name)
        if method_name == "txt2img":
            params = self.parent.sessionparams.update_params()
        elif method_name == "switch_model":
            params = {}
            params = types.SimpleNamespace(**params)
        elif method_name == "use_prev_as_init":
            params = self.add_use_prev_as_init_params()
        elif method_name in ["restart_loop", "param_editor"]:
            params = {}
            params = types.SimpleNamespace(**params)
            params.active = True
        else:
            params = getattr(self, f"get_{method_name}_params")()

            print(params.W)
        self.parameters[(method_name, method_id)] = params

        # add method to list
        if method_name not in self.methods:
            self.methods[method_name] = {}
        self.methods[method_name][method_id] = getattr(self, method_name)
        self.method_list.addItem(f"{method_name} ({method_id})")
        #self.method_list.item(self.method_list.count() - 1).clicked.connect(getattr(self, f"{method_name}_ui_func"))
    def process_methods_thread(self):
        self.stop = False
        worker = Worker(self.process_methods)
        self.parent.threadpool.start(worker)
    def stop_processing(self):
        self.stop = True
    @Slot()
    def process_methods(self, progress_callback=None, retrigger=False):
        # get list of method items in method_list
        if retrigger == False:
            self.param_modifier = None
        method_items = [self.method_list.item(i) for i in range(self.method_list.count())]

        # loop through method items
        for item in method_items:
            if self.stop:
                break
            # get method name and id
            method_name, method_id = item.text().split(" (")
            method_id = int(method_id[:-1])  # remove closing parenthesis

            # get method and parameters
            method = self.methods[method_name][method_id]
            params = self.parameters[(method_name, method_id)]

            # call method with parameters
            getattr(self, method.__name__)(params)

            # update progress
            #if progress_callback:
            #    progress_callback()
    def create_input_widgets_dict(self):
        # get input widgets from parent widget
        #print(self.parent.widgets[self.parent.current_widget].ui_unicontrol.unicontrol.w.findChildren(QWidget))

        input_widgets = [
            widget for widget in self.parent.widgets[self.parent.current_widget].w.findChildren(QWidget)

        ]
        #print(input_widgets)
        # create dictionary of input widgets and values for QComboBox widgets
        input_widgets_dict = {}
        for widget in input_widgets:
            input_widgets_dict[widget.objectName()] = type(widget)
        #print(input_widgets_dict)
        return input_widgets_dict

    def get_combo_box_values(self, name):
        # find QComboBox widget with given object name
        combo_box = self.parent.unicontrol.findChild(QComboBox, name)

        # return values if widget was found, otherwise return empty list
        if combo_box:
            return combo_box.items()
        return []

    def get_widget_types(self, params, current_widget):
        #current_widget = self.parent.widgets[self.parent.current_widget].w
        results = []
        combo_box_items = []
        for key, value in params.__dict__.items():
            try:
                type_str = str(getattr(current_widget, key))
                #print(type_str)
                if 'QSpinBox' in type_str:
                    #result = {(key, 'QSpinBox'):getattr(current_widget, key).value()}
                    result = {(key, 'QSpinBox'):value}
                elif 'QDoubleSpinBox' in type_str:
                    #result = {(key, 'QDoubleSpinBox'):getattr(current_widget, key).value()}
                    result = {(key, 'QDoubleSpinBox'):value}
                elif 'QLineEdit' in type_str:
                    #result = {(key, 'QLineEdit'):getattr(current_widget, key).text()}
                    result = {(key, 'QLineEdit'):value}
                elif 'QCheckBox' in type_str:
                    #result = {(key, 'QCheckBox'):getattr(current_widget, key).isChecked()}
                    result = {(key, 'QCheckBox'):value}
                elif 'QTextEdit' in type_str:
                    #result = {(key, 'QCheckBox'):getattr(current_widget, key).isChecked()}
                    result = {(key, 'QTextEdit'):value}
                elif 'QComboBox' in type_str:
                    items = [getattr(current_widget, key).itemText(i) for i in range(getattr(current_widget, key).count())]
                    result = {(key,'QComboBox'):items}
                results.append(result)
            except Exception as e:
                pass
        print(results)
        return results
    def txt2img(self, params):
        if self.param_modifier is not None:
            for key, value in self.param_modifier.__dict__.items():
                params.__dict__[key] = value
        params.sampler = translate_sampler(params.sampler)
        if type(params.grad_inject_timing) is not int:
            if params.grad_inject_timing.isnumeric():
                params.grad_inject_timing = int(params.grad_inject_timing)
            elif "," in params.grad_inject_timing:
                params.grad_inject_timing = params.grad_inject_timing.split(",")
            elif "." in params.grad_inject_timing:
                params.grad_inject_timing = float(params.grad_inject_timing)
            else:
                params.grad_inject_timing = params.grad_inject_timing
        self.parent.deforum_ui.run_deforum_six_txt2img(params=params)
    def txt2img_ui_func(self):
        pass
    def switch_model(self, params):
        from backend.shared import model_killer
        model_killer()
    def switch_model_ui_func(self):
        pass
    def restart_loop(self, params=None):
        if self.stop == True:
            return

        self.process_methods(retrigger=True)
    def restart_loop_ui_func(self):
        pass
    def use_prev_as_init(self, params):
        if self.param_modifier == None:
            self.param_modifier = types.SimpleNamespace()
        self.param_modifier.init_image = gs.temppath
        self.param_modifier.use_init = True
    def add_use_prev_as_init_params(self):
        params = types.SimpleNamespace()
        params.use_init = True
        params.init_img = ""
        return params
    def use_prev_as_init_ui_func(self):

        pass

    def param_editor(self):
        pass
    @Slot()
    def param_editor_ui_func(self, item):
        method_name, method_id = item.text().split(" (")
        method_id = int(method_id[:-1])  # remove closing parenthesis
        params = self.parent.sessionparams.params
        # create widget
        parameter_widget = QDialog(self.widget)
        parameter_widget.setWindowTitle(f"{method_name} ({method_id}) Parameters")
        self.paramlayout = QFormLayout()
        parameter_widget.setLayout(self.paramlayout)

        # create combo box to select parameter to edit
        self.param_combo_box = QComboBox()
        self.param_combo_box.addItems(vars(params))
        self.param_combo_box.currentTextChanged.connect(self.update_parameter_form)
        self.paramlayout.addRow("Parameter:", self.param_combo_box)

        # create form to edit parameter
        form = QFormLayout()
        self.paramlayout.addRow(form)

        # create ok and cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(parameter_widget.accept)
        button_box.rejected.connect(parameter_widget.reject)
        self.paramlayout.addWidget(button_box)

        # show widget and get result
        result = parameter_widget.exec_()
        if result == QDialog.Accepted:
            if self.param_modifier == None:
                self.param_modifier = types.SimpleNamespace()
            selected_param = self.param_combo_box.currentText()
            input_widget = self.param_input_widget
            if isinstance(input_widget, QLineEdit):
                setattr(self.param_modifier, selected_param, input_widget.text())
            elif isinstance(input_widget, QSpinBox):
                setattr(self.param_modifier, selected_param, input_widget.value())
            elif isinstance(input_widget, QDoubleSpinBox):
                setattr(self.param_modifier, selected_param, input_widget.value())
            elif isinstance(input_widget, QComboBox):
                setattr(self.param_modifier, selected_param, input_widget.currentText())
            elif isinstance(input_widget, QCheckBox):
                setattr(self.param_modifier, selected_param, input_widget.isChecked())
        print(self.param_modifier.__dict__[selected_param])

    def update_parameter_form(self):
        selected_param = self.param_combo_box.currentText()
        input_widget = getattr(self.parent.widgets["unicontrol"].w, selected_param)
        print(input_widget)
        type_str = str(getattr(self.parent.widgets["unicontrol"].w, selected_param))
        print(type_str)
        if 'QSpinBox' in type_str:
            self.param_input_widget = QSpinBox()
            self.param_input_widget.setMaximum(4096)
            self.param_input_widget.setValue(self.parent.sessionparams.params.__dict__[selected_param])
        if 'QDoubleSpinBox' in type_str:
            self.param_input_widget = QDoubleSpinBox()
            self.param_input_widget.setMaximum(4096.0)
            self.param_input_widget.setValue(self.parent.sessionparams.params.__dict__[selected_param])
        elif 'QLineEdit' in type_str:
            print("line edit found")
            self.param_input_widget = QLineEdit()
            self.param_input_widget.setText(str(self.parent.sessionparams.params.__dict__[selected_param]))
        elif 'QTextEdit' in type_str:
            self.param_input_widget = QTextEdit()
            self.param_input_widget.setText(str(self.parent.sessionparams.params.__dict__[selected_param]))
        elif 'QCheckBox' in type_str:
            self.param_input_widget = QCheckBox()
            self.param_input_widget.setChecked(self.parent.sessionparams.params.__dict__[selected_param])
        elif'QComboBox' in type_str:
            self.param_input_widget = QComboBox()
            items = [getattr(self.parent.widgets["unicontrol"].w, selected_param).itemText(i) for i in range(getattr(self.parent.widgets["unicontrol"].w, selected_param).count())]
            self.param_input_widget.addItems(items)
        self.paramlayout.addWidget(self.param_input_widget)
            #widget.setCurrentText(str(params.__dict__[name]))

        #if isinstance(input_widget, QLineEdit):
        #    self.parameter_input.setText(input_widget.text())
        #elif isinstance(input_widget, QSpinBox):
        #    self.parameter_input.setValue(input_widget.value())
        #elif isinstance(input_widget, QDoubleSpinBox):
        #    self.parameter_input.setValue(input_widget.value())
        #elif isinstance(input_widget, QComboBox):
        #    self.parameter_input.clear()
        #    for i in range(input_widget.count()):
        #        self.parameter_input.addItem(input_widget.itemText(i))
        #        self.parameter_input.setCurrentIndex(input_widget.currentIndex())
        #elif isinstance(input_widget, QCheckBox):
        #    self.parameter_input.setChecked(input_widget.isChecked())

    def space_engineer_example(self):
        pass
    def space_engineer_example_ui_func(self, item):
        print("This was ran on s single click")
        pass
    def get_space_engineer_example_params(self):
        params = types.SimpleNamespace()
        params.prompts = "Test Prompt"
        params.W = 512
        params.H = 512
        return params