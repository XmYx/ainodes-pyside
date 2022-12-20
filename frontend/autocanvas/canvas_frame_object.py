import time

from PySide6 import QtCore, QtGui
from PySide6.QtCore import QObject, QEvent, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QGraphicsProxyWidget, QGraphicsDropShadowEffect

# The Mandatory singleton import where everything (should be) is stored
from backend.singleton import singleton
from frontend.ui_classes import SimplePromptDisplay
gs = singleton
__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)
__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)

class Rectangle(object):
    def __init__(self, parent, prompt, x, y, w, h, id, order = None, img_path = None, image = None, render_index=None, params=None):
        self.parent = parent
        self.prompt = prompt
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = image
        self.render_index = render_index
        self.images = []
        self.params = params
        #print(f"Hello, I'm a rectangle with seed {params.seed}")
        if self.image is not None:
            self.images.append(self.image)
            self.render_index = 0
        self.order = order
        self.color = __idleColor__
        self.timestring = time.time()
        self.active = True
        self.PILImage = None
        self.running = False
        self.img_path = img_path
        #self.signals = RectangleCallbacks()
        self.timer = QtCore.QTimer()
        self.subwidgets = {}
        self.subwidgets['prompt'] = SimplePromptDisplay()
        self.subwidgets['prompt'].w.prompts.setAlignment(QtCore.Qt.AlignCenter)
        self.proxy = QGraphicsProxyWidget()
        self.proxy.setWidget(self.subwidgets['prompt'].w)
        self.prompt_visible = None
        self.event_filter = MyEventFilter()
        self.subwidgets['prompt'].w.installEventFilter(self.event_filter)
        #self.show_prompt()

    def play(self):
        if self.parent.running == True:
            if self.images != []:
                self.timer = QtCore.QTimer()
                self.timer.timeout.connect(self.iterate)
                self.timer.start(80)
                #self.signals.start_main.emit()
                self.running = True

    def iterate(self):
        self.render_index = (self.render_index + 1) % len(self.images)
        if self.render_index == len(self.images):
            self.render_index = 0
        self.image = self.images[self.render_index]
        if self.running == False:
            self.parent.newimage = True
            self.parent.update()
        #print(self.render_index)
        #(len(self.images))
    def iterate_back(self):
        self.render_index = (self.render_index - 1) % len(self.images)
        if self.render_index == -1:
            self.render_index = len(self.images)
        self.image = self.images[self.render_index]
        if self.running == False:
            self.parent.newimage = True
            self.parent.update()

        #print(self.render_index)

    def stop(self):
        self.timer.stop()
        self.running = False

    def show_prompt(self):
        pass
        if self.image is not None:
            print("showing..")

            self.subwidgets['prompt'].w.setStyleSheet(
                """
                QLabel {
                    font: 36pt 'Segoe UI Italic';
                    background-color: rgba(55, 55, 55, 0.5);
                    border-radius: 20px;
                    padding: 10px;
                }
                """
            )

            self.parent.scene.addItem(self.proxy)
            #self.subwidgets['prompt'].setGeometry(self.widget.w.rect())
            self.subwidgets['prompt'].w.prompts.setText(self.prompt)
            #self.subwidgets['prompt'].w.prompts.setWordWrap(True)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setColor(QtGui.QColor(0, 0, 0))
            shadow.setOffset(25, 25)

            # Apply the effect to the label
            self.subwidgets['prompt'].w.setGraphicsEffect(shadow)

            #self.subwidgets['prompt'].setWindowOpacity(0)
            self.subwidgets['prompt'].w.setGeometry(self.x, self.y, self.w, self.h)
            self.subwidgets['prompt'].w.show()
            self.animation = QtCore.QPropertyAnimation(
                self.subwidgets['prompt'].w, b"windowOpacity"
            )
            self.animation.setDuration(250)
            self.animation.setStartValue(0)
            self.animation.setEndValue(0.75)
            self.animation.start()
            self.prompt_visible = True
            self.subwidgets['prompt'].w.deletebutton.clicked.connect(self.annihilation)
    def annihilation(self):
        self.hide_prompt(destroy=True)
        #self.parent.uid = self.id
        #self.parent.removeitem_by_uid_from_scene()
        #self.subwidgets['prompt'].w.destroy()
        self.parent.parent.parent.delete_outpaint_frame()
        gs.donthover = None

    def hide_prompt(self, destroy=None):
        if self.image is not None:
            print("hiding..")
            self.subwidgets['prompt'].w.deletebutton.clicked.disconnect()

            #self.subwidgets['prompt'].hide()
            self.animation = QtCore.QPropertyAnimation(
                self.subwidgets['prompt'].w, b"windowOpacity"
            )
            self.animation.setDuration(250)
            self.animation.setStartValue(self.subwidgets['prompt'].w.windowOpacity())
            self.animation.setEndValue(0)
            self.animation.start()
            self.prompt_visible = None
        if destroy:
            QTimer.singleShot(250, self.finish_annihilation)
    def finish_annihilation(self):
        self.parent.scene.removeItem(self.proxy)
        self.subwidgets['prompt'].w.destroy()

class RectangleProxy(QGraphicsProxyWidget):
    def __init__(self, widget, parent):
        super(RectangleProxy, self).__init__()
        self.widget = widget
        self.parent = parent
        #self.event_filter = MyEventFilter()
        #self.installEventFilter(self.event_filter)


class MyEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            # Handle the QEvent::Enter event here
            gs.donthover = True
            print("filtering")
        elif event.type() == QEvent.Leave:
            print("filter off")
            gs.donthover = None
        # Return False to continue processing the event
        return False
