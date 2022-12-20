
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Signal, QSize, QRect, QObject, Qt
from PySide6.QtGui import Qt, QColor, QFont
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QSlider, QDockWidget, QHBoxLayout, QSpinBox, \
    QPushButton

from backend.singleton import singleton
from frontend.autocanvas.canvas_autocanvas import Canvas

gs = singleton
import random

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)
__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)


class RectangleCallbacks(QObject):
    start_main = Signal()

class PaintUI(QDockWidget):

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"Outpaint")
        self.setAccessibleName(u'outpaintCanvas')
        self.parent = parent
        self.acceptDrops()
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setBaseSize(QSize(800, 680))
        self.setLayout(QVBoxLayout())

        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.W_spinbox = QSpinBox()
        self.W_spinbox.setMinimum(256)
        self.W_spinbox.setMaximum(16000)
        self.W_spinbox.setValue(512)
        self.W_spinbox.setSingleStep(64)
        self.H_spinbox = QSpinBox()
        self.H_spinbox.setMinimum(256)
        self.H_spinbox.setMaximum(16000)
        self.H_spinbox.setValue(512)
        self.H_spinbox.setSingleStep(64)

        self.W = QSlider()
        self.W.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.W.setMinimumSize(QSize(100, 15))
        self.W.setMaximumSize(QSize(1000, 15))
        self.W.setMinimum(512)
        self.W.setMaximum(16000)
        self.W.setValue(4096)
        self.W.setPageStep(64)
        self.W.setSingleStep(64)
        self.W.setOrientation(Qt.Horizontal)

        self.H = QSlider()
        self.H.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.H.setMinimumSize(QSize(100, 15))
        self.H.setMaximumSize(QSize(1000, 15))
        self.H.setMinimum(512)
        self.H.setMaximum(16000)
        self.H.setValue(4096)
        self.H.setPageStep(64)
        self.H.setSingleStep(64)
        self.H.setOrientation(Qt.Horizontal)
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 0, 5, 0)
        self.canvas = Canvas(self)
        self.widget_2 = QWidget()
        self.horizontalLayout = QHBoxLayout(self.widget_2)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontal")
        self.horizontalLayout.setContentsMargins(5, 0, 5, 0)
        self.btnOne = MyQPushButton(text="Menu!", parent=self, )

        #self.menu = QMenu(self)
        #self.menu.addAction("First Item")
        #self.menu.addAction("Second Item")
        #self.menu.addAction("Third Item")

        #self.btnOne.setMenu(self.menu)

        #self.horizontalLayout.addWidget(self.W)
        #self.horizontalLayout.addWidget(self.W_spinbox)

        #self.horizontalLayout.addWidget(self.H)
        #self.horizontalLayout.addWidget(self.H_spinbox)

        self.verticalLayout_2.addWidget(self.canvas)
        self.canvas.scene.addWidget(self.btnOne)
        #self.btnOne.setGeometry(0, 0, 250, 250)
        self.btnOne.setStyleSheet("QPushButton{image: url(:/frontend/icons/activity.svg);border-radius: 1px;}"
                      "QPushButton:hover{image: url(:frontend/icons/plus.svg);border-radius: 1px;}")
        #self.verticalLayout_2.addWidget(self.widget_2)

        self.setWidget(self.dockWidgetContents)
        #self.canvas.setMouseTracking(True)  # Mouse events
        #self.canvas.hoverCheck()
        #self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.H.valueChanged.connect(self.update_spinners)
        self.W.valueChanged.connect(self.update_spinners)
        self.H_spinbox.valueChanged.connect(self.update_sliders)
        self.W_spinbox.valueChanged.connect(self.update_sliders)
        self.setAcceptDrops(True)
        self.canvas.setAcceptDrops(True)
    def dropEvent(self, event):
        print(event.mimeData())
    def update_spinners(self):
        self.H_spinbox.setValue(self.H.value())
        self.W_spinbox.setValue(self.W.value())

    def update_sliders(self):
        self.H.setValue(int(self.H_spinbox.value()))
        self.W.setValue(int(self.W_spinbox.value()))

    def rectangleDraw(self):
        self.canvas.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)
class MyQPushButton(QPushButton):
    def __init__(self, text, parent=None):
        super(MyQPushButton, self).__init__(text, parent)
        self.parent = parent
        self.setText(text)
def spiralOrder(matrix):
    ans = []

    if (len(matrix) == 0):
        return ans

    m = len(matrix)
    n = len(matrix[0])
    seen = [[0 for i in range(n)] for j in range(m)]
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    x = 0
    y = 0
    di = 0
    c = 0
    # Iterate from 0 to R * C - 1
    for i in range(m * n):
        matrix[x][y]['order'] = c
        ans.append(matrix[x][y])
        c += 1
        ###print(matrix[x][y])
        seen[x][y] = True
        cr = x + dr[di]
        cc = y + dc[di]

        if (0 <= cr and cr < m and 0 <= cc and cc < n and not(seen[cr][cc])):
            x = cr
            y = cc
        else:
            di = (di + 1) % 4
            x += dr[di]
            y += dc[di]
    return ans

def random_path(order, columns):
    templist = []
    newlist = []
    for i in order:
        for x in i:
            ###print(x['order'])
            templist.append(x["order"])
    ###print(templist)
    x = 0
    c = 0
    steps = len(templist) - 1
    newlist.append(templist[x])
    while templist != []:
        ###print(len(templist))
        ###print(templist[x])
        match = False
        while match == False:

            newpair = random.choice(templist)

            ###print(f"we are loooking for a match between:\n{templist[x]} and {newpair}")

            if newpair - 1 == templist[x] or newpair + 1 == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1


            elif newpair - columns == templist[x] or newpair + columns == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1

            elif newpair - columns - 1 == templist[x] or newpair + columns + 1 == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1

#            if templist == []:
#                return
            #    match = True
            #    x = len(templist) - 1
            #    break
        #if x == len(templist) - 1:
            #break

    ##print(f"this is the new list {newlist}")
    for i in order:
        for x in i:
            x["order"] = newlist[i]
    return order




class FlipWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super(FlipWidget, self).__init__()

        # Create the property animation
        self.animation = QtCore.QPropertyAnimation(self, b"scale")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)

        # Create the flip button
        self.flip_button = QPushButton("Flip")
        self.flip_button.clicked.connect(self.animate)

        # Set the initial scale to 1
        #self.setScale(1)

        # Create the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.flip_button)
        self.setLayout(self.layout)

    def animate(self):
        self.scale = 1
        self.animation.start()

        # Set the initial scale to 1


    @QtCore.Property(float)
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        self.update()
    def paintEvent(self, event):
        # Get the size of the widget
        width = self.width()
        height = self.height()

        # Create the painter
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Set the transformation origin to the center of the widget
        painter.translate(width / 2, height / 2)
        painter.scale(self._scale, self._scale)
        painter.translate(-width / 2, -height / 2)

        # Draw the content of the widget
        painter.drawText(0, 0, width, height, QtCore.Qt.AlignCenter, "Flip me!")