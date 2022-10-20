from PySide6.QtCore import Signal, QLine, QPoint, QRectF, QSize, QRect
from PySide6.QtGui import Qt, QColor, QFont, QPalette, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor, \
    QPixmap
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu, QGraphicsScene, \
    QGraphicsView, QGraphicsItem, QGraphicsWidget, QLabel

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)


class Rectangle(object):
    def __init__(self, x, y, w, h, id):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class Canvas(QLabel):

    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap(600, 300)
        self.setPixmap(self.pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QPainter(self.pixmap)
        painter.begin(self.pixmap)
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        #self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


"""class Outpaint(QGraphicsWidget):

    keyFramesUpdated = Signal()
    #selectionChanged = Signal(VideoSample)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.canvas = QPixmap(4096, 4096)
        self.scene.addPixmap(self.canvas)

        self.duration = 1000
        self.length = 1000

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.clicking = False  # Check if mouse left button is being pressed
        self.is_in = False  # check if user is in the widget
        #self.setMouseTracking(False)  # Mouse events
        #self.setAutoFillBackground(True)  # background
        self.pos = None

        self.posy = None
        self.rectangle = None
        self.rectList = []
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, self.length, 200)
        self.setWindowTitle("TESTE")

        # Set Background
        pal = QPalette()
        pal.setColor(QPalette.Base, self.backgroundColor)
        self.setPalette(pal)

    def rectangleDraw(self):
        self.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)

    def paintEvent(self, event):

        qp = QPainter(self.canvas)
        qp.device()
        #qp.begin()
        qp.setPen(self.textColor)
        qp.setFont(self.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0
        # Draw time
        scale = self.getScale()
        qp.setPen(QPen(Qt.darkGreen, 5, Qt.SolidLine))
        #qp.drawLine(0, 500, self.width(), 40)
        if self.rectangle is not None:
            qp.drawRect(self.rectangle)


        #qp.end()

    # Mouse movement
    def mouseMoveEvent(self, e):

        print("5135435")

        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer
        self.pos = e.pos()
        #self.rectangleDraw()

        if self.clicking:
            pass
            #self.oldPos = self.pointerPos
            #x = self.pos
            #self.pointerPos = x
            #self.pointerTimePos = self.pointerPos*self.getScale()





        self.update()

    # Mouse pressed

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()

            #print(self.keyClicked)
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.clicking = True  # Set clicking check to true

        elif e.button() == Qt.RightButton:
            self.popMenu = QMenu(self)
            menuPosition = QCursor.pos()
            self.popMenu.clear()
            #populate
            self.populateBtnContext()
            #show
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            #self.popMenu.delete_action.triggered.connect(self.delete_action)

    def populateBtnContext(self):

        # Do some if here :
        self.popMenu.delete_action = QAction('delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)

    # Mouse release


    def delete_action(self):
        pass
    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false

    # Enter
    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    # check selection



    # Get time string from seconds
    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        #return "%02d:%02d:%02d" % (h, m, s)
        return "%05d" % (seconds)


    # Get scale from length
    def getScale(self):
        #return float(self.duration)/float(self.width())
        pass


    # Set background color
    def setBackgroundColor(self, color):
        self.backgroundColor = color

    # Set text color
    def setTextColor(self, color):
        self.textColor = color

    # Set Font
    def setTextFont(self, font):
        self.font = font"""








class OutpaintUI(QDockWidget):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        self.setWindowModality(Qt.WindowModal)
        self.setMouseTracking(True)  # Mouse events

        #self.resize(759, 544)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setBaseSize(QSize(800, 680))
        self.setLayout(QVBoxLayout())
        #self.verticalLayout = QVBoxLayout(self)
        #self.dockWidget = QDockWidget(self)
        #self.dockWidget.setObjectName(u"dockWidget")
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        #self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 0, 5, 0)
        self.canvas = Canvas()
        #self.item = QGraphicsItem(self.outpaint.canvas)

        #self.graphicsView = QGraphicsView()


        #self.graphicsView.setScene(self.outpaint.scene)

        #self.sample_1 = VideoSample(20)
        #self.timeline.videoSamples.append(self.sample_1)
        self.tZoom = QSlider()
        
        self.tZoom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tZoom.setMinimumSize(QSize(100, 15))
        self.tZoom.setMaximumSize(QSize(1000, 15))
        self.tZoom.setMinimum(0.0)
        self.tZoom.setMaximum(10.0)
        self.tZoom.setValue(1.0)
        self.tZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.canvas)
        #self.verticalLayout_2.addWidget(self.tZoom)

        self.setWidget(self.dockWidgetContents)
        #self.tZoom.valueChanged.connect(self.update_timelineZoom)
        #self.timeline.scale = 1


    def rectangleDraw(self):
        self.outpaint.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)

    """def mouseMoveEvent(self, e):
        print('event is still there...')
        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer
        self.pos = e.pos()
        self.rectangleDraw()

        #if self.clicking:
            #pass
    def paintEvent(self, event):

        qp = QPainter(self.outpaint.canvas)
        #qp.device()
        qp.begin(self.outpaint.canvas)
        qp.setPen(self.outpaint.textColor)
        qp.setFont(self.outpaint.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0
        # Draw time
        #scale = self.outpaint.getScale()
        qp.setPen(QPen(Qt.darkGreen, 5, Qt.SolidLine))
        #qp.drawLine(0, 500, self.width(), 40)
        if self.outpaint.rectangle is not None:
            qp.drawRect(self.outpaint.rectangle)
        qp.end()
class Outpaint(QWidget):

    #keyFramesUpdated = Signal()
    #selectionChanged = Signal(VideoSample)

    def __init__(self):
        super().__init__()

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.rectangle = None
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("TESTE")

        # Set Background
        pal = QPalette()
        pal.setColor(QPalette.Base, self.backgroundColor)
        self.setPalette(pal)

    def paintEvent(self, event):

        qp = QPainter()
        #qp.device()
        qp.begin(self)
        qp.setPen(self.textColor)
        qp.setFont(self.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0


        # Draw down line
        qp.setPen(QPen(Qt.darkCyan, 5, Qt.SolidLine))
        qp.drawLine(0, 500, self.width(), 40)
        if self.rectangle is not None:
            qp.drawRect(self.rectangle)
        qp.end()
        # Draw dash lines

    def mouseMoveEvent(self, e):
        print("outpaint move event")
        print(e.pos().y(), e.pos().x())


        self.pos = e.pos()
        self.rectangleDraw()
        self.update()

    def mousePressEvent(self, e):
        if self.is_in == True:
            print('mousePressEvent while in')
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()
            self.clicking = False
        elif e.button() == Qt.RightButton:
            pass



    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false



    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    def getScale(self):
        return float(self.width())


        #self.update()
    def setTextFont(self, font):
        self.font = font
        
        point = 0
        qp.setPen(QPen(self.textColor))
        qp.drawLine(0, 40, self.width(), 40)
        while point <= self.width():
            if point % 30 != 0:
                qp.drawLine(3 * point, 40, 3 * point, 30)
            else:
                qp.drawLine(3 * point, 40, 3 * point, 20)
            point += 10

        if self.pos is not None and self.is_in:
            qp.drawLine(self.pos, 0, self.pos, 40)

        if self.pointerPos is not None:
            self.pointerTimePos = int(self.pointerTimePos)
            line = QLine(QPoint(int(self.pointerTimePos/self.getScale()), 40),
                         QPoint(int(self.pointerTimePos/self.getScale()), self.height()))
            poly = QPolygon([QPoint(int(self.pointerTimePos/self.getScale() - 10), 20),
                             QPoint(int(self.pointerTimePos/self.getScale() + 10), 20),
                             QPoint(int(self.pointerTimePos/self.getScale()), 40)])
        else:
            line = QLine(QPoint(0, 0), QPoint(0, self.height()))
            poly = QPolygon([QPoint(-10, 20), QPoint(10, 20), QPoint(0, 40)])

        if self.selectedValueType is not None:
            for i in self.keyFrameList:
                if i is not None:
                    if i.valueType == self.selectedValueType:

                        kfbrush = QBrush(Qt.darkRed)
                        kfStartPoint = int(int(i.position) / self.getScale())
                        scaleMod = 5
                        kfPoly = QPolygon([QPoint(int(kfStartPoint - scaleMod), 50), QPoint(kfStartPoint, 45), QPoint(kfStartPoint + scaleMod, 50), QPoint(kfStartPoint, 55)])
                        qp.setPen(Qt.darkRed)
                        qp.setBrush(kfbrush)
                        qp.drawPolygon(kfPoly)

        # Draw samples
        t = 0
        for sample in self.videoSamples:
            # Clear clip path
            path = QPainterPath()

            path.addRoundedRect(QRectF((t + sample.startPos)/scale, 50, sample.duration / scale, 200), 10, 10)

            qp.setClipPath(path)

            # Draw sample
            path = QPainterPath()
            qp.setPen(sample.color)
            qp.setBrush(sample.color)



            #path.addRoundedRect(QRectF(((t + sample.startPos)/scale), 50, (sample.duration / scale), 50), 10, 10)
            path.addRect((t + sample.startPos)/scale, 50, (sample.duration / scale), 50)
            #sample.startPos = (t + sample.startPos)*scale
            sample.endPos = (t + sample.startPos)/scale + sample.duration/scale
            qp.fillPath(path, sample.color)
            qp.drawPath(path)


            # Draw preview pictures
            if sample.picture is not None:
                if sample.picture.size().width() < sample.duration/scale:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t/scale, 52.5, sample.picture.size().width(), 45), 10, 10)
                    qp.setClipPath(path)
                    qp.drawPixmap(QRect(int(t/scale), 52.5, sample.picture.size().width(), 45), sample.picture)
                else:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t / scale, 52.5, sample.duration/scale, 45), 10, 10)
                    qp.setClipPath(path)
                    pic = sample.picture.copy(0, 0, sample.duration/scale, 45)
                    qp.drawPixmap(QRect(int(t / scale), 52.5, sample.duration/scale, 45), pic)
            t += sample.duration

        # Clear clip path
        path = QPainterPath()
        path.addRect(self.rect().x(), self.rect().y(), self.rect().width(), self.rect().height())
        qp.setClipPath(path)

        # Draw pointer
        qp.setPen(Qt.darkCyan)
        qp.setBrush(QBrush(Qt.darkCyan))

        qp.drawPolygon(poly)
        qp.drawLine(line)
        qp.end()

    # Mouse movement
    def mouseMoveEvent(self, e):



        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer

        self.checkKeyframeHover(self.pos)

        if self.clicking:

            self.oldPos = self.pointerPos
            x = self.pos
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos*self.getScale()


            if self.keyHover == True:
                for item in self.keyFrameList:
                    if self.selectedKey is item.uid:
                        item.position = int(self.pointerPos*self.scale)
            if self.edgeGrabActive == True:


                for sample in self.videoSamples:
                    sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
            elif self.middleHoverActive == True:

                self.scale = self.getScale()
                for sample in self.videoSamples:
                    change = (x - self.oldPos)
                    change = (change * self.scale)
                    #print(change)
                    sample.startPos = sample.startPos + change
                    sample.endPos = sample.endPos + change



        self.update()

    # Mouse pressed
    def checkKeyframeHover(self, x):
        for item in self.keyFrameList:
            kfStartPoint = int(int(item.position) / self.getScale())
            if kfStartPoint - 5 < x < kfStartPoint + 5 and 55 > self.posy > 45:
                self.keyHover = True
                #print(item.uid)
                self.hoverKey = item.uid

    def checkKeyClicked(self):
        for item in self.keyFrameList:
            if self.hoverKey is item.uid:
                self.selectedKey = self.hoverKey
                self.keyHover = True

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()
            self.checkKeyClicked()

            #print(self.keyClicked)
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.clicking = True  # Set clicking check to true
            if self.edgeGrab == True:
                self.edgeGrabActive = True
            else:
                self.edgeGrabActive = False
            if self.middleHover == True:
                self.middleHoverActive = True
            else:
                self.middleHoverActive = False
        elif e.button() == Qt.RightButton:
            self.popMenu = QMenu(self)
            menuPosition = QCursor.pos()
            self.checkKeyClicked()
            self.popMenu.clear()
            #populate
            self.populateBtnContext()
            #show
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            self.popMenu.delete_action.triggered.connect(self.delete_action)

    def populateBtnContext(self):

        # Do some if here :
        self.popMenu.delete_action = QAction('delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)

    # Mouse release


    def delete_action(self):
        for idx, item in enumerate(self.keyFrameList):
            print(idx)
            print(item)
            if self.hoverKey is item.uid:
                self.keyFrameList.pop(idx)

                #item.remove()
                #return
    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false
            self.selectedKey = None
            self.keyHover = False
            self.hoverKey = None
            self.keyFramesUpdated.emit()

    # Enter
    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    # check selection
    def checkSelection(self, x):
        # Check if user clicked in video sample
        for sample in self.videoSamples:
            if sample.startPos + 25 < x < sample.endPos - 25:
                sample.color = Qt.darkCyan
                self.middleHover = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.middleHover = False


    def checkEdges(self, x, y=50):

        for sample in self.videoSamples:
            if sample.startPos < x < sample.startPos + 24:
                sample.color = Qt.darkMagenta
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            elif sample.endPos - 24 < x < sample.endPos:
                sample.color = Qt.darkGreen
                self.edgeGrab = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.edgeGrab = False


    # Get time string from seconds
    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        #return "%02d:%02d:%02d" % (h, m, s)
        return "%05d" % (seconds)


    # Get scale from length
    def getScale(self):
        return float(self.duration)/float(self.width())

    # Get duration
    def getDuration(self):
        return self.duration

    # Get selected sample
    def getSelectedSample(self):
        return self.selectedSample

    # Set background color
    def setBackgroundColor(self, color):
        self.backgroundColor = color

    # Set text color
    def setTextColor(self, color):
        self.textColor = color

    # Set Font
    def setTextFont(self, font):
        self.font = font"""