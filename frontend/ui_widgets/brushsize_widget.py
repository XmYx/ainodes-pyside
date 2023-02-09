from PySide6 import QtGui, QtCore, QtWidgets


class BrushSizeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.brush_size = 50
        self.setFixedSize(100, 100)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        painter.setBrush(QtGui.QBrush(QtCore.Qt.black))
        painter.drawEllipse(
            QtCore.QRectF(
                0,
                0,
                self.brush_size,
                self.brush_size
            )
        )

    def setBrushSize(self, brush_size):
        self.brush_size = brush_size
        self.update()