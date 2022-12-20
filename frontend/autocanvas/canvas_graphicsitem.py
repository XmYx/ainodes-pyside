from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsItem


class Rectangle(QGraphicsPixmapItem):
    def __init__(self, parent, prompt, x, y, w, h, id, order = None, img_path = None, image = None, render_index=None, params=None):
        self.pixmap = QPixmap('mypixmap.png')
        # Create a QGraphicsPixmapItem to show the pixmap
        self.setPixmap(self.pixmap)
        # Set the position of the pixmap item
        self.setPos(self.x, self.y)
        # Make the pixmap item movable with the mouse
        self.setFlag(QGraphicsItem.ItemIsMovable)
        # Set the layer value of the pixmap item
        self.setZValue(10)
        # Add the pixmap item to the scene
        self.parent.scene.addItem(self)