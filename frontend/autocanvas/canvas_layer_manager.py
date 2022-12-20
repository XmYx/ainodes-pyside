import sys
from PySide6 import QtWidgets, QtCore

# Create the application object
app = QtWidgets.QApplication(sys.argv)

# Create the main window
window = QtWidgets.QWidget()

# Create the layout for the main window
layout = QtWidgets.QVBoxLayout()

# Create a QGraphicsScene
scene = QtWidgets.QGraphicsScene()

# Create a QGraphicsView to display the scene
view = QtWidgets.QGraphicsView(scene)

# Create a QListWidget to display the list of layers
layer_manager = QtWidgets.QListWidget()

# Handle changes to the layers in the layer manager
def on_item_changed(item):
    # Get the layer associated with the item
    layer = item.data(QtCore.Qt.UserRole)
    print(item.checkState)
    print(item.checkState())
    # If the item's check state has changed, set the layer's visibility
    if item.checkState() != layer.isVisible():
        
        layer.setVisible(item.checkState())

# Connect the itemChanged signal of the layer manager to the on_item_changed slot
layer_manager.itemChanged.connect(on_item_changed)

# Add some layers to the scene
for i in range(5):
    # Create a QGraphicsItemGroup to represent the layer
    layer = scene.createItemGroup([])

    # Create a QListWidgetItem to represent the layer in the layer manager
    item = QtWidgets.QListWidgetItem("Layer {}".format(i+1))

    # Set the item's data to the layer
    item.setData(QtCore.Qt.UserRole, layer)

    # Set the item's check state to the layer's visibility
    item.setCheckState(QtCore.Qt.Checked if layer.isVisible() else QtCore.Qt.Unchecked)

    # Add the item to the layer manager
    layer_manager.addItem(item)

# Add the view and layer manager to the layout
layout.addWidget(view)
layout.addWidget(layer_manager)

# Set the main window's layout
window.setLayout(layout)

# Show the main window
window.show()

# Run the application loop
app.exec_()