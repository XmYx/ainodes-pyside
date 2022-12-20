import sys
from PySide6 import QtGui, QtCore, QtWidgets
import requests
from PySide6.QtCore import QSize, QMimeData, QEvent, QPoint
from PySide6.QtGui import QIcon, QPixmap, QDrag, Qt, QImage
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem, \
    QListView, QAbstractItemView
from bs4 import BeautifulSoup


# Create the layout for the main window

'''
Large images: tbs=isz:l
Medium images: tbs=isz:m
Icon sized images: tba=isz:i
Image size larger than 400×300: tbs=isz:lt,islt:qsvga
Image size larger than 640×480: tbs=isz:lt,islt:vga
Image size larger than 800×600: tbs=isz:lt,islt:svga
Image size larger than 1024×768: tbs=isz:lt,islt:xga
Image size larger than 1600×1200: tbs=isz:lt,islt:2mp
Image size larger than 2272×1704: tbs=isz:lt,islt:4mp
Image sized exactly 1000×1000: tbs=isz:ex,iszw:1000,iszh:1000
 Images in full color: tbs=ic:color
Images in black and white: tbs=ic:gray
Images that are red: tbs=ic:specific,isc:red [orange, yellow, green, teal, blue, purple, pink, white, gray, black, brown]
Image type Face: tbs=itp:face
Image type Photo: tbs=itp:photo
Image type Clipart: tbs=itp:clipart
Image type Line drawing: tbs=itp:lineart
Image type Animated (gif): tbs=itp:animated (thanks Dan)
Group images by subject: tbs=isg:to
Show image sizes in search results: tbs=imgo:1'''
# When the search button is clicked, perform the search and display the results

class ImageSearchWidget(QWidget):
    def __init__(self):
        super(ImageSearchWidget, self).__init__()
        self.layout = QVBoxLayout(self)

        # Create a line edit for the user to enter their search query
        self.query_edit = QLineEdit()

        # Create a button to initiate the search
        self.search_button = QPushButton("Search")

        self.search_button.clicked.connect(self.on_search_clicked)

        # Create a list widget to display the search results
        self.search_results = MyListView()

        # Add the line edit, search button, and search results to the layout
        self.layout.addWidget(self.query_edit)
        self.layout.addWidget(self.search_button)
        self.layout.addWidget(self.search_results)

        # Set the main window's layout
        self.setLayout(self.layout)

    def on_search_clicked(self):
        # Get the search query from the line edit
        print("task")
        query = self.query_edit.text()
        search_url = f"http://www.google.com/search?q=%22{query}%22&tbm=isch&tbs=ic:trans,isz:lt,islt:vga"
        # Use the requests library to send a GET request to the Google Images search URL
        response = requests.get(search_url.format(query=query))
        # Parse the response using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract the URLs of the images from the response
        images = soup.find_all("img", attrs={"src": True})
        # Clear the list of search results
        #self.search_results.clear()
        self.search_results.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.search_results.setIconSize(QSize(100, 100))
        # Add the search results to the list
        for image_url in images:
            # Create a QListWidgetItem to represent the image
            try:
                response = requests.get(image_url["src"])
                pixmap = QPixmap()
                pixmap.loadFromData(response.content)
                item = QListWidgetItem(QIcon(pixmap), "str")
                self.search_results.addItem(item)
            except:
                pass

class MyListView(QListWidget):

    def __init__(self, parent=None):
        QListWidget.__init__(self, parent)
        # Enable drag and drop for the list view
        self.setDragEnabled(True)

        # Set the size of the icons in the view
        self.setIconSize(QSize(32, 32))

        # Set the selection mode to allow only single item selection
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.installEventFilter(self)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        pixmap = item.icon().pixmap(item.icon().actualSize(QSize(512, 512)))  #QSize(512, 512)
        drag = QDrag(self)
        mimedata = QMimeData()
        mimedata.setImageData(pixmap)
        drag.setMimeData(mimedata)
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() / 2, pixmap.height() / 2))
        print(pixmap.size())
        drag.exec_(supportedActions)
class MyListView2(QListView):
    def __init__(self, parent=None):
        QListView.__init__(self, parent)
        # Enable drag and drop for the list view
        self.setDragEnabled(True)

        # Set the size of the icons in the view
        self.setIconSize(QSize(32, 32))

        # Set the selection mode to allow only single item selection
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        # Set the event filter on the list view
        self.installEventFilter(self)

    def eventFilter(self, source, event):
        # Check if the event is the QEvent.MouseButtonPress event
        if event.type() == QEvent.MouseButtonPress and source is self:
            # Get the index of the selected item in the QListView
            viewport = self.viewport()

            # Get the image from the selected item in the viewport
            image = viewport.selectedIndexes()[0].data(Qt.DecorationRole)
            if image:
                # Create a QMimeData object to hold the image data
                mimeData = QMimeData()
                mimeData.setImageData(image.pixmap(self.iconSize()).toImage())

                # Start the drag event with the specified mime data and drag action
                self.startDrag(mimeData, Qt.CopyAction)

        # Return the parent eventFilter method
        return super().eventFilter(source, event)

