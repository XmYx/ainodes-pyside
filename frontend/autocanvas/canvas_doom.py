from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget, QApplication, QVBoxLayout

# Create the main window
app = QApplication()
web = QWebEngineView()

# Create the web view that will display the game
web.setUrl("https://playclassic.games/games/first-person-shooter-dos-games-online/play-doom-ii-hell-earth-online/play/")

# Set the web view as the central widget of the main window
window = QWidget()

# Create the layout for the main window
layout = QVBoxLayout()
layout.addWidget(web)

# Set the main window's layout
window.setLayout(layout)

# Show the main window
window.show()

# Run the application loop
app.exec()
