# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'thumbnails.ui'
##
## Created by: Qt User Interface Compiler version 6.3.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDockWidget, QListView,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSlider, QVBoxLayout, QWidget)

class Thumbnails(object):
    def setupUi(self, thumbnails):
        if not thumbnails.objectName():
            thumbnails.setObjectName(u"thumbnails")
        thumbnails.setWindowModality(Qt.WindowModal)
        thumbnails.resize(759, 544)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(thumbnails.sizePolicy().hasHeightForWidth())
        thumbnails.setSizePolicy(sizePolicy)
        thumbnails.setBaseSize(QSize(800, 680))
        self.verticalLayout = QVBoxLayout(thumbnails)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.dockWidget = QDockWidget(thumbnails)
        self.dockWidget.setObjectName(u"dockWidget")
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.thumbs = QListWidget(self.dockWidgetContents)
        self.thumbs.setObjectName(u"thumbs")
        sizePolicy.setHeightForWidth(self.thumbs.sizePolicy().hasHeightForWidth())
        self.thumbs.setSizePolicy(sizePolicy)
        self.thumbs.setBaseSize(QSize(800, 680))
        self.thumbs.setFocusPolicy(Qt.NoFocus)
        self.thumbs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thumbs.setAcceptDrops(False)
#if QT_CONFIG(tooltip)
        self.thumbs.setToolTip(u"")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(accessibility)
        self.thumbs.setAccessibleName(u"")
#endif // QT_CONFIG(accessibility)
        self.thumbs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thumbs.setProperty("showDropIndicator", False)
        self.thumbs.setDragEnabled(False)
        self.thumbs.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.thumbs.setDefaultDropAction(Qt.IgnoreAction)
        self.thumbs.setIconSize(QSize(150, 150))
        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setMovement(QListView.Free)
        self.thumbs.setResizeMode(QListView.Adjust)
        self.thumbs.setLayoutMode(QListView.Batched)
        self.thumbs.setGridSize(QSize(150, 200))
        self.thumbs.setViewMode(QListView.IconMode)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setWordWrap(True)
        self.thumbs.setSelectionRectVisible(False)
        self.thumbs.setSortingEnabled(False)

        self.verticalLayout_2.addWidget(self.thumbs)

        self.refresh = QPushButton(self.dockWidgetContents)
        self.refresh.setObjectName(u"refresh")

        self.verticalLayout_2.addWidget(self.refresh)

        self.thumbsZoom = QSlider(self.dockWidgetContents)
        self.thumbsZoom.setObjectName(u"thumbsZoom")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.thumbsZoom.sizePolicy().hasHeightForWidth())
        self.thumbsZoom.setSizePolicy(sizePolicy1)
        self.thumbsZoom.setMinimumSize(QSize(0, 15))
        self.thumbsZoom.setMinimum(5)
        self.thumbsZoom.setMaximum(512)
        self.thumbsZoom.setValue(150)
        self.thumbsZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.thumbsZoom)

        self.dockWidget.setWidget(self.dockWidgetContents)

        self.verticalLayout.addWidget(self.dockWidget)


        self.retranslateUi(thumbnails)

        QMetaObject.connectSlotsByName(thumbnails)
    # setupUi

    def retranslateUi(self, thumbnails):
        thumbnails.setWindowTitle(QCoreApplication.translate("thumbnails", u"Thumbnails", u"thumbs"))
#if QT_CONFIG(tooltip)
        thumbnails.setToolTip("")
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(accessibility)
        thumbnails.setAccessibleName("")
#endif // QT_CONFIG(accessibility)
        self.refresh.setText(QCoreApplication.translate("thumbnails", u"refresh", None))
    # retranslateUi

