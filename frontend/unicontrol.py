from PySide6 import QtUiTools, QtCore
from PySide6.QtCore import QFile, QObject, QEasingCurve, QRect


class UniControl(QObject):

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/unicontrol.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        self.initAnimation()

        self.w.hideAdvButton.clicked.connect(self.hideAdvanced_anim)
        self.w.hideAesButton.clicked.connect(self.hideAesthetic_anim)
        self.w.hideAnimButton.clicked.connect(self.hideAnimation_anim)
        self.w.hidePlottingButton.clicked.connect(self.hidePlotting_anim)
        self.w.showHideAll.clicked.connect(self.show_hide_all_anim)


        self.advHidden = False
        self.aesHidden = False
        self.aniHidden = False
        self.ploHidden = False
        self.showAll = False

        self.initAnim.start()
        self.hideAdvanced_anim()
        self.hideAesthetic_anim()
        self.hideAnimation_anim()
        self.hidePlotting_anim()

        x = "W"

        getattr(self.w, x).setValue(15)
        print("QSlider" in str(getattr(self.w, x)))
        print("QCombobox" in str(getattr(self.w, x)))
        print("QTextEdit" in str(getattr(self.w, x)))

    def hideAdvanced_anim(self):
        if self.advHidden == False:
            self.setupAnimations(hideAdv=True)
            self.hideAdvAnim.start()
            self.advHidden = True
        elif self.advHidden == True:
            self.setupAnimations(showAdv=True)
            self.showAdvAnim.start()
            self.advHidden = False
        return

    def hideAesthetic_anim(self):
        if self.aesHidden == False:
            self.setupAnimations(hideAes=True)
            self.hideAesAnim.start()
            self.aesHidden = True
        elif self.aesHidden == True:
            self.setupAnimations(showAes=True)
            self.showAesAnim.start()
            self.aesHidden = False
        return

    def hideAnimation_anim(self):
        if self.aniHidden == False:
            self.setupAnimations(hideAni=True)
            self.hideAniAnim.start()
            self.aniHidden = True
        elif self.aniHidden == True:
            self.setupAnimations(showAni=True)
            self.showAniAnim.start()
            self.aniHidden = False
        return

    def show_hide_all_anim(self):
        print(self.showAll)
        if self.showAll == False:
            self.setupAnimations(hideAdv=True, hideAes=True, hideAni=True, hidePlo=True)
            self.hideAdvAnim.start()
            self.advHidden = True
            self.hideAesAnim.start()
            self.aesHidden = True
            self.hideAniAnim.start()
            self.aniHidden = True
            self.hidePloAnim.start()
            self.ploHidden = True

            self.showAll = True
        elif self.showAll == True:
            self.setupAnimations(showAdv=True, showAes=True, showAni=True, showPlo=True)
            self.showAdvAnim.start()
            self.advHidden = False
            self.showAesAnim.start()
            self.aesHidden = False
            self.showAniAnim.start()
            self.aniHidden = False
            self.showPloAnim.start()
            self.ploHidden = False

            self.showAll = False
    def hidePlotting_anim(self):
        if self.ploHidden == False:
            self.setupAnimations(hidePlo=True)
            self.hidePloAnim.start()
            self.ploHidden = True
        elif self.ploHidden == True:
            self.setupAnimations(showPlo=True)
            self.showPloAnim.start()
            self.ploHidden = False
        return

    def setupAnimations(self, hideAdv=False, showAdv=False, showAes=False, hideAes=False, showAni=False, hideAni=False, showPlo=False, hidePlo=False):
        if showAdv == True:
            self.showAdvAnim = QtCore.QPropertyAnimation(self.w.adv_values, b"maximumHeight")
            self.showAdvAnim.setDuration(1500)
            self.showAdvAnim.setStartValue(self.w.adv_values.height())
            self.showAdvAnim.setEndValue(self.w.height())
            self.showAdvAnim.setEasingCurve(QEasingCurve.Linear)
        if hideAdv == True:
            self.hideAdvAnim = QtCore.QPropertyAnimation(self.w.adv_values, b"maximumHeight")
            self.hideAdvAnim.setDuration(500)
            self.hideAdvAnim.setStartValue(self.w.adv_values.height())
            self.hideAdvAnim.setEndValue(0)
            self.hideAdvAnim.setEasingCurve(QEasingCurve.Linear)
        if showAes == True:
            self.showAesAnim = QtCore.QPropertyAnimation(self.w.aesthetic_values, b"maximumHeight")
            self.showAesAnim.setDuration(1500)
            self.showAesAnim.setStartValue(self.w.aesthetic_values.height())
            self.showAesAnim.setEndValue(self.w.height())
            self.showAesAnim.setEasingCurve(QEasingCurve.Linear)
        if hideAes == True:
            self.hideAesAnim = QtCore.QPropertyAnimation(self.w.aesthetic_values, b"maximumHeight")
            self.hideAesAnim.setDuration(500)
            self.hideAesAnim.setStartValue(self.w.aesthetic_values.height())
            self.hideAesAnim.setEndValue(0)
            self.hideAesAnim.setEasingCurve(QEasingCurve.Linear)
        if showAni == True:
            self.showAniAnim = QtCore.QPropertyAnimation(self.w.anim_values, b"maximumHeight")
            self.showAniAnim.setDuration(1500)
            self.showAniAnim.setStartValue(self.w.anim_values.height())
            self.showAniAnim.setEndValue(self.w.height())
            self.showAniAnim.setEasingCurve(QEasingCurve.Linear)
        if hideAni == True:
            self.hideAniAnim = QtCore.QPropertyAnimation(self.w.anim_values, b"maximumHeight")
            self.hideAniAnim.setDuration(500)
            self.hideAniAnim.setStartValue(self.w.anim_values.height())
            self.hideAniAnim.setEndValue(0)
            self.hideAniAnim.setEasingCurve(QEasingCurve.Linear)
        if showPlo == True:
            self.showPloAnim = QtCore.QPropertyAnimation(self.w.plotting_frame, b"maximumHeight")
            self.showPloAnim.setDuration(1500)
            self.showPloAnim.setStartValue(self.w.plotting_frame.height())
            self.showPloAnim.setEndValue(self.w.height())
            self.showPloAnim.setEasingCurve(QEasingCurve.Linear)
        if hidePlo == True:
            self.hidePloAnim = QtCore.QPropertyAnimation(self.w.plotting_frame, b"maximumHeight")
            self.hidePloAnim.setDuration(500)
            self.hidePloAnim.setStartValue(self.w.plotting_frame.height())
            self.hidePloAnim.setEndValue(0)
            self.hidePloAnim.setEasingCurve(QEasingCurve.Linear)
        return
    def initAnimation(self):

        self.initAnim = QtCore.QPropertyAnimation(self.w.dockWidget, b"maximumWidth")
        self.initAnim.setDuration(1500)
        self.initAnim.setStartValue(0)
        self.initAnim.setEndValue(self.parent.width())
        self.initAnim.setEasingCurve(QEasingCurve.Linear)




class UniControl_UI:
    def __init__(self, parent):
        self.parent = parent
        self.unicontrol = self.parent.unicontrol

