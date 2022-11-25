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
        self.w.negative_prompts.setVisible(False)

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

        self.ui_unicontrol = UniControl_UI(self)


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
        self.unicontrol = self.parent
        self.connections()

    def connections(self):




        self.unicontrol.w.scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.scale.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.strength_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.strength.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.ddim_eta_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.ddim_eta.valueChanged.connect(self.update_float_scale_values)

        #self.unicontrol.w.mask_blur_slider.valueChanged.connect(self.update_float_values)
        #self.unicontrol.w.mask_blur.valueChanged.connect(self.update_float_scale_values)
        #self.unicontrol.w.reconstruction_blur_slider.valueChanged.connect(self.update_float_values)
        #self.unicontrol.w.reconstruction_blur.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.gradient_steps_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.gradient_steps.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.gradient_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.gradient_scale.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.clip_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.clip_scale.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.aesthetics_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.aesthetics_scale.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.mean_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.mean_scale.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.var_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.var_scale.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.exposure_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.exposure_scale.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.exposure_target_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.exposure_target.valueChanged.connect(self.update_float_scale_values)


        self.unicontrol.w.colormatch_scale_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.colormatch_scale.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.ignore_sat_weight_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.ignore_sat_weight.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.cut_pow_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.cut_pow.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.clamp_grad_threshold_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.clamp_grad_threshold.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.clamp_start_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.clamp_start.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.clamp_stop_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.clamp_stop.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.mask_contrast_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.mask_contrast.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.mask_brightness_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.mask_brightness.valueChanged.connect(self.update_float_scale_values)
        self.unicontrol.w.mask_overlay_blur_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.mask_overlay_blur.valueChanged.connect(self.update_float_scale_values)

        self.unicontrol.w.midas_weight_slider.valueChanged.connect(self.update_float_values)
        self.unicontrol.w.midas_weight.valueChanged.connect(self.update_float_scale_values)

    def update_float_values(self):

        self.unicontrol.w.scale.setValue(self.unicontrol.w.scale_slider.value()/10)
        self.unicontrol.w.strength.setValue(self.unicontrol.w.strength_slider.value()/10)
        self.unicontrol.w.ddim_eta.setValue(self.unicontrol.w.ddim_eta_slider.value()/10)
        #self.unicontrol.w.mask_blur.setValue(self.unicontrol.w.mask_blur_slider.value())
        #self.unicontrol.w.reconstruction_blur.setValue(self.unicontrol.w.reconstruction_blur_slider.value())
        self.unicontrol.w.gradient_steps.setValue(self.unicontrol.w.gradient_steps_slider.value())
        self.unicontrol.w.gradient_scale.setValue(self.unicontrol.w.gradient_scale_slider.value()/ 1000000000)
        self.unicontrol.w.clip_scale.setValue(self.unicontrol.w.clip_scale_slider.value()/10)
        self.unicontrol.w.aesthetics_scale.setValue( self.unicontrol.w.aesthetics_scale_slider.value())
        self.unicontrol.w.mean_scale.setValue(self.unicontrol.w.mean_scale_slider.value()/10)
        self.unicontrol.w.var_scale.setValue(self.unicontrol.w.var_scale_slider.value())
        self.unicontrol.w.exposure_scale.setValue(self.unicontrol.w.exposure_scale_slider.value()/10)
        self.unicontrol.w.exposure_target.setValue(self.unicontrol.w.exposure_target_slider.value()/10)
        self.unicontrol.w.colormatch_scale.setValue(self.unicontrol.w.colormatch_scale_slider.value())
        self.unicontrol.w.ignore_sat_weight.setValue(self.unicontrol.w.ignore_sat_weight_slider.value()/10)
        self.unicontrol.w.cut_pow.setValue(self.unicontrol.w.cut_pow_slider.value()/1000)
        self.unicontrol.w.clamp_grad_threshold.setValue(self.unicontrol.w.clamp_grad_threshold_slider.value())
        self.unicontrol.w.clamp_start.setValue(self.unicontrol.w.clamp_start_slider.value()/10)
        self.unicontrol.w.clamp_stop.setValue(self.unicontrol.w.clamp_stop_slider.value()/100)
        self.unicontrol.w.mask_contrast.setValue(self.unicontrol.w.mask_contrast_slider.value())
        self.unicontrol.w.mask_brightness.setValue(self.unicontrol.w.mask_brightness_slider.value())
        self.unicontrol.w.mask_overlay_blur.setValue(self.unicontrol.w.mask_overlay_blur_slider.value())
        self.unicontrol.w.midas_weight.setValue(self.unicontrol.w.midas_weight_slider.value())

    def update_float_scale_values(self):
        self.unicontrol.w.scale_slider.setValue(int(self.unicontrol.w.scale.value()*10))
        self.unicontrol.w.strength_slider.setValue(int(self.unicontrol.w.strength.value()*10))
        self.unicontrol.w.ddim_eta_slider.setValue(int(self.unicontrol.w.ddim_eta.value()*10))
        #self.unicontrol.w.mask_blur_slider.setValue(int(self.unicontrol.w.mask_blur_slider.value()))
        #self.unicontrol.w.reconstruction_blur_slider.setValue(int(self.unicontrol.w.reconstruction_blur_slider.value()))
        self.unicontrol.w.gradient_steps_slider.setValue(int(self.unicontrol.w.gradient_steps.value()))
        self.unicontrol.w.gradient_scale_slider.setValue(int(self.unicontrol.w.gradient_scale.value()* 1000000000))
        self.unicontrol.w.clip_scale_slider.setValue(int(self.unicontrol.w.clip_scale.value()*10))
        self.unicontrol.w.aesthetics_scale_slider.setValue( int(self.unicontrol.w.aesthetics_scale.value()))
        self.unicontrol.w.mean_scale_slider.setValue(int(self.unicontrol.w.mean_scale.value()*10))
        self.unicontrol.w.var_scale_slider.setValue(int(self.unicontrol.w.var_scale.value()))
        self.unicontrol.w.exposure_scale_slider.setValue(int(self.unicontrol.w.exposure_scale.value()*10))
        self.unicontrol.w.exposure_target_slider.setValue(int(self.unicontrol.w.exposure_target.value()*10))
        self.unicontrol.w.colormatch_scale_slider.setValue(int(self.unicontrol.w.colormatch_scale.value()))
        self.unicontrol.w.ignore_sat_weight_slider.setValue(int(self.unicontrol.w.ignore_sat_weight.value()*10))
        self.unicontrol.w.cut_pow_slider.setValue(int(self.unicontrol.w.cut_pow.value()*1000))
        self.unicontrol.w.clamp_grad_threshold_slider.setValue(int(self.unicontrol.w.clamp_grad_threshold.value()))
        self.unicontrol.w.clamp_start_slider.setValue(int(self.unicontrol.w.clamp_start.value()*10))
        self.unicontrol.w.clamp_stop_slider.setValue(int(self.unicontrol.w.clamp_stop.value()*100))
        self.unicontrol.w.mask_contrast_slider.setValue(int(self.unicontrol.w.mask_contrast.value()))
        self.unicontrol.w.mask_brightness_slider.setValue(int(self.unicontrol.w.mask_brightness.value()))
        self.unicontrol.w.mask_overlay_blur_slider.setValue(int(self.unicontrol.w.mask_overlay_blur.value()))
        self.unicontrol.w.midas_weight_slider.setValue(int(self.unicontrol.w.midas_weight.value()))
