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

        #self.w.hideAdvButton.clicked.connect(self.hideAdvanced_anim)
        #self.w.hideAesButton.clicked.connect(self.hideAesthetic_anim)
        #self.w.hideAnimButton.clicked.connect(self.hideAnimation_anim)
        #self.w.hidePlottingButton.clicked.connect(self.hidePlotting_anim)
        #self.w.showHideAll.clicked.connect(self.show_hide_all_anim)


        self.w.toggle_sampler.stateChanged.connect(self.hideSampler_anim)
        self.w.show_output_setup.stateChanged.connect(self.hideOutput_anim)
        self.w.show_init_setup.stateChanged.connect(self.hideInitImage_anim)
        self.w.show_mask_setup.stateChanged.connect(self.hideMaskImage_anim)
        self.w.toggle_outpaint.stateChanged.connect(self.hideOutpaint_anim)
        self.w.toggle_animations.stateChanged.connect(self.hideAnimation_anim)
        self.w.toggle_plotting.stateChanged.connect(self.hidePlotting_anim)
        self.w.toggle_aesthetics.stateChanged.connect(self.hideAesthetic_anim)
        self.w.toggle_embeddings.stateChanged.connect(self.hideEmbedding_anim)
        self.w.toggle_plugins.stateChanged.connect(self.hidePlugins_anim)
        self.w.toggle_colors.stateChanged.connect(self.hideColors_anim)
        self.w.toggle_grad.stateChanged.connect(self.hideGrad_anim)

        self.w.negative_prompts.setVisible(False)


        self.init_anims()
        self.initAnim.start()
        self.hide_all()


        self.ui_unicontrol = UniControl_UI(self)


        x = "W"

        getattr(self.w, x).setValue(15)
        print("QSlider" in str(getattr(self.w, x)))
        print("QCombobox" in str(getattr(self.w, x)))
        print("QTextEdit" in str(getattr(self.w, x)))

    def hide_all(self):
        self.showAll = False
        self.show_hide_all_anim()

    def hideColors_anim(self):
        if self.colHidden is True:
            self.showColAnim.start()
        else:
            self.hideColAnim.start()
        self.colHidden = not self.colHidden

    def hideGrad_anim(self):
        if self.graHidden is True:
            self.showGraAnim.start()
        else:
            self.hideGraAnim.start()
        self.graHidden = not self.graHidden

    def hideSampler_anim(self):
        if self.samHidden is True:
            self.showSamAnim.start()
        else:
            self.hideSamAnim.start()
        self.samHidden = not self.samHidden

    def hideAesthetic_anim(self):
        if self.aesHidden is True:
            self.showAesAnim.start()
        else:
            self.hideAesAnim.start()
        self.aesHidden = not self.aesHidden

    def hideAnimation_anim(self):
        if self.aniHidden is True:
            self.showAniAnim.start()
        else:
            self.hideAniAnim.start()
        self.aniHidden = not self.aniHidden


    def hidePlotting_anim(self):
        if self.ploHidden is True:
            self.showPloAnim.start()
        else:
            self.hidePloAnim.start()
        self.ploHidden = not self.ploHidden


    def hideOutput_anim(self):
        if self.opuHidden is True:
            self.showOpuAnim.start()
        else:
            self.hideOpuAnim.start()
        self.opuHidden = not self.opuHidden

    def hideInitImage_anim(self):
        if self.iniHidden is True:
            self.showIniAnim.start()
        else:
            self.hideIniAnim.start()
        self.iniHidden = not self.iniHidden

    def hideMaskImage_anim(self):
        if self.masHidden is True:
            self.showMasAnim.start()
        else:
            self.hideMasAnim.start()
        self.masHidden = not self.masHidden

    def hideOutpaint_anim(self):
        if self.outHidden is True:
            self.showOutAnim.start()
        else:
            self.hideOutAnim.start()
        self.outHidden = not self.outHidden

    def hideEmbedding_anim(self):
        if self.enbHidden is True:
            self.showEmbAnim.start()
        else:
            self.hideEmbAnim.start()
        self.enbHidden = not self.enbHidden

    def hidePlugins_anim(self):
        if self.pinHidden is True:
            self.showPinAnim.start()
        else:
            self.hidePinAnim.start()
        self.pinHidden = not self.pinHidden


    def show_hide_all_anim(self):
        print(self.showAll)
        if self.showAll == False:
            self.hideSamAnim.start()
            self.samHidden = True
            self.hideAesAnim.start()
            self.aesHidden = True
            self.hideAniAnim.start()
            self.aniHidden = True
            self.hidePloAnim.start()
            self.ploHidden = True

            self.hideColAnim.start()
            self.colHidden = True
            self.hideEmbAnim.start()
            self.enbHidden = True
            self.hideGraAnim.start()
            self.graHidden = True
            self.hideIniAnim.start()
            self.iniHidden = True
            self.hideMasAnim.start()
            self.masHidden = True
            self.hideOutAnim.start()
            self.outHidden = True
            self.hideOpuAnim.start()
            self.opuHidden = True
            self.hidePinAnim.start()
            self.pinHidden = True



            self.showAll = True
        elif self.showAll == True:
            self.showSamAnim.start()
            self.samHidden = False
            self.showAesAnim.start()
            self.aesHidden = False
            self.showAniAnim.start()
            self.aniHidden = False
            self.showPloAnim.start()
            self.ploHidden = False

            self.showColAnim.start()
            self.colHidden = False
            self.showEmbAnim.start()
            self.enbHidden = False
            self.showGraAnim.start()
            self.graHidden = False
            self.showIniAnim.start()
            self.intHidden = False
            self.showMasAnim.start()
            self.masHidden = False
            self.showOutAnim.start()
            self.outHidden = False
            self.showOpuAnim.start()
            self.opuHidden = False
            self.showPinAnim.start()
            self.pinHidden = False
            self.showAll = False

    def init_anims(self):
        self.showSamAnim = QtCore.QPropertyAnimation(self.w.sampler_values, b"maximumHeight")
        self.showSamAnim.setDuration(1500)
        self.showSamAnim.setStartValue(self.w.sampler_values.height())
        self.showSamAnim.setEndValue(self.w.height())
        self.showSamAnim.setEasingCurve(QEasingCurve.Linear)
        self.hideSamAnim = QtCore.QPropertyAnimation(self.w.sampler_values, b"maximumHeight")
        self.hideSamAnim.setDuration(500)
        self.hideSamAnim.setStartValue(self.w.sampler_values.height())
        self.hideSamAnim.setEndValue(0)
        self.hideSamAnim.setEasingCurve(QEasingCurve.Linear)

        self.showAesAnim = QtCore.QPropertyAnimation(self.w.aesthetic_values, b"maximumHeight")
        self.showAesAnim.setDuration(1500)
        self.showAesAnim.setStartValue(self.w.aesthetic_values.height())
        self.showAesAnim.setEndValue(self.w.height())
        self.showAesAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideAesAnim = QtCore.QPropertyAnimation(self.w.aesthetic_values, b"maximumHeight")
        self.hideAesAnim.setDuration(500)
        self.hideAesAnim.setStartValue(self.w.aesthetic_values.height())
        self.hideAesAnim.setEndValue(0)
        self.hideAesAnim.setEasingCurve(QEasingCurve.Linear)

        self.showAniAnim = QtCore.QPropertyAnimation(self.w.anim_values, b"maximumHeight")
        self.showAniAnim.setDuration(1500)
        self.showAniAnim.setStartValue(self.w.anim_values.height())
        self.showAniAnim.setEndValue(self.w.height())
        self.showAniAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideAniAnim = QtCore.QPropertyAnimation(self.w.anim_values, b"maximumHeight")
        self.hideAniAnim.setDuration(500)
        self.hideAniAnim.setStartValue(self.w.anim_values.height())
        self.hideAniAnim.setEndValue(0)
        self.hideAniAnim.setEasingCurve(QEasingCurve.Linear)

        self.showPloAnim = QtCore.QPropertyAnimation(self.w.plotting_frame, b"maximumHeight")
        self.showPloAnim.setDuration(1500)
        self.showPloAnim.setStartValue(self.w.plotting_frame.height())
        self.showPloAnim.setEndValue(self.w.height())
        self.showPloAnim.setEasingCurve(QEasingCurve.Linear)

        self.hidePloAnim = QtCore.QPropertyAnimation(self.w.plotting_frame, b"maximumHeight")
        self.hidePloAnim.setDuration(500)
        self.hidePloAnim.setStartValue(self.w.plotting_frame.height())
        self.hidePloAnim.setEndValue(0)
        self.hidePloAnim.setEasingCurve(QEasingCurve.Linear)

        self.showColAnim = QtCore.QPropertyAnimation(self.w.color_expo_values, b"maximumHeight")
        self.showColAnim.setDuration(1500)
        self.showColAnim.setStartValue(self.w.color_expo_values.height())
        self.showColAnim.setEndValue(self.w.height())
        self.showColAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideColAnim = QtCore.QPropertyAnimation(self.w.color_expo_values, b"maximumHeight")
        self.hideColAnim.setDuration(500)
        self.hideColAnim.setStartValue(self.w.color_expo_values.height())
        self.hideColAnim.setEndValue(0)
        self.hideColAnim.setEasingCurve(QEasingCurve.Linear)

        self.showEmbAnim = QtCore.QPropertyAnimation(self.w.embedding_values, b"maximumHeight")
        self.showEmbAnim.setDuration(1500)
        self.showEmbAnim.setStartValue(self.w.embedding_values.height())
        self.showEmbAnim.setEndValue(self.w.height())
        self.showEmbAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideEmbAnim = QtCore.QPropertyAnimation(self.w.embedding_values, b"maximumHeight")
        self.hideEmbAnim.setDuration(500)
        self.hideEmbAnim.setStartValue(self.w.embedding_values.height())
        self.hideEmbAnim.setEndValue(0)
        self.hideEmbAnim.setEasingCurve(QEasingCurve.Linear)

        self.showGraAnim = QtCore.QPropertyAnimation(self.w.grad_values, b"maximumHeight")
        self.showGraAnim.setDuration(1500)
        self.showGraAnim.setStartValue(self.w.grad_values.height())
        self.showGraAnim.setEndValue(self.w.height())
        self.showGraAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideGraAnim = QtCore.QPropertyAnimation(self.w.grad_values, b"maximumHeight")
        self.hideGraAnim.setDuration(500)
        self.hideGraAnim.setStartValue(self.w.grad_values.height())
        self.hideGraAnim.setEndValue(0)
        self.hideGraAnim.setEasingCurve(QEasingCurve.Linear)

        self.showIniAnim = QtCore.QPropertyAnimation(self.w.init_values, b"maximumHeight")
        self.showIniAnim.setDuration(1500)
        self.showIniAnim.setStartValue(self.w.init_values.height())
        self.showIniAnim.setEndValue(self.w.height())
        self.showIniAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideIniAnim = QtCore.QPropertyAnimation(self.w.init_values, b"maximumHeight")
        self.hideIniAnim.setDuration(500)
        self.hideIniAnim.setStartValue(self.w.init_values.height())
        self.hideIniAnim.setEndValue(0)
        self.hideIniAnim.setEasingCurve(QEasingCurve.Linear)

        self.showMasAnim = QtCore.QPropertyAnimation(self.w.mask_values, b"maximumHeight")
        self.showMasAnim.setDuration(1500)
        self.showMasAnim.setStartValue(self.w.mask_values.height())
        self.showMasAnim.setEndValue(self.w.height())
        self.showMasAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideMasAnim = QtCore.QPropertyAnimation(self.w.mask_values, b"maximumHeight")
        self.hideMasAnim.setDuration(500)
        self.hideMasAnim.setStartValue(self.w.mask_values.height())
        self.hideMasAnim.setEndValue(0)
        self.hideMasAnim.setEasingCurve(QEasingCurve.Linear)

        self.showOutAnim = QtCore.QPropertyAnimation(self.w.outpaint_values, b"maximumHeight")
        self.showOutAnim.setDuration(1500)
        self.showOutAnim.setStartValue(self.w.outpaint_values.height())
        self.showOutAnim.setEndValue(self.w.height())
        self.showOutAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideOutAnim = QtCore.QPropertyAnimation(self.w.outpaint_values, b"maximumHeight")
        self.hideOutAnim.setDuration(500)
        self.hideOutAnim.setStartValue(self.w.outpaint_values.height())
        self.hideOutAnim.setEndValue(0)
        self.hideOutAnim.setEasingCurve(QEasingCurve.Linear)

        self.showOpuAnim = QtCore.QPropertyAnimation(self.w.output_values, b"maximumHeight")
        self.showOpuAnim.setDuration(1500)
        self.showOpuAnim.setStartValue(self.w.output_values.height())
        self.showOpuAnim.setEndValue(self.w.height())
        self.showOpuAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideOpuAnim = QtCore.QPropertyAnimation(self.w.output_values, b"maximumHeight")
        self.hideOpuAnim.setDuration(500)
        self.hideOpuAnim.setStartValue(self.w.output_values.height())
        self.hideOpuAnim.setEndValue(0)
        self.hideOpuAnim.setEasingCurve(QEasingCurve.Linear)

        self.showPinAnim = QtCore.QPropertyAnimation(self.w.plugin_values, b"maximumHeight")
        self.showPinAnim.setDuration(1500)
        self.showPinAnim.setStartValue(self.w.plugin_values.height())
        self.showPinAnim.setEndValue(self.w.height())
        self.showPinAnim.setEasingCurve(QEasingCurve.Linear)

        self.hidePinAnim = QtCore.QPropertyAnimation(self.w.plugin_values, b"maximumHeight")
        self.hidePinAnim.setDuration(500)
        self.hidePinAnim.setStartValue(self.w.plugin_values.height())
        self.hidePinAnim.setEndValue(0)
        self.hidePinAnim.setEasingCurve(QEasingCurve.Linear)


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
        self.unicontrol.w.strength.setValue(self.unicontrol.w.strength_slider.value()/100)
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
        self.unicontrol.w.strength_slider.setValue(int(self.unicontrol.w.strength.value()*100))
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
