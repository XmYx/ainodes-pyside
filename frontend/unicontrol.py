import os

from PySide6 import QtUiTools, QtCore
from PySide6.QtCore import QFile, QObject, QEasingCurve, QRect, Signal
from PySide6.QtWidgets import QFileDialog

from backend.singleton import singleton
from backend.torch_gc import torch_gc
from backend.sqlite import model_db_civitai
gs = singleton

class Callbacks(QObject):
    model_changed = Signal(str)

class UniControl(QObject):

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        self.signals = Callbacks()
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/unicontrol.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        self.initAnimation()

        self.w.toggle_sampler.stateChanged.connect(self.hideSampler_anim)
        self.w.show_output_setup.stateChanged.connect(self.hideOutput_anim)
        self.w.show_init_setup.stateChanged.connect(self.hideInitImage_anim)
        self.w.show_mask_setup.stateChanged.connect(self.hideMaskImage_anim)
        self.w.toggle_outpaint.stateChanged.connect(self.hideOutpaint_anim)
        self.w.toggle_animations.stateChanged.connect(self.hideAnimation_anim)
        self.w.toggle_plotting.stateChanged.connect(self.hidePlotting_anim)
        self.w.toggle_multi_model_batch.stateChanged.connect(self.hideMml_anim)
        self.w.toggle_aesthetics.stateChanged.connect(self.hideAesthetic_anim)
        self.w.toggle_embeddings.stateChanged.connect(self.hideEmbedding_anim)
        self.w.toggle_plugins.stateChanged.connect(self.hidePlugins_anim)
        self.w.toggle_colors.stateChanged.connect(self.hideColors_anim)
        self.w.toggle_grad.stateChanged.connect(self.hideGrad_anim)
        self.w.toggle_negative_prompt.toggled.connect(self.toggle_n_prompt)
        self.w.seamless.toggled.connect(self.toggle_seamless)
        self.w.update_models.clicked.connect(self.update_model_list)
        self.w.update_vae.clicked.connect(self.update_vae_list)
        self.w.update_hyper.clicked.connect(self.update_hypernetworks_list)
        self.w.update_aesthetics.clicked.connect(self.update_aesthetics_list)
        self.w.stop_dream.clicked.connect(self.stop_all)
        self.w.selected_model.currentIndexChanged.connect(self.select_new_model)
        self.w.selected_vae.currentIndexChanged.connect(self.select_new_vae)
        self.w.selected_hypernetwork.currentIndexChanged.connect(self.select_new_hypernetwork)
        self.w.selected_aesthetic_embedding.currentIndexChanged.connect(self.select_new_aesthetic_embedding)
        self.w.select_input_video.clicked.connect(self.select_input_video)
        self.w.select_mask_video.clicked.connect(self.select_mask_video)
        self.w.select_init_image.clicked.connect(self.select_init_image)
        self.w.select_mask_image.clicked.connect(self.select_mask_image)
        self.w.select_color_match_image.clicked.connect(self.select_color_match_image)
        self.w.select_mse_init.clicked.connect(self.select_mse_init)
        self.w.select_colormatch_preview.clicked.connect(self.select_colormatch_preview)

        self.w.negative_prompts.setVisible(False)
        self.init_anims()
        self.initAnim.start()
        self.hide_all()
        self.ui_unicontrol = UniControl_UI(self)
        self.civitai_api = model_db_civitai.civit_ai_api()

    def select_init_image(self):
        filename = QFileDialog.getOpenFileName(caption='Select Init image', filter='Image (*.png *.jpg)')
        self.w.init_image.setText(filename[0])

    def select_mse_init(self):
        filename = QFileDialog.getOpenFileName(caption='Select mse image', filter='Image (*.png *.jpg)')
        self.w.init_mse_image.setText(filename[0])

    def select_colormatch_preview(self):
        filename = QFileDialog.getOpenFileName(caption='Select colormatch preview', filter='Image (*.png *.jpg)')
        self.w.colormatch_preview.setText(filename[0])

    def select_color_match_image(self):
        filename = QFileDialog.getOpenFileName(caption='Select colormatch image', filter='Image (*.png *.jpg)')
        self.w.colormatch_image.setText(filename[0])

    def select_mask_image(self):
        filename = QFileDialog.getOpenFileName(caption='Select Init image', filter='Image (*.png *.jpg)')
        self.w.mask_file.setText(filename[0])

    def select_input_video(self):
        filename = QFileDialog.getOpenFileName(caption='Select input video', filter='Video (*.mp4 *.avi)')
        self.w.video_init_path.setText(filename[0])

    def select_mask_video(self):
        filename = QFileDialog.getOpenFileName(caption='Select mask video', filter='Video (*.mp4 *.avi)')
        self.w.video_mask_path.setText(filename[0])


    def set_prompt(self, prompt):
        self.w.prompts.setPlainText(prompt)

    def stop_all(self):
        gs.stop_all = True

    def add_to_model_list(self, models):
        for model in models:
            if '.ckpt' in model:
                self.w.selected_model.addItem(model)

    def update_vae_list(self):
        item_count = self.w.selected_vae.count()
        model_items = []
        current_vae = None
        if item_count > 0:
            current_text = self.w.selected_vae.currentText()
            current_vae = current_text if current_text != '' else None
        files = os.listdir(gs.system.vae_dir)
        self.w.selected_vae.clear()
        self.w.selected_vae.addItem('None')
        for model in files:
            if '.ckpt' in model:
                self.w.selected_vae.addItem(model)
        item_count = self.w.selected_vae.count()
        model_items = []
        for i in range(0, item_count-1):
            model_items.append(self.w.selected_vae.itemText(i))
        print('item_count', item_count)
        current_vae = 'None' if current_vae == None else current_vae
        if current_vae != 'None':
            self.w.selected_vae.setCurrentIndex(model_items.index(current_vae))
        else:
            self.w.selected_vae.setCurrentIndex(0)

    def select_new_vae(self):
        current_text = self.w.selected_vae.currentText()
        new_vae = 'None'
        if current_text != 'None':
            new_vae = os.path.join(gs.system.vae_path, current_text)
        gs.diffusion.selected_vae = new_vae

    def select_new_hypernetwork(self):
        current_text = self.w.selected_hypernetwork.currentText()
        new_hyper_net = 'None'
        if current_text != 'None':
            new_hyper_net = os.path.join(gs.system.hypernetwork_dir, current_text)
        gs.diffusion.selected_hypernetwork = new_hyper_net

    def select_new_aesthetic_embedding(self):
        current_text = self.w.selected_aesthetic_embedding.currentText()
        new_aesthetic_embedding = 'None'
        if current_text != 'None':
            new_aesthetic_embedding = os.path.join(gs.system.aesthetic_gradients_dir, current_text)
        gs.diffusion.selected_aesthetic_embedding = new_aesthetic_embedding



    def update_hypernetworks_list(self):
        item_count = self.w.selected_hypernetwork.count()
        model_items = []
        current_hypernet = None
        if item_count > 0:
            current_text = self.w.selected_hypernetwork.currentText()
            current_hypernet = current_text if current_text != '' else None
        files = os.listdir(gs.system.hypernetwork_dir)
        self.w.selected_hypernetwork.clear()
        self.w.selected_hypernetwork.addItem('None')
        for model in files:
            if '.pt' in model:
                self.w.selected_hypernetwork.addItem(model)
        item_count = self.w.selected_hypernetwork.count()
        model_items = []
        for i in range(0, item_count):
            model_items.append(self.w.selected_hypernetwork.itemText(i))
        current_hypernet = 'None' if current_hypernet == None else current_hypernet
        if current_hypernet != 'None':
            self.w.selected_hypernetwork.setCurrentIndex(model_items.index(current_hypernet))
        else:
            self.w.selected_hypernetwork.setCurrentIndex(0)

    def update_aesthetics_list(self):
        item_count = self.w.selected_aesthetic_embedding.count()
        model_items = []
        current_aesthetic_embedding = None
        if item_count > 0:
            current_text = self.w.selected_aesthetic_embedding.currentText()
            current_aesthetic_embedding = current_text if current_text != '' else None
        files = os.listdir(gs.system.aesthetic_gradients_dir)
        self.w.selected_aesthetic_embedding.clear()
        self.w.selected_aesthetic_embedding.addItem('None')
        for model in files:
            if '.pt' in model:
                self.w.selected_aesthetic_embedding.addItem(model)
        item_count = self.w.selected_aesthetic_embedding.count()
        model_items = []
        for i in range(0, item_count):
            model_items.append(self.w.selected_aesthetic_embedding.itemText(i))
        current_aesthetic_embedding = 'None' if current_aesthetic_embedding == None else current_aesthetic_embedding
        if current_aesthetic_embedding != 'None':
            self.w.selected_aesthetic_embedding.setCurrentIndex(model_items.index(current_aesthetic_embedding))
        else:
            self.w.selected_aesthetic_embedding.setCurrentIndex(0)



    def update_model_list(self):

        if not os.path.isfile(gs.system.sd_model_file):
            target_model = None
        else:
            if 'custom' in gs.system.sd_model_file:
                target_model = 'custom/' + os.path.basename(gs.system.sd_model_file) # to work around the signal which triggers once we start changing the dropdowns items
            else:
                target_model = os.path.basename(gs.system.sd_model_file)

        self.w.selected_model.clear()
        files = os.listdir(gs.system.models_path)
        files = [f for f in files if os.path.isfile(gs.system.models_path+'/'+f)] #Filtering only the files.
        model_items = files
        for model in files:
            if '.ckpt' in model or 'safetensors' in model:
                self.w.selected_model.addItem(model)
                self.w.multi_model_batch_list.addItem(model)
        files = os.listdir(gs.system.custom_models_dir)
        files = [f for f in files if os.path.isfile(gs.system.custom_models_dir + '/' +f)] #Filtering only the files.
        model_items.append(files)
        for model in files:
            if '.ckpt' in model or 'safetensors' in model:
                self.w.selected_model.addItem('custom/' + model)
                self.w.multi_model_batch_list.addItem('custom/' + model)
        item_count = self.w.selected_model.count()
        model_items = []
        for i in range(0, item_count):
            model_items.append(self.w.selected_model.itemText(i))
        if target_model is None:
            if item_count > 0:
                print('model from config does not exist therefore we choose first model from the loaded list')
                self.w.selected_model.setCurrentIndex(0)
            else:
                print(f'you have no models installed in {gs.system.models_path} please install any model before you run this software, you can try to download a model using the download feature')
        else:
            if item_count > 0:
                self.w.selected_model.setCurrentIndex(model_items.index(target_model))
            else:
                self.w.selected_model.setCurrentIndex(0)


    def select_new_model(self):
        new_model = os.path.join(gs.system.models_path,self.w.selected_model.currentText())
        gs.system.sd_model_file = new_model
        if 'sd' in gs.models:
            del gs.models['sd']
        torch_gc()

    def toggle_seamless(self):
        self.w.axis.setVisible(self.w.seamless.isChecked())

    def toggle_n_prompt(self):
        self.w.negative_prompts.setVisible(self.w.toggle_negative_prompt.isChecked())

    def hide_all(self):
        self.init_anims()
        self.showAll = False
        self.show_hide_all_anim()

    def hideColors_anim(self):
        self.init_anims()
        if self.colHidden is True:
            self.showColAnim.start()
        else:
            self.hideColAnim.start()
        self.colHidden = not self.colHidden

    def hideGrad_anim(self):
        self.init_anims()
        if self.graHidden is True:
            self.showGraAnim.start()
        else:
            self.hideGraAnim.start()
        self.graHidden = not self.graHidden

    def hideSampler_anim(self):
        self.init_anims()
        if self.samHidden is True:
            self.showSamAnim.start()
        else:
            self.hideSamAnim.start()
        self.samHidden = not self.samHidden

    def hideAesthetic_anim(self):
        self.init_anims()
        if self.aesHidden is True:
            self.showAesAnim.start()
        else:
            self.hideAesAnim.start()
        self.aesHidden = not self.aesHidden

    def hideAnimation_anim(self):
        self.init_anims()
        if self.aniHidden is True:
            self.showAniAnim.start()
            #self.parent.timeline.show_anim_action()
            self.parent.widgets[self.parent.current_widget].w.keyframes.setVisible(True)
        else:
            self.hideAniAnim.start()
            #self.parent.timeline.hide_anim_action()
            self.parent.widgets[self.parent.current_widget].w.keyframes.setVisible(False)
        self.aniHidden = not self.aniHidden


    def hidePlotting_anim(self):
        self.init_anims()
        if self.ploHidden is True:
            self.showPloAnim.start()
        else:
            self.hidePloAnim.start()
        self.ploHidden = not self.ploHidden

    def hideMml_anim(self):
        self.init_anims()
        if self.mmlHidden is True:
            self.showMmlAnim.start()
        else:
            self.hideMmlAnim.start()
        self.mmlHidden = not self.mmlHidden

    def hideOutput_anim(self):
        self.init_anims()
        if self.opuHidden is True:
            self.showOpuAnim.start()
        else:
            self.hideOpuAnim.start()
        self.opuHidden = not self.opuHidden

    def hideInitImage_anim(self):
        self.init_anims()
        if self.iniHidden is True:
            self.showIniAnim.start()
        else:
            self.hideIniAnim.start()
        self.iniHidden = not self.iniHidden

    def hideMaskImage_anim(self):
        self.init_anims()
        if self.masHidden is True:
            self.showMasAnim.start()
        else:
            self.hideMasAnim.start()
        self.masHidden = not self.masHidden

    def hideOutpaint_anim(self):
        self.init_anims()
        if self.outHidden is True:
            self.showOutAnim.start()
            self.parent.thumbs.w.dockWidget.setVisible(True)
            self.w.preview_mode.setCurrentIndex(1)
            self.parent.secondary_toolbar.setVisible(True)
        else:
            self.hideOutAnim.start()
            self.parent.thumbs.w.dockWidget.setVisible(False)
            self.parent.secondary_toolbar.setVisible(False)
        self.outHidden = not self.outHidden

    def hideEmbedding_anim(self):
        self.init_anims()
        if self.enbHidden is True:
            self.showEmbAnim.start()
        else:
            self.hideEmbAnim.start()
        self.enbHidden = not self.enbHidden

    def hidePlugins_anim(self):
        self.init_anims()
        if self.pinHidden is True:
            self.showPinAnim.start()
        else:
            self.hidePinAnim.start()
        self.pinHidden = not self.pinHidden


    def show_hide_all_anim(self):
        print('self.showAll', self.showAll)
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
            self.hideMmlAnim.start()
            self.mmlHidden = True
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
            self.hideMmlAnim.start()
            self.mmlHidden = False
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

        self.showMmlAnim = QtCore.QPropertyAnimation(self.w.multi_model_batch, b"maximumHeight")
        self.showMmlAnim.setDuration(1500)
        self.showMmlAnim.setStartValue(self.w.multi_model_batch.height())
        self.showMmlAnim.setEndValue(self.w.height())
        self.showMmlAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideMmlAnim = QtCore.QPropertyAnimation(self.w.multi_model_batch, b"maximumHeight")
        self.hideMmlAnim.setDuration(500)
        self.hideMmlAnim.setStartValue(self.w.multi_model_batch.height())
        self.hideMmlAnim.setEndValue(0)
        self.hideMmlAnim.setEasingCurve(QEasingCurve.Linear)
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
