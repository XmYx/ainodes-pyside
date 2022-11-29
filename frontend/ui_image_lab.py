import os
import re
import shutil
import codecs
from types import SimpleNamespace

from PIL import Image
from PySide6 import QtUiTools, QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, QFile, Signal
from PySide6.QtWidgets import QFileDialog

from backend.modelloader import load_upscaler
from backend.singleton import singleton
from backend.upscale import Upscale
from backend.img2ascii import to_ascii
from backend.watermark import add_watermark
from backend.modelmerge import merge_models, merge_ebl_model
from backend.aestetics_score import get_aestetics_score
import backend.interrogate
from backend.guess_prompt import get_prompt_guess_img
from backend.hypernetworks.modules import images
from backend.sdv2.superresolution import run_sr
#from volta_accelerate import convert_to_onnx, convert_to_trt
gs = singleton

interrogator = backend.interrogate.InterrogateModels("interrogate")

class ImageLab_ui(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/img_lab.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class DropListView(QtWidgets.QListWidget):

    fileDropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(DropListView, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.fileDropped.emit(links)
        else:
            event.ignore()

class Callbacks(QObject):
    upscale_start = Signal()
    upscale_stop = Signal()
    upscale_counter = Signal(int)
    img_to_txt_start = Signal()
    watermark_start = Signal()
    model_merge_start = Signal()
    ebl_model_merge_start = Signal()
    run_aestetic_prediction = Signal()
    run_interrogation = Signal()
    run_volta_accel = Signal()
    run_upscale_20 = Signal()

class ImageLab():  # for signaling, could be a QWidget  too

    def __init__(self):
        super().__init__()
        self.signals = Callbacks()
        self.imageLab = ImageLab_ui()
        self.dropWidget = DropListView()
        self.dropWidget.setAccessibleName('fileList')
        self.dropWidget.fileDropped.connect(self.pictureDropped)
        self.imageLab.w.dropZone.addWidget(self.dropWidget)
        self.imageLab.w.startUpscale.clicked.connect(self.signal_start_upscale)
        self.imageLab.w.startImgToTxt.clicked.connect(self.signal_start_img_to_txt)
        self.imageLab.w.startWaterMark.clicked.connect(self.signal_start_watermark)
        self.imageLab.w.selectA.clicked.connect(self.select_model_a)
        self.imageLab.w.selectB.clicked.connect(self.select_model_b)
        self.imageLab.w.Merge.clicked.connect(self.start_merge)
        self.imageLab.w.MergeEBL.clicked.connect(self.start_ebl_merge)
        self.imageLab.w.run_aestetic_prediction.clicked.connect(self.aestetic_prediction)
        self.imageLab.w.aestetics_prediction_output.clicked.connect(self.set_aestetic_prediction_output)
        self.imageLab.w.alpha.valueChanged.connect(self.update_alpha)
        self.imageLab.w.alphaNew.valueChanged.connect(self.update_alpha)
        self.imageLab.w.select_interrogation_output_folder.clicked.connect(self.set_interrogation_output_folder)
        self.imageLab.w.run_interrogation.clicked.connect(self.signal_run_interrogation)
        self.imageLab.w.select_model.clicked.connect(self.select_accel_model)
        self.imageLab.w.run_volta_accel.clicked.connect(self.signal_run_volta_accel)
        self.imageLab.w.upscale_20.clicked.connect(self.run_upscale_20)



    def run_upscale_20(self, progress_callback=False):
        run_sr(image_list=self.fileList,
               target_h=self.imageLab.w.upscale_h.value(),
               target_w=self.imageLab.w.upscale_w.value(),
               prompt=self.imageLab.w.textEdit.toPlainText(),
               seed=self.imageLab.w.seed.text(),
               num_samples=self.imageLab.w.samples.value(),
               scale=self.imageLab.w.scale.value(),
               steps=self.imageLab.w.steps.value(),
               eta=self.imageLab.w.ddim_eta.value(),
               noise_level=self.imageLab.w.noise_level.value()
        )

    # todo get this working on linux boxes
    def run_volta_accel(self, progress_callback=False):
        args = {}
        args['model_path'] = self.imageLab.w.accel_path.text()
        args = SimpleNamespace(**args)
        args.image_size = (512, 512)
        args.max_seq_length = 77
        args.max_gpu_memory = self.imageLab.w.max_gpu_memory.value()

        convert_to_onnx(args)
        convert_to_trt(args)

    def signal_run_volta_accel(self):
        self.signals.run_volta_accel.emit()

    def select_accel_model(self):
        accel_model_filename = QFileDialog.getOpenFileName()[0]
        self.imageLab.w.accel_path.setText(accel_model_filename)


    def signal_run_interrogation(self):
        self.signals.run_interrogation.emit()

    def run_interrogation(self, progress_callback=False):
        keep_folder_structure = self.imageLab.w.keep_folder_structure.isChecked()
        copy_image = self.imageLab.w.copy_image.isChecked()
        copy_info_text = self.imageLab.w.copy_info_text.isChecked()
        interrogate = self.imageLab.w.interrogate.isChecked()
        guess_prompt = self.imageLab.w.guess_prompt.isChecked()
        matcher = re.compile(r'(.*?)(\..*)')
        if interrogate:
            interrogator.load()
        if interrogate or guess_prompt:
            if len(self.fileList) > 0:
                for filename in self.fileList:
                    out_folder = self.imageLab.w.interrogation_output_folder.text()
                    print(filename)
                    if '.png' in filename:
                        try:
                            image = Image.open(filename).convert("RGB")
                        except Exception as e:
                            continue
                        image = images.resize_image(1, image, 512, 512)
                        interrogate_caption = ''
                        if interrogate:
                            print(type(image))
                            interrogate_caption = interrogator.generate_caption(image)
                            print(interrogate_caption)
                        prompt_guess = ''
                        if guess_prompt:
                            prompt_guess = get_prompt_guess_img(image)

                        drive, path = os.path.splitdrive(filename)
                        raw_file_path, raw_filename = os.path.split(path)
                        raw_file_path = re.sub(r'.', '', raw_file_path, count = 1)
                        raw_only_filename = matcher.match(raw_filename)[1]
                        if keep_folder_structure:
                            out_folder = os.path.join(out_folder, raw_file_path)
                        os.makedirs(out_folder, exist_ok=True)
                        if interrogate:
                            f = codecs.open(os.path.join(out_folder, raw_only_filename + '_interrogate.txt'), 'w', "utf-8")
                            f.write(interrogate_caption)
                            f.close()
                        if guess_prompt:
                            f = codecs.open(os.path.join(out_folder, raw_only_filename + '_prompt_guess.txt'), 'w', "utf-8")
                            f.write(prompt_guess)
                            f.close()
                        if copy_image:
                            shutil.copyfile(filename, os.path.join(out_folder, raw_filename))
                        if copy_info_text:
                            src_raw_file_path, src_raw_filename = os.path.split(path)
                            src_raw_only_filename = matcher.match(src_raw_filename)[1]
                            src = os.path.join(src_raw_file_path, src_raw_only_filename + '_settings.txt')
                            if os.path.isfile(src):
                                dst = os.path.join(out_folder, raw_only_filename + '_settings.txt')
                                shutil.copyfile(src, dst)
                            src = os.path.join(src_raw_file_path, src_raw_only_filename + '.yaml')
                            if os.path.isfile(src):
                                dst = os.path.join(out_folder, raw_only_filename + '_settings.txt')
                                shutil.copyfile(src, dst)

        else:
            print('nothing selected to interrogate')
        if 'clip' in gs.models:
            del gs.models['clip']





    def set_interrogation_output_folder(self):
        interrogation_output_folder = QFileDialog.getExistingDirectory()
        self.imageLab.w.interrogation_output_folder.setText(interrogation_output_folder)


    def update_alpha(self):
        self.imageLab.w.alphaNumber.display(str(self.imageLab.w.alpha.value() / 10))
        self.imageLab.w.alphaNewNumber.display(str(self.imageLab.w.alphaNew.value() / 10))


    def aestetic_prediction(self):
        self.signals.run_aestetic_prediction.emit()
        #self.run_aestetic_prediction()

    def set_aestetic_prediction_output(self):
        aestetic_prediction_output_directory = QFileDialog.getExistingDirectory()
        self.imageLab.w.aestetics_output_folder.setText(aestetic_prediction_output_directory)

    def run_aestetic_prediction(self, progress_callback=False):
        print('Aestetics calculation started')
        matcher = re.compile(r'(.*?)(\..*)')
        aesthetics_keep_folder_structure = self.imageLab.w.aesthetics_keep_folder_structure.isChecked()
        if len(self.fileList) > 0:
            for file in self.fileList:
                score = get_aestetics_score(file)
                if score[0][0] > self.imageLab.w.min_aestetics_level.value():
                    # "{:.1f}".format(number)
                    if aesthetics_keep_folder_structure:
                        out_folder = self.imageLab.w.aestetics_output_folder.text()
                    out_folder = os.path.join(out_folder, "{:.1f}".format(float(score[0][0])))
                    os.makedirs(out_folder, exist_ok=True)
                    filename = os.path.basename(file)
                    srcpath = os.path.dirname(os.path.abspath(file))
                    dst = os.path.join(out_folder, filename)
                    purename = matcher.match(filename)
                    txtfile = purename[1] + '_settings.txt'
                    shutil.copyfile(file, dst)
                    txtsrc = os.path.join(srcpath,txtfile)
                    if os.path.exists(txtsrc):
                        txtdst = os.path.join(out_folder, txtfile)
                        shutil.copyfile(txtsrc, txtdst)
                    txtfile = filename + '_settings.txt'
                    shutil.copyfile(file, dst)
                    txtsrc = os.path.join(srcpath,txtfile)
                    if os.path.exists(txtsrc):
                        txtdst = os.path.join(out_folder, txtfile)
                        shutil.copyfile(txtsrc, txtdst)
        print('Aestetics calculation finished')


    def select_model_a(self):
        self.modela = list(QFileDialog.getOpenFileName())
        print(list(self.modela))
        print(type(self.modela))
        self.imageLab.w.modelApath.setText(self.modela[0])

    def selectDirectory(self):

        selected_directory = QFileDialog.getExistingDirectory()

        # Use the selected directory...
        print('selected_directory:', selected_directory)

    def select_model_b(self):
        self.modelb = list(QFileDialog.getOpenFileName())
        self.imageLab.w.modelBpath.setText(self.modelb[0])

    def start_merge(self):
        self.signals.model_merge_start.emit()

    def start_ebl_merge(self):
        self.signals.ebl_model_merge_start.emit()

    def model_merge_start(self, progress_callback=None):
        merge_models(self.modela[0], self.modelb[0], self.imageLab.w.alpha.value() / 10, self.imageLab.w.modelOutputName.text(),self.imageLab.w.device.currentText())

    def ebl_model_merge_start(self, progress_callback=None):
        merge_ebl_model(src=self.modela[0], dst=self.modelb[0],output=self.imageLab.w.modelOutputName.text(), alpha=self.imageLab.w.alpha.value() / 10, alpha_new=self.imageLab.w.alphaNew.value() / 10)

    def signal_start_upscale(self):
        self.signals.upscale_start.emit()

    def signal_start_img_to_txt(self):
        self.signals.img_to_txt_start.emit()

    def signal_start_watermark(self):
        self.signals.watermark_start.emit()

    def show(self):
        self.imageLab.w.show()

    def pictureDropped(self, l):
        self.fileList = []
        self.dropWidget.clear()
        for path in l:
            if os.path.isdir(path):
                result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']
                for file in result:
                    self.fileList.append(file)
                    self.dropWidget.insertItem(0, file)
            elif os.path.isfile(path):
                self.dropWidget.insertItem(0, path)
                self.fileList.append(path)
        self.imageLab.w.filesCount.display(str(len(self.fileList)))

    def upscale_count(self, num):
        self.signals.upscale_counter.emit(num)


    def run_upscale(self, progress_callback=None):
        self.upscale = Upscale()
        self.upscale.signals.upscale_counter.connect(self.upscale_count)
        model_name=''
        if self.imageLab.w.ESRGAN.isChecked():
            if self.imageLab.w.RealESRGAN.isChecked():
                model_name = 'RealESRGAN_x4plus'
            if self.imageLab.w.RealESRGANAnime.isChecked():
                model_name = 'RealESRGAN_x4plus_anime_6B'

            load_upscaler(self.imageLab.w.GFPGAN.isChecked(), self.imageLab.w.ESRGAN.isChecked(), model_name)
        if len(self.fileList) > 0:
            if self.imageLab.w.ESRGAN.isChecked() or self.imageLab.w.GFPGAN.isChecked():
                self.upscale.upscale_and_reconstruct(self.fileList,
                                             upscale          = self.imageLab.w.ESRGAN.isChecked(),
                                             upscale_scale    = self.imageLab.w.esrScale.value(),
                                             upscale_strength = self.imageLab.w.esrStrength.value()/100,
                                             use_gfpgan       = self.imageLab.w.GFPGAN.isChecked(),
                                             strength         = self.imageLab.w.gfpStrength.value()/100,
                                             image_callback   = None)

        self.signals.upscale_stop.emit()

    def run_img2txt(self, progress_callback=None):
        grayscale = 0
        if self.imageLab.w.grayscaleType.isChecked():
            grayscale = 1

        if len(self.fileList) > 0:
            for path in self.fileList:
                to_ascii(path, self.imageLab.w.img2txtRatio.value()/100, grayscale)

    def run_watermark(self, progress_callback=None):
        text = self.imageLab.w.watermarkText.text()
        font_size = self.imageLab.w.fontSize.value()
        if len(self.fileList) > 0:
            for path in self.fileList:
                add_watermark(path=path, watermark=text, font_size=int(font_size))
