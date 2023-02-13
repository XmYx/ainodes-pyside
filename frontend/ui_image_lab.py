import codecs
import os
import re
import shutil

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6 import QtUiTools, QtCore, QtWidgets
from PySide6.QtCore import QObject, QFile, Signal, Slot, Qt, QPointF, QRect
from PySide6.QtGui import QPainter, QPixmap, QImage, QPen, QBrush
from PySide6.QtWidgets import QFileDialog, QGraphicsView, QGraphicsRectItem, QGraphicsItem, QGraphicsScene

import backend.interrogate
from backend.aestetics_score import get_aestetics_score
from backend.guess_prompt import get_prompt_guess_img
from backend.hypernetworks.modules import images
from backend.img2ascii import to_ascii
from backend.modelloader import load_upscaler
from backend.modelmerge import merge_models
from backend.sdv2.superresolution import run_sr
from backend.singleton import singleton
from backend.upscale import Upscale
from backend.watermark import add_watermark

#from volta_accelerate import convert_to_onnx, convert_to_trt
gs = singleton

interrogator = backend.interrogate.InterrogateModels("interrogate")

class MoveableRect(QGraphicsRectItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptedMouseButtons(Qt.RightButton)
        self.setCursor(Qt.OpenHandCursor)
        self.drag_start = QPointF()

class CheckerBoardScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(self.create_checker_board()))

    def create_checker_board(self):
        checker_board = QImage(64, 64, QImage.Format_RGB32)
        checker_board.fill(Qt.white)
        painter = QPainter(checker_board)
        painter.fillRect(0, 0, 32, 32, Qt.lightGray)
        painter.fillRect(32, 32, 32, 32, Qt.lightGray)
        return checker_board
class CropImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setUpdatesEnabled(True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self._drag_start = None
        self.is_rect_clicked = False
        self.crop_rect = None

    def update_rubber_band_(self):
        if self.crop_rect:
            rect = self.crop_rect.rect()
            scale = self.viewportTransform().m11()
            self.crop_rect.setGeometry(rect.x() / scale, rect.y() / scale, rect.width() / scale, rect.height() / scale)

    def add_image(self, image):
        self.scene.addItem(image)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
        if event.button() == Qt.RightButton:
            self._drag_rect_start = event.pos()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self._drag_start:
                offset = self._drag_start - event.pos()
                self.setSceneRect(self.sceneRect().translated(offset.x(), offset.y()))
                self._drag_start = event.pos()

        if event.buttons() & Qt.RightButton:
            d = event.pos() - self._drag_rect_start
            self.crop_rect.moveBy(d.x(), d.y())
            self._drag_rect_start = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = None
            self.is_rect_clicked = False
            self.setCursor(Qt.OpenHandCursor)
        if event.button() == Qt.RightButton:
            self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.2 if zoom_in else 1/1.2
        self.scale(factor, factor)



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
    image_text_ready = Signal(str)
    crop_image = Signal()
    show_crop_image = Signal()
    set_crop_image_scale = Signal()

class ImageLab():  # for signaling, could be a QWidget  too

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.signals = Callbacks()
        self.imageLab = ImageLab_ui()
        self.dropWidget = DropListView()
        self.dropWidget.setAccessibleName('fileList')
        self.dropWidget.fileDropped.connect(self.pictureDropped)
        self.imageLab.w.dropZone.addWidget(self.dropWidget)
        self.imageLab.w.startUpscale.clicked.connect(self.signal_start_upscale)
        self.imageLab.w.startImgToTxt.clicked.connect(self.signal_start_img_to_txt)
        self.imageLab.w.startWaterMark.clicked.connect(self.signal_start_watermark)
        self.imageLab.w.selectA.clicked.connect(self.selected_model_a)
        self.imageLab.w.selectB.clicked.connect(self.selected_model_b)
        self.imageLab.w.selectC.clicked.connect(self.selected_model_c)
        self.imageLab.w.Merge.clicked.connect(self.start_merge)
        self.imageLab.w.run_aestetic_prediction.clicked.connect(self.aestetic_prediction)
        self.imageLab.w.aestetics_prediction_output.clicked.connect(self.set_aestetic_prediction_output)

        self.imageLab.w.select_interrogation_output_folder.clicked.connect(self.set_interrogation_output_folder)
        self.imageLab.w.run_interrogation.clicked.connect(self.signal_run_interrogation)
        self.imageLab.w.upscale_20.clicked.connect(self.run_upscale_20)
        self.imageLab.w.select_watermark_output_folder.clicked.connect(self.set_watermark_output_folder)
        self.imageLab.w.set_crop_image_scale.clicked.connect(self.set_crop_image_scale_signal)
        self.imageLab.w.crop.clicked.connect(self.crop)
        self.imageLab.w.crop_size.currentIndexChanged.connect(self.set_crop_image_scale_signal)
        self.imageLab.w.next_image.clicked.connect(self.show_next_crop_image)
        self.prepare_cop_area()

    def prepare_cop_area(self):
        self.graphics_view = CropImageView()
        self.imageLab.w.crop_layout.addWidget(self.graphics_view)
        self.current_crop_index = 0
    def crop(self):
        self.signals.crop_image.emit()
    def crop_image(self):
        image = Image.open(self.fileList[self.current_crop_index])
        width, height = image.size
        scale_factor = self.imageLab.w.image_scale.value() / 100
        image = image.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.LANCZOS)
        rect = self.graphics_view.crop_rect.sceneBoundingRect()
        rect = QRect(rect.x(), rect.y(), rect.width(), rect.height())
        cropped_im = image.crop((rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()))
        crop_size = self.get_crop_size()
        cropped_im.resize((int(crop_size), int(crop_size)), resample=Image.LANCZOS)

        root, _ = os.path.splitext(self.fileList[self.current_crop_index])

        cropped_im.save(root + '_cropped.png')
    def show_next_image(self):
        # Open the next image in the list
        image = QImage(self.images[self.current_index])

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(image)

        # Set the pixmap to be displayed in the QLabel
        self.image_label.setPixmap(pixmap)

    def show_next_crop_image(self):
        if self.current_crop_index < len(self.fileList) - 1:
            self.current_crop_index += 1
            self.signals.show_crop_image.emit()
        else:
            self.parent.signals.status_update.emit('No more Images to crop')

    def get_crop_size(self):
        crop_size = 512
        if self.imageLab.w.crop_size.currentText() == '768x768':
            crop_size = 768
        elif self.imageLab.w.crop_size.currentText() == '1024x1024':
            crop_size = 1024
        return crop_size

    def create_crop_rect(self):
        crop_size = self.get_crop_size()
        self.graphics_view.crop_rect = MoveableRect(0, 0, crop_size, crop_size)
        self.graphics_view.crop_rect.setPen(QPen(Qt.red, 2))
        self.graphics_view.scene().addItem(self.graphics_view.crop_rect)


    def show_crop_image(self):
        image = QPixmap(self.fileList[self.current_crop_index])
        self.graphics_view.crop_rect = None
        self.graphics_view.setScene(QGraphicsScene())
        self.graphics_view.scene().addPixmap(image)
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.fitInView(self.graphics_view.sceneRect(), Qt.KeepAspectRatio)
        self.graphics_view.update()
        self.create_crop_rect()

    def set_crop_image_scale_signal(self):
        self.signals.set_crop_image_scale.emit()
    def set_crop_image_scale(self):

        im = Image.open(self.fileList[self.current_crop_index])
        width, height = im.size
        scale_factor = self.imageLab.w.image_scale.value() / 100

        im = im.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.LANCZOS)
        self.graphics_view.setScene(QGraphicsScene())
        self.graphics_view.scene().addPixmap(QPixmap.fromImage(ImageQt(im)))
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.fitInView(self.graphics_view.sceneRect(), Qt.KeepAspectRatio)
        self.create_crop_rect()


    def resize_image(self, image, new_size):
        image = image.resize(new_size, Image.LANCZOS)
        return image

    @Slot()
    def run_upscale_20_thread(self):
        self.parent.run_as_thread(self.run_upscale_20)


    @Slot()
    def run_volta_accel_thread(self):
        self.parent.run_as_thread(self.run_volta_accel)

    @Slot()
    def ebl_model_merge_start_thread(self):
        self.parent.run_as_thread(self.ebl_model_merge_start)

    @Slot()
    def run_aestetic_prediction_thread(self):
        self.parent.run_as_thread(self.run_aestetic_prediction)

    @Slot()
    def run_interrogation_thread(self):
        self.parent.run_as_thread(self.run_interrogation)

    @Slot()
    def img_to_text_start(self):
        self.parent.run_as_thread(self.run_img2txt)


    @Slot()
    def watermark_start(self):
        self.parent.run_as_thread(self.run_watermark)


    @Slot()
    def model_merge_start_thread(self):
        self.parent.run_as_thread(self.model_merge_start)



    @Slot()
    def upscale_start(self):
        self.parent.signals.status_update.emit("Upscale started...")
        self.upscale_thread()

    def upscale_stop(self):
        self.parent.signals.status_update.emit("Upscale finished...")

    def upscale_count(self, num):
        self.parent.signals.status_update.emit(f"Upscaled {str(num)} image(s)...")

    @Slot()
    def upscale_thread(self):
        self.parent.run_as_thread(self.run_upscale)


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
                    print('filename', filename)
                    if '.png' in filename:
                        try:
                            image = Image.open(filename).convert("RGB")
                        except Exception as e:
                            continue
                        image = images.resize_image(1, image, 512, 512)
                        interrogate_caption = ''
                        if interrogate:
                            print('type(image)', type(image))
                            interrogate_caption = interrogator.generate_caption(image)
                            print('interrogate_caption', interrogate_caption)
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


    def set_watermark_output_folder(self):
        watermark_output_folder = QFileDialog.getExistingDirectory()
        self.imageLab.w.watermark_output_folder.setText(watermark_output_folder)


    def set_interrogation_output_folder(self):
        interrogation_output_folder = QFileDialog.getExistingDirectory()
        self.imageLab.w.interrogation_output_folder.setText(interrogation_output_folder)

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
        out_folder = self.imageLab.w.aestetics_output_folder.text()
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


    def selected_model_a(self):
        self.modela = list(QFileDialog.getOpenFileName())
        self.imageLab.w.modelApath.setText(self.modela[0])


    def selected_model_b(self):
        self.modelb = list(QFileDialog.getOpenFileName())
        self.imageLab.w.modelBpath.setText(self.modelb[0])

    def selected_model_c(self):
        self.modelc = list(QFileDialog.getOpenFileName())
        self.imageLab.w.modelCpath.setText(self.modelc[0])


    def start_merge(self):
        self.signals.model_merge_start.emit()

    def start_ebl_merge(self):
        self.signals.ebl_model_merge_start.emit()

    def model_merge_start(self, progress_callback=None):

        model_2 = None if self.imageLab.w.modelCpath.text() == '' else self.imageLab.w.modelCpath.text()
        #interp_method=self.imageLab.w.merge_mode.currentText()
        if self.imageLab.w.modelCpath.text() != '':
            interp_method = 'Add difference'
        else:
            interp_method = 'Weighted sum'

        merge_models(model_0=self.imageLab.w.modelApath.text(), model_1=self.imageLab.w.modelBpath.text(),
                     model_2=model_2, multiplier=self.imageLab.w.alpha.value(),
                     output=self.imageLab.w.modelOutputName.text(),device=self.imageLab.w.device.currentText(),
                     safe_tensors=self.imageLab.w.save_as_safetensors.isChecked(),
                     interp_method=interp_method, save_as_half=self.imageLab.w.save_as_half.isChecked() )

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
        if len(self.fileList) > 0:
            self.imageLab.w.filesCount.display(str(len(self.fileList)))
            self.current_crop_index = 0
            self.signals.show_crop_image.emit()

    def upscale_count(self, num):
        pass
        #self.signals.upscale_counter.emit(num)


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
        self.imageLab.w.startUpscale.setEnabled(False)
        try:
            if len(self.fileList) > 0:
                if self.imageLab.w.ESRGAN.isChecked() or self.imageLab.w.GFPGAN.isChecked():
                    self.upscale.upscale_and_reconstruct(self.fileList,
                                                 upscale          = self.imageLab.w.ESRGAN.isChecked(),
                                                 upscale_scale    = self.imageLab.w.esrScale.value(),
                                                 upscale_strength = self.imageLab.w.esrStrength.value(),
                                                 use_gfpgan       = self.imageLab.w.GFPGAN.isChecked(),
                                                 strength         = self.imageLab.w.gfpStrength.value(),
                                                 image_callback   = None,
                                                 gfpgan_seed=int(self.imageLab.w.gfp_seed.text()))
        except:
            pass
        finally:
            self.imageLab.w.startUpscale.setEnabled(True)
            self.signals.upscale_stop.emit()

    def run_img2txt(self, progress_callback=None):
        grayscale = 0
        if self.imageLab.w.grayscaleType.isChecked():
            grayscale = 1

        if len(self.fileList) > 0:
            for path in self.fileList:
                ascii_image = to_ascii(path, self.imageLab.w.img2txtRatio.value()/100, grayscale)
                self.signals.image_text_ready.emit(ascii_image)

    def set_image_text(self, ascii_image):
        self.imageLab.w.image_text_output.setText(ascii_image)

    def run_watermark(self, progress_callback=None):
        text = self.imageLab.w.watermarkText.text()
        font_size = self.imageLab.w.fontSize.value()
        rotation = self.imageLab.w.rotation.value()
        pos_x = self.imageLab.w.pos_x.value()
        pos_y = self.imageLab.w.pos_y.value()
        fill = self.imageLab.w.fill.value()
        self.imageLab.w.startWaterMark.setEnabled(False)
        destination = self.imageLab.w.watermark_output_folder.text()
        try:
            if len(self.fileList) > 0:
                for path in self.fileList:
                    add_watermark(path=path, watermark=text, font_size=int(font_size),
                                  rotation=rotation, pos_x=pos_x, pos_y=pos_y,
                                    fill=fill, destination=destination)
        except Exception as e:
            print('Watermark failed due to: ', e)
            pass
        finally:
            self.imageLab.w.startWaterMark.setEnabled(True)