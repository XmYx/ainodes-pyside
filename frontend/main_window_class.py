import random
import time

import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
# from PyQt6 import QtCore as qtc
# from PyQt6 import QtWidgets as qtw
# from PyQt6 import uic
# from PyQt6.Qt import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import *
from einops import rearrange

from ui_classes import *

from frontend.node_window_class import NodeWindow
from frontend.worker_class import Worker
from backend.singleton import singleton
gs = singleton
from ldm.generate import Generate

gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

class GenerateWindow(QWidget):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/main/main_window.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, threadpool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threadpool = threadpool



        self.image_path = ""

        #print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        #uic.loadUi("frontend/main/main_window.ui", self)



        self.home()
        self.w.statusBar().showMessage('Ready')
        self.w.progressBar = QProgressBar()


        self.w.statusBar().addPermanentWidget(self.w.progressBar)

        # This is simply to show the bar
        self.w.progressBar.setGeometry(30, 40, 200, 25)
        self.w.progressBar.setValue(50)

        self.nodeWindow = NodeWindow()
        self.load_history()
        #self.show_anim()

        self.w.actionAnim.triggered.connect(self.show_anim)
        self.w.actionPreview.triggered.connect(self.show_preview)
        self.w.actionPrompt.triggered.connect(self.show_prompt)
        #self.actionRunControl.triggered.connect(self.show_runner)
        self.w.actionSampler.triggered.connect(self.show_sampler)
        self.w.actionSliders.triggered.connect(self.show_sizer_count)
        self.w.actionThumbnails.triggered.connect(self.show_thumbnails)

        #self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        #self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        #self.pix_map_item = self.preview.scene.addPixmap(self.pix_map)
        """self.global_factor = 1
        self.pix_map_item = QGraphicsPixmapItem()

    def scaleImage(self, factor):
        _pixmap = self.pic.scaledToHeight(int(factor*self.viewport().geometry().height()), Qt.SmoothTransformation)
        self.pix_map_item.setPixmap(_pixmap)
        self.preview.scene.setSceneRect(QRectF(_pixmap.rect()))

    def wheelEvent(self, event):
        factor = 1.5

        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            self.centerOn(scene_pos)

            if event.angleDelta().y() > 0 and self.global_factor < 20:
                self.global_factor *= factor
                self.scaleImage(self.global_factor)
            elif event.angleDelta().y() < 0 and self.global_factor > 0.2:
                self.global_factor /= factor
                self.scaleImage(self.global_factor)
        else:
            return super().wheelEvent(event)"""

    def home(self):

        self.w.preview = Preview()
        self.w.sizer_count = SizerCount()
        self.w.sampler = Sampler()
        #self.runner = Runner()
        self.w.anim = Anim()
        self.w.prompt = Prompt()
        self.w.dynaview = Dynaview()
        self.w.dynaimage = Dynaimage()

        self.w.thumbnails = Thumbnails()

        #app2  = qapp(sys.argv)
        #self.nodes = NodeEditorWindow()
        #self.nodes.nodeeditor.addNodes()

        #wnd.show()

        self.w.thumbnails.w.thumbs.itemClicked.connect(self.viewImageClicked)
        #self.thumbnails.thumbs.addItem(QListWidgetItem(QIcon('frontend/main/splash.png'), "Earth"))



        self.w.sizer_count.w.heightNumber.display(str(self.w.sizer_count.w.heightSlider.value()))
        self.w.sizer_count.w.widthNumber.display(str(self.w.sizer_count.w.widthSlider.value()))
        self.w.sizer_count.w.samplesNumber.display(str(self.w.sizer_count.w.samplesSlider.value()))
        self.w.sizer_count.w.batchSizeNumber.display(str(self.w.sizer_count.w.batchSizeSlider.value()))
        self.w.sizer_count.w.stepsNumber.display(str(self.w.sizer_count.w.stepsSlider.value()))
        self.w.sizer_count.w.scaleNumber.display(str(self.w.sizer_count.w.scaleSlider.value()))



        self.w.setCentralWidget(self.w.preview.w)

        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sampler.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sizer_count.w.dockWidget)

        #self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.runner)
        #self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.anim.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.w.prompt.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.thumbnails.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.dynaview.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.dynaimage.w.dockWidget)
        self.w.dynaview.w.setMinimumSize(QtCore.QSize(512, 512))


        self.w.tabifyDockWidget(self.w.thumbnails.w.dockWidget, self.w.sampler.w.dockWidget)

        self.w.thumbnails.w.dockWidget.setWindowTitle('Thumbnails')
        self.w.sampler.w.dockWidget.setWindowTitle('Sampler')
        self.w.sizer_count.w.dockWidget.setWindowTitle('Sliders')
        self.w.prompt.w.dockWidget.setWindowTitle('Prompt')
        self.w.dynaview.w.dockWidget.setWindowTitle('Tensor Preview')
        self.w.dynaimage.w.dockWidget.setWindowTitle('Image Preview')
        self.w.preview.w.setWindowTitle('Canvas')
        #print(dir(self.w))
        #self.w.tabWidget_1.setTabText(0, "TEST")

        self.vpainter = {}
        #self.resizeDocks({self.thumbnails}, {100}, QtWidgets.Horizontal);

        self.w.preview.w.scene = QGraphicsScene()
        self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)

        self.w.preview.canvas = QPixmap(512, 512)
        self.vpainter["tins"] = QPainter()
        self.vpainter["iins"] = QPainter()
        self.vpainter["main"] = QPainter()
        self.vpainter["main"].begin(self.w.preview.canvas)

        self.w.preview.canvas.fill(Qt.white)
        self.vpainter["main"].end()
        self.w.preview.w.scene.addPixmap(self.w.preview.canvas)


        #self.w.preview.canvas.fill(Qt.black)
        #self.w.preview.w.scene.addPixmap(self.w.preview.canvas)

        self.w.thumbnails.w.thumbsZoom.valueChanged.connect(self.updateThumbsZoom)
        self.w.thumbnails.w.refresh.clicked.connect(self.load_history)

        self.w.imageItem = QGraphicsPixmapItem()
        self.w.imageItem.pixmap().fill(Qt.white)
        #self.w.preview.w.scene.addPixmap(self.w.imageItem.pixmap())
        #self.w.preview.w.scene.update()
        self.newPixmap = {}
        self.tpixmap = {}
        self.updateRate = 3





    def updateThumbsZoom(self):
        size = self.w.thumbnails.w.thumbsZoom.value()
        self.w.thumbnails.w.thumbs.setGridSize(QSize(size, size))
        self.w.thumbnails.w.thumbs.setIconSize(QSize(size, size))
    def update_scaleNumber(self):
        float = self.w.sizer_count.w.scaleSlider.value() / 100
        self.w.sizer_count.w.scaleNumber.display(str(float))
    def update_gfpganNumber(self):
        float = self.w.sizer_count.w.gfpganSlider.value() / 10
        self.w.sizer_count.w.gfpganNumber.display(str(float))


    def show_anim(self):
        self.w.anim.w.show()
    def show_preview(self):
        self.w.preview.w.show()
    def show_prompt(self):
        self.w.prompt.w.show()
    #def show_runner(self):
    #self.runner.show()
    def show_sampler(self):
        self.w.sampler.w.show()
    def show_sizer_count(self):
        self.w.sizer_count.w.show()
    def show_thumbnails(self):
        self.w.thumbnails.w.show()

    def load_history(self):
        self.w.thumbnails.w.thumbs.clear()
        for image in gs.album:
            self.w.thumbnails.w.thumbs.addItem(QListWidgetItem(QIcon(image), str(image)))
    def viewThread(self, item):
        worker = Worker(self.viewImageClicked(item))
        self.threadpool.start(worker)

    def viewImageClicked(self, item):
        try:
            while gs.callbackBusy == True:
                time.sleep(0.1)
            #gs.callbackBusy = True
            vins = random.randint(10000, 99999)
            imageSize = item.icon().actualSize(QSize(10000, 10000))
            qimage = QImage(item.icon().pixmap(imageSize).toImage())
            self.newPixmap[vins] = QPixmap(qimage.size())

            self.vpainter[vins] = QPainter()

            newItem = QGraphicsPixmapItem()
            #vpixmap = self.w.imageItem.pixmap()


            #self.vpainter[vins].device()
            self.vpainter[vins].begin(self.newPixmap[vins])


            self.vpainter[vins].drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
            newItem.setPixmap(self.newPixmap[vins])

            #self.w.imageItem.setPixmap(vpixmap)
            #self.w.preview.w.graphicsView.modified = True
            for items in self.w.preview.w.scene.items():
                self.w.preview.w.scene.removeItem(items)
            self.w.preview.w.scene.addItem(newItem)
            self.w.preview.w.graphicsView.fitInView(newItem, Qt.AspectRatioMode.KeepAspectRatio)
            self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.vpainter[vins].end()
            #gs.callbackBusy = False
        except:
            pass

        #self.w.preview.w.scene.update()
        #self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)

        #rad = self.w.preview.w.graphicsView.penwidth / 2 + 2
        #self.w.preview.w.graphicsView.update(QRect(self.lastPoint, position).normalized().adjusted(-rad, -rad, +rad, +rad))
        #self.w.preview.w.graphicsView.lastPoint = position







        #for item in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(item)


        #self.w.preview.w.scene.clear()
        #imageSize = item.icon().actualSize(QSize(512, 512))
        #print(f'image item type: {type(self.w.imageItem)}')
        #self.w.imageItem.setPixmap(item.icon().pixmap(imageSize))

        #self.w.preview.w.scene.addItem(imageItem)
        #self.w.preview.w.scene.setPixmap(self.w.imageItem)

        #self.w.preview.w.scene.update()

    def run_txt2img(self, progress_callback):

        prompt_list = self.w.prompt.w.textEdit.toPlainText()
        prompt_list = prompt_list.split('\n')
        #self.w.setCentralWidget(self.w.dynaimage.w)
        width=self.w.sizer_count.w.widthSlider.value()
        height=self.w.sizer_count.w.heightSlider.value()
        scale=self.w.sizer_count.w.scaleSlider.value()
        steps=self.w.sizer_count.w.stepsSlider.value()
        samples=self.w.sizer_count.w.samplesSlider.value()
        batchsize=self.w.sizer_count.w.batchSizeSlider.value()
        seamless=self.w.sampler.w.seamless.isChecked()
        full_precision=self.w.sampler.w.fullPrecision.isChecked()
        sampler=self.w.sampler.w.comboBox.currentText()
        upscale=[self.w.sizer_count.w.upscaleSlider.value()]
        gfpgan_strength=self.w.sizer_count.w.gfpganSlider.value() / 100

        self.onePercent = 100 / (batchsize * steps * samples * len(prompt_list))

        if self.w.sampler.w.seedEdit.text() != '':
            seed=int(self.w.sampler.w.seedEdit.text())
        else:
            seed=''


        if gs.defaults.general.default_path_mode == "subfolders":
            outdir = gs.defaults.general.outdir
        else:
            outdir = f'{gs.defaults.general.outdir}/_batch_images'




        """The full list of arguments to Generate() are:
        gr = Generate(
                  weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
                  config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
                  iterations  = <integer>     // how many times to run the sampling (1)
                  steps       = <integer>     // 50
                  seed        = <integer>     // current system time
                  sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
                  grid        = <boolean>     // false
                  width       = <integer>     // image width, multiple of 64 (512)
                  height      = <integer>     // image height, multiple of 64 (512)
                  cfg_scale   = <float>       // condition-free guidance scale (7.5)
                  )

"""
        self.progress = 0.0
        self.update = 3
        for i in range(batchsize):
            for prompt in prompt_list:
                print(f"Full Precision {full_precision}")

                results = gr.prompt2image(prompt   = prompt,
                                          outdir   = outdir,
                                          cfg_scale = scale,
                                          width  = width,
                                          height = height,
                                          iterations = samples,
                                          steps = steps,
                                          seamless = seamless,
                                          sampler_name = sampler,
                                          seed = seed,
                                          upscale = upscale,
                                          gfpgan_strength = gfpgan_strength,
                                          strength = 0.0,
                                          full_precision = full_precision,
                                          step_callback=self.liveUpdate,
                                          image_callback=self.image_cb)
                for row in results:
                    print(f'filename={row[0]}')
                    print(f'seed    ={row[1]}')
                    filename = random.randint(10000, 99999)
                    output = f'outputs/{filename}.png'
                    row[0].save(output)
                    self.image_path = output
                    print("We did set the image")
                    self.w.thumbnails.w.thumbs.addItem(QListWidgetItem(QIcon(self.image_path), str(self.w.prompt.w.textEdit.toPlainText())))
                    #self.get_pic(clear=False)



                #self.get_pic(clear=False)
                #image_qt = QImage(self.image_path)

                #self.w.preview.pic = QGraphicsPixmapItem()
                #self.w.preview.pic.setPixmap(QPixmap.fromImage(image_qt))

                #self.w.preview.w.scene.clear()
                #self.w.preview.w.scene.addItem(self.w.preview.pic)
                #self.w.preview.w.scene.update()



                #all_images.append(results)

                #return all_images








    def txt2img_thread(self):
        # Pass the function to execute
        worker = Worker(self.run_txt2img)
        worker.signals.progress.connect(self.test_output)
        #worker.signals.result.connect(self.set_widget)

        # Execute
        self.threadpool.start(worker)

        #progress bar test:
        #self.progress_thread()
    def test_thread(self, data1, data2):
        # Pass the function to execute
        worker = Worker(self.test_output(data1, data2))
        self.threadpool.start(worker)

    def liveUpdate(self, data1, data2):
        self.progress = self.progress + self.onePercent
        self.w.progressBar.setValue(self.progress)

        if self.update == 3:
            print("Live Update")
            self.test_output(data1, data2)
            self.update = 0
        else:
            self.update += 1
            print("No Live Update")


    def test_output(self, data1, data2):
        try:

            gs.callbackBusy = True

            #transform = T.ToPILImage()
            #img = transform(data1)
            #img = Image.fromarray(data1.astype(np.uint8))
            #img = QImage.fromTensor(data1)

            x_samples = torch.clamp((data1 + 1.0) / 2.0, min=0.0, max=1.0)
            if len(x_samples) != 1:
                raise Exception(
                    f'>> expected to get a single image, but got {len(x_samples)}')
            x_sample = 255.0 * rearrange(
                x_samples[0].cpu().numpy(), 'c h w -> h w c'
            )

            #self.x_sample = cv2.cvtColor(self.x_sample.astype(np.uint8), cv2.COLOR_RGB2BGR)
            x_sample = x_sample.astype(np.uint8)
            dPILimg = Image.fromarray(x_sample)

            tins = random.randint(10000, 99999)
            self.vpainter[tins] = QPainter()
            self.tpixmap = QPixmap(512, 512)

            #self.vpainter[tins].device()
            self.vpainter[tins].begin(self.tpixmap)
            self.dqimg = ImageQt(dPILimg)
            #self.qimage[tins] = ImageQt(dPILimg)
            self.vpainter[tins].drawImage(QRect(0, 0, 512, 512), self.dqimg)
            #self.w.dynaview.w.label.setPixmap(self.tpixmap[tins].scaled(512, 512, Qt.AspectRatioMode.IgnoreAspectRatio))
            #self.vpainter[tins].end()

            #self.w.dynaview.w.label.update()
            #gs.callbackBusy = False

            #dqimg = ImageQt(dPILimg)
            #qimg = QPixmap.fromImage(dqimg)
            self.w.dynaview.w.label.setPixmap(self.tpixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))
            self.vpainter[tins].end()
            gs.callbackBusy = False
        except:
            pass
        #dynapixmap = QPixmap(QPixmap.fromImage(dqimg))

    def image_cb(self, image, seed=None, upscaled=False, use_prefix=None, first_seed=None):
        try:

            #gs.callbackBusy = True
            #dimg = ImageQt(image)
            #dpixmap = QPixmap(QPixmap.fromImage(dimg))
            iins = random.randint(10000, 99999)

            self.vpainter[iins] = QPainter()

            dpixmap = QPixmap(512, 512)
            #self.vpainter[iins] = QPainter(dpixmap)

            self.vpainter[iins].begin(dpixmap)
            self.vpainter[iins].device()


            qimage = ImageQt(image)
            self.vpainter[iins].drawImage(QRect(0, 0, 512, 512), qimage)



            self.w.dynaimage.w.label.setPixmap(dpixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))
            self.vpainter["iins"].end()
            #gs.callbackBusy = False
            #self.w.dynaimage.w.label.update()
        except:
            pass

    def get_pic(self, clear=False): #from self.image_path
        #for item in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(item)

        print("trigger")
        image_qt = QImage(self.image_path)

        self.w.preview.pic = QGraphicsPixmapItem()
        self.w.preview.pic.setPixmap(QPixmap.fromImage(image_qt))
        if clear == True:
            self.w.preview.w.scene.clear()
        self.w.preview.w.scene.addItem(self.w.preview.pic)

        self.w.preview.w.graphicsView.fitInView(self.w.preview.pic, Qt.AspectRatioMode.KeepAspectRatio)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        #gs.obj_to_delete = self.w.preview.pic
    def zoom_IN(self):
        self.w.preview.w.graphicsView.scale(1.25, 1.25)
    def zoom_OUT(self):
        self.w.preview.w.graphicsView.scale(0.75, 0.75)