import base64
import copy
import time
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QListWidgetItem
import torchvision.transforms as T
import torch
from einops import rearrange

from backend.singleton import singleton
from backend.web_requests.web_images import WebImages


gs = singleton
class UiImage:
    def __init__(self, parent):
        self.parent = parent
        self.web_images = WebImages()
        self.stopwidth = False
        self.canvas_single_temp = False
        self.render_index = 0
        self.make_grid = False
        self.all_images = []
        self.latent_rgb_factors = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float, device='cuda')
    def show_model_preview_images(self, model):
        for image in model['images']:
            self.show_image_from_url(image['url'])

    def show_image_from_url(self, url):
        self.web_images.get_image(url)

    def show_web_image_on_canvas(self, image_string):
        try:
            gs.temppath = ''
            self.parent.params.max_frames = 0
            image = Image.open(BytesIO(image_string))
            image = image.convert("RGB")
            mode = image.mode
            size = image.size
            enc_image = base64.b64encode(image.tobytes()).decode()
            self.parent.deforum_ui.signals.txt2img_image_cb.emit(enc_image, mode, size)

        except Exception as e:
            print('Error while fetching the images from web: ', e)
            
            
    def image_preview_signal(self, image, *args, **kwargs):
        try:
            mode = image.mode
            size = image.size
            enc_image = base64.b64encode(image.tobytes()).decode()
            self.parent.deforum_ui.signals.txt2img_image_cb.emit(enc_image, mode, size)
            self.parent.signals.image_ready.emit()
        except Exception as e:
            print('image_preview_signal', e)
            
            
    @Slot()
    def image_preview_func(self, image=None):
        try:
            img = image #self.image

            if self.parent.params.canvas_single == True and (self.parent.canvas.canvas.rectlist == [] or self.parent.canvas.canvas.rectlist is None):
                self.parent.params.canvas_single = False
                self.canvas_single_temp = True

            if self.parent.params.canvas_single == True:

                if self.parent.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.parent.canvas.canvas.rectlist[self.render_index].images is not None:
                            templist = self.parent.canvas.canvas.rectlist[self.render_index].images
                        else:
                            templist = []
                        self.parent.canvas.canvas.rectlist[self.render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.parent.canvas.canvas.rectlist[self.render_index].render_index)
                        self.parent.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.parent.canvas.canvas.rectlist[self.render_index].render_index}"))

                        if self.parent.canvas.canvas.anim_inpaint == True:
                            templist[self.parent.canvas.canvas.rectlist[self.render_index].render_index] = qimage
                            self.parent.canvas.canvas.anim_inpaint = False
                        elif self.parent.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.parent.canvas.canvas.rectlist[self.render_index].render_index == None:
                                self.parent.canvas.canvas.rectlist[self.render_index].render_index = 0
                            else:
                                self.parent.canvas.canvas.rectlist[self.render_index].render_index += 1
                        self.parent.canvas.canvas.rectlist[self.render_index].images = templist
                        self.parent.canvas.canvas.rectlist[self.render_index].image = self.parent.canvas.canvas.rectlist[self.render_index].images[self.parent.canvas.canvas.rectlist[self.render_index].render_index]
                        #self.parent.canvas.canvas.rectlist[self.render_index].image = qimage
                        self.parent.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                        self.parent.canvas.canvas.rectlist[self.render_index].img_path = gs.temppath
                    self.parent.canvas.canvas.newimage = True
                    self.parent.canvas.canvas.update()
                    self.parent.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.parent.params.canvas_single == False:

                if self.canvas_single_temp == True:
                    self.canvas_single_temp = False
                    self.parent.params.canvas_single = True

                if img is not None:
                    image = img
                    w, h = image.size
                    self.add_next_rect(h, w)
                    self.render_index = len(self.parent.canvas.canvas.rectlist) - 1

                    # for items in self.parent.canvas.canvas.rectlist:
                    #    if items.id == self.parent.canvas.canvas.render_item:
                    if self.parent.canvas.canvas.rectlist[self.render_index].images is not None:
                        templist = self.parent.canvas.canvas.rectlist[self.render_index].images
                    else:
                        templist = []
                    self.parent.canvas.canvas.rectlist[self.render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.parent.canvas.canvas.rectlist[self.render_index].images = templist
                    if self.parent.canvas.canvas.rectlist[self.render_index].render_index == None:
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index = 0
                    else:
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index += 1
                    self.parent.canvas.canvas.rectlist[self.render_index].image = \
                    self.parent.canvas.canvas.rectlist[self.render_index].images[
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index]
                    self.parent.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                    self.parent.canvas.canvas.rectlist[self.render_index].params = self.parent.params

                    self.parent.canvas.canvas.newimage = True
                    self.parent.canvas.canvas.redraw()
                    self.parent.canvas.canvas.update()

            if self.parent.params.canvas_single == False and self.parent.params.max_frames > 1:
                self.parent.params.canvas_single = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))
        except Exception as e:
            print('image_preview_func', e)

    @Slot()
    def image_preview_func_str(self, image, mode, size):
        try:
            decoded_image = base64.b64decode(image.encode())
            image = Image.frombytes(mode, size, decoded_image)
            image = image.convert('RGB')
            self.image_preview_func(image)
        except Exception as e:
            print('image_preview_func_str', e)
    @Slot()
    def render_index_image_preview_func_str(self, image, mode, size, render_index):
        decoded_image = base64.b64decode(image.encode())
        self.render_index_image_preview_func(Image.frombytes(mode, size, decoded_image), render_index)
    @Slot()
    def render_index_image_preview_func(self, image=None, render_index=None):
        try:
            img = image
            if self.parent.outpaint.batch_process == 'run_hires_batch':
                self.parent.outpaint.last_batch_image = img
            if self.parent.params.canvas_single == True:
                if self.parent.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.parent.canvas.canvas.rectlist[render_index].images is not None:
                            templist = self.parent.canvas.canvas.rectlist[render_index].images
                        else:
                            templist = []
                        self.parent.canvas.canvas.rectlist[render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.parent.canvas.canvas.rectlist[render_index].render_index)
                        self.parent.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.parent.canvas.canvas.rectlist[render_index].render_index}"))

                        if self.parent.canvas.canvas.anim_inpaint == True:
                            templist[self.parent.canvas.canvas.rectlist[render_index].render_index] = qimage
                            self.parent.canvas.canvas.anim_inpaint = False
                        elif self.parent.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.parent.canvas.canvas.rectlist[render_index].render_index == None:
                                self.parent.canvas.canvas.rectlist[render_index].render_index = 0
                            else:
                                self.parent.canvas.canvas.rectlist[render_index].render_index += 1
                        self.parent.canvas.canvas.rectlist[render_index].images = templist
                        self.parent.canvas.canvas.rectlist[render_index].image = \
                            self.parent.canvas.canvas.rectlist[render_index].images[
                                self.parent.canvas.canvas.rectlist[render_index].render_index]
                        self.parent.canvas.canvas.rectlist[render_index].timestring = time.time()
                        self.parent.canvas.canvas.rectlist[render_index].img_path = gs.temppath
                    self.parent.canvas.canvas.newimage = True
                    self.parent.canvas.canvas.update()
                    self.parent.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.parent.params.canvas_single == False:

                if img is not None:
                    image = img
                    h, w = image.size
                    self.add_next_rect(h, w)
                    render_index = len(self.parent.canvas.canvas.rectlist) - 1

                    # for items in self.parent.canvas.canvas.rectlist:
                    #    if items.id == self.parent.canvas.canvas.render_item:
                    if self.parent.canvas.canvas.rectlist[render_index].images is not None:
                        templist = self.parent.canvas.canvas.rectlist[render_index].images
                    else:
                        templist = []
                    self.parent.canvas.canvas.rectlist[render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.parent.canvas.canvas.rectlist[render_index].images = templist
                    if self.parent.canvas.canvas.rectlist[render_index].render_index == None:
                        self.parent.canvas.canvas.rectlist[render_index].render_index = 0
                    else:
                        self.parent.canvas.canvas.rectlist[render_index].render_index += 1
                    self.parent.canvas.canvas.rectlist[render_index].image = self.parent.canvas.canvas.rectlist[render_index].images[self.parent.canvas.canvas.rectlist[render_index].render_index]
                    self.parent.canvas.canvas.rectlist[render_index].timestring = time.time()
                    self.parent.canvas.canvas.rectlist[render_index].params = self.parent.params
            self.parent.canvas.canvas.newimage = True
            self.parent.canvas.canvas.redraw()
            self.parent.canvas.canvas.update()

            if self.parent.params.canvas_single == False and self.parent.params.max_frames > 1:
                self.parent.params.canvas_single = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))

        except Exception as e:
            print('render_index_image_preview_func', e)
    def image_preview_signal_op(self, image, *args, **kwargs):
        mode = image.mode
        size = image.size
        enc_image = base64.b64encode(image.tobytes()).decode()
        self.parent.outpaint.signals.txt2img_image_op.emit(enc_image, mode, size)
        self.parent.signals.image_ready.emit()
    @Slot()
    def image_preview_func_str_op(self, image, mode, size):
        decoded_image = base64.b64decode(image.encode())
        self.image_preview_func_op(Image.frombytes(mode, size, decoded_image))

    @Slot()
    def image_preview_func_op(self, image=None):
        try:
            img = image #self.image
            # store the last image for a part of the batch hires process
            if self.parent.outpaint.batch_process == 'run_hires_batch':
                index = self.render_index
                if index < 0:
                    index = 0
                self.parent.outpaint.betterslices.append((img.convert('RGBA'),
                                          self.parent.canvas.canvas.rectlist[index].x,
                                          self.parent.canvas.canvas.rectlist[index].y))

            if self.parent.params.canvas_single == True and (self.parent.canvas.canvas.rectlist == [] or self.parent.canvas.canvas.rectlist is None):
                self.parent.params.canvas_single = False
                self.canvas_single_temp = True

            if self.parent.params.canvas_single == True:

                if self.parent.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.parent.canvas.canvas.rectlist[self.render_index].images is not None:
                            templist = self.parent.canvas.canvas.rectlist[self.render_index].images
                        else:
                            templist = []
                        self.parent.canvas.canvas.rectlist[self.render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.parent.canvas.canvas.rectlist[self.render_index].render_index)
                        self.parent.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.parent.canvas.canvas.rectlist[self.render_index].render_index}"))

                        if self.parent.canvas.canvas.anim_inpaint == True:
                            templist[self.parent.canvas.canvas.rectlist[self.render_index].render_index] = qimage
                            self.parent.canvas.canvas.anim_inpaint = False
                        elif self.parent.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.parent.canvas.canvas.rectlist[self.render_index].render_index == None:
                                self.parent.canvas.canvas.rectlist[self.render_index].render_index = 0
                            else:
                                self.parent.canvas.canvas.rectlist[self.render_index].render_index += 1
                        self.parent.canvas.canvas.rectlist[self.render_index].images = templist
                        self.parent.canvas.canvas.rectlist[self.render_index].image = self.parent.canvas.canvas.rectlist[self.render_index].images[self.parent.canvas.canvas.rectlist[self.render_index].render_index]
                        #self.parent.canvas.canvas.rectlist[self.render_index].image = qimage
                        self.parent.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                        self.parent.canvas.canvas.rectlist[self.render_index].img_path = gs.temppath
                    self.parent.canvas.canvas.newimage = True
                    self.parent.canvas.canvas.update()
                    self.parent.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.parent.params.advanced == False:

                if self.canvas_single_temp == True:
                    self.canvas_single_temp = False
                    self.parent.params.canvas_single = True

                if img is not None:
                    image = img
                    h, w = image.size
                    self.add_next_rect(h, w)
                    self.render_index = len(self.parent.canvas.canvas.rectlist) - 1

                    # for items in self.parent.canvas.canvas.rectlist:
                    #    if items.id == self.parent.canvas.canvas.render_item:
                    if self.parent.canvas.canvas.rectlist[self.render_index].images is not None:
                        templist = self.parent.canvas.canvas.rectlist[self.render_index].images
                    else:
                        templist = []
                    self.parent.canvas.canvas.rectlist[self.render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.parent.canvas.canvas.rectlist[self.render_index].images = templist
                    if self.parent.canvas.canvas.rectlist[self.render_index].render_index == None:
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index = 0
                    else:
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index += 1
                    self.parent.canvas.canvas.rectlist[self.render_index].image = \
                    self.parent.canvas.canvas.rectlist[self.render_index].images[
                        self.parent.canvas.canvas.rectlist[self.render_index].render_index]
                    self.parent.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                    self.parent.canvas.canvas.rectlist[self.render_index].params = self.parent.params

                    self.parent.canvas.canvas.newimage = True
                    self.parent.canvas.canvas.redraw()
                    self.parent.canvas.canvas.update()

            if self.parent.params.canvas_single == False and self.parent.params.max_frames > 1:
                self.parent.params.canvas_single = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))
        except Exception as e:
            print('image_preview_func_op', e)


    def add_next_rect(self, h, w):
        #w = self.parent.widgets[self.current_widget].w.W.value()
        #h = self.parent.widgets[self.current_widget].w.H.value()
        resize = False
        try:
            params = copy.deepcopy(self.parent.params)
            if self.parent.canvas.canvas.rectlist == []:
                self.parent.canvas.canvas.w = w
                self.parent.canvas.canvas.h = h
                self.parent.canvas.canvas.addrect_atpos(x=0, y=0, params=params)
                self.parent.cheight = h
                self.parent.w = w
                self.parent.canvas.canvas.render_item = self.parent.canvas.canvas.selected_item
                # print(f"this should only haappen once {self.parent.cheight}")
                # self.parent.canvas.canvas.resize_canvas(w=self.parent.w, h=self.parent.cheight)
            elif self.parent.canvas.canvas.rectlist != []:
                for i in self.parent.canvas.canvas.rectlist:
                    if i.id == self.parent.canvas.canvas.render_item:
                        if i.id == self.parent.canvas.canvas.render_item:
                            w = self.parent.canvas.canvas.rectlist[self.parent.canvas.canvas.rectlist.index(i)].w
                            x = self.parent.canvas.canvas.rectlist[self.parent.canvas.canvas.rectlist.index(i)].x + w + 20
                            y = i.y
                            if x > 3000:
                                x = 0
                                y = self.parent.cheight + 25
                                if self.stopwidth == False:
                                    self.stopwidth = True
                            if self.stopwidth == False:
                                self.parent.w = x + w
                                resize = True
                            if self.parent.cheight < y + i.h:
                                self.parent.cheight = y + i.h
                                resize = True
                            # self.parent.canvas.canvas.selected_item = None
                self.parent.canvas.canvas.addrect_atpos(x=x, y=y, params=params)
                self.parent.canvas.canvas.render_item = self.parent.canvas.canvas.selected_item
            # if resize == True:
            # pass
            # print(self.parent.w, self.parent.cheight)
            self.parent.canvas.canvas.resize_canvas(w=self.parent.w, h=self.parent.cheight)
            # self.parent.canvas.canvas.update()
            # self.parent.canvas.canvas.redraw()
        except Exception as e:
            print('add_next_rect', e)

    def tensor_preview_signal(self, data, data2):
        self.data = data
        if isinstance(self.data, np.ndarray):
            print('is ndarray')
            self.data = torch.from_numpy(self.data)

        if data2 is not None:
            self.data2 = data2
        else:
            self.data2 = None
        self.parent.deforum_ui.signals.deforum_step.emit()

    def tensor_preview_schedule(self):  # TODO: Rename this function to tensor_draw_function
        try:
            if len(self.data) != 1:
                print(
                    f'we got {len(self.data)} Tensors but Tensor Preview will show only one')

            # Applying RGB fix on incoming tensor found at: https://github.com/keturn/sd-progress-demo/
            self.data = torch.einsum('...lhw,lr -> ...rhw', self.data[0], self.latent_rgb_factors)
            self.data = (((self.data + 1) / 2)
                         .clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         .byte())
            # Copying to cpu as numpy array
            self.data = rearrange(self.data, 'c h w -> h w c').cpu().numpy()
            dPILimg = Image.fromarray(self.data)
            dqimg = ImageQt(dPILimg)
            # Setting Canvas's Tensor Preview item, then calling function to draw it.
            self.parent.canvas.canvas.tensor_preview_item = dqimg
            self.parent.canvas.canvas.tensor_preview()
            del dPILimg
            del dqimg
        except Exception as e:
            print('tensor_preview_schedule', e)


    @Slot(object)
    def draw_tempRects_signal(self, values):
        self.parent.canvas.canvas.draw_tempRects_signal(values)

