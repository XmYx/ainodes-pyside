import copy
import math
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PySide6 import QtCore
from PySide6.QtCore import Slot, QRect, Signal, QObject, QThread, QMetaObject, Q_ARG
from PySide6.QtGui import QPixmap, QIcon, QImage, QPainter, Qt

from PySide6.QtWidgets import QListWidgetItem

from backend.singleton import singleton
from frontend.ui_paint import spiralOrder

gs = singleton




class mySignals(QObject):
    add_rect = Signal(object)
    canvas_update = Signal()

class Outpainting:

    def __init__(self, parent):
        self.parent = parent
        self.current_widget = parent.current_widget
        self.signals = mySignals()


    def outpaint_offset_signal(self):

        value = int(self.parent.widgets[self.current_widget].w.mask_offset.value())
        self.parent.canvas.canvas.set_offset(value)


    @Slot()
    def update_outpaint_parameters(self):
        W = self.parent.widgets[self.current_widget].w.W.value()
        H = self.parent.widgets[self.current_widget].w.H.value()
        # W, H = map(lambda x: x - x % 64, (W, H))
        self.parent.widgets[self.current_widget].w.W.setValue(W)
        self.parent.widgets[self.current_widget].w.H.setValue(H)

        self.parent.canvas.canvas.w = W
        self.parent.canvas.canvas.h = H

    def get_params(self):
        params = self.parent.sessionparams.update_params()
        # print(f"Created Params")
        return params


    @Slot(str)
    def create_params(self, uid=None):
        for i in self.parent.canvas.canvas.rectlist:
            if i.id == uid:
                i.params = copy.deepcopy(self.parent.params)



    @Slot(str)
    def update_params(self, uid=None, params=None):

        if self.parent.canvas.canvas.selected_item is not None:
            for i in self.parent.canvas.canvas.rectlist:
                if uid is not None:
                    if i.id == uid:
                        if params == None:
                            params = self.get_params()
                        i.params = copy.deepcopy(params)
                else:
                    if i.id == self.parent.canvas.canvas.selected_item:
                        params = self.get_params()
                        i.params = copy.deepcopy(params)


    @Slot()
    def show_outpaint_details(self):
        self.parent.thumbs.w.thumbnails.clear()
        if self.parent.canvas.canvas.selected_item is not None:

            for items in self.parent.canvas.canvas.rectlist:
                if items.id == self.parent.canvas.canvas.selected_item:
                    # print(items.params)
                    try:
                        self.parent.sessionparams.params = items.params.__dict__
                        self.parent.update_ui_from_params()
                    except Exception as e:
                        print(f"Error, could not update  because of: {e}")

                    if items.params != {}:
                        pass
                        # print(f"showing strength of {items.params['strength'] * 100}")
                        # self.parent.widgets[self.current_widget].w.steps.setValue(items.params.steps)
                        # self.parent.widgets[self.current_widget].w.steps_slider.setValue(items.params.steps)
                        # self.parent.widgets[self.current_widget].w.scale.setValue(items.params['scale'] * 10)
                        # self.parent.widgets[self.current_widget].w.scale_slider.setValue(items.params['scale'] * 10)
                        # self.parent.widgets[self.current_widget].w.strength.setValue(int(items.params['strength'] * 100))
                        # self.parent.widgets[self.current_widget].w.strength_slider.setValue(int(items.params['strength'] * 100))
                        # self.parent.widgets[self.current_widget].w.reconstruction_blur.setValue(items.params['reconstruction_blur'])
                        # self.parent.widgets[self.current_widget].w.mask_blur.setValue(items.params['mask_blur'])
                        # self.parent.widgets[self.current_widget].w.prompts.setText(items.params['prompts'])
                        # self.parent.widgets[self.current_widget].w.seed.setText(str(items.params['seed']))
                        # self.parent.widgets[self.current_widget].w.mask_offset.setValue(items.params['mask_offset'])

                    if items.images is not []:
                        for i in items.images:
                            if i is not None:
                                image = i.copy(0, 0, i.width(), i.height())
                                pixmap = QPixmap.fromImage(image)
                                self.parent.thumbs.w.thumbnails.addItem(
                                    QListWidgetItem(QIcon(pixmap), f"{items.render_index}"))

    def redo_current_outpaint(self):
        self.parent.canvas.canvas.redo_outpaint(self.parent.canvas.canvas.selected_item)


    def select_outpaint_image(self, item):
        width = self.parent.widgets[self.current_widget].w.W.value()
        height = self.parent.widgets[self.current_widget].w.H.value()
        templist = self.parent.canvas.canvas.rectlist
        imageSize = item.icon().actualSize(QtCore.QSize(width, height))
        if self.parent.canvas.canvas.selected_item is not None:
            for i in templist:
                if i.id == self.parent.canvas.canvas.selected_item:
                    qimage = QImage(item.icon().actualSize(QtCore.QSize(width, height)), QImage.Format_ARGB32)
                    painter = QPainter()
                    painter.begin(qimage)
                    painter.drawPixmap(0, 0, item.icon().pixmap(imageSize))
                    painter.end()
                    i.image = qimage
                    i.timestring = time.time()
        self.parent.canvas.canvas.update()
        self.parent.canvas.canvas.rectlist = templist
        self.parent.canvas.canvas.newimage = True
        self.parent.canvas.canvas.update()

    def delete_outpaint_frame(self):
        # self.parent.canvas.canvas.undoitems = []
        if self.parent.canvas.canvas.selected_item is not None:
            x = 0
            for i in self.parent.canvas.canvas.rectlist:
                if i.id == self.parent.canvas.canvas.selected_item:
                    self.parent.canvas.canvas.undoitems.append(i)
                    self.parent.canvas.canvas.rectlist.pop(x)
                    pass
                x += 1
        self.parent.canvas.canvas.pixmap.fill(Qt.transparent)
        self.parent.canvas.canvas.newimage = True
        self.parent.canvas.canvas.selected_item = None
        self.parent.canvas.canvas.update()
        # self.parent.canvas.canvas.draw_rects()
        self.parent.thumbs.w.thumbnails.clear()

    # maybe unused ???
    def test_save_outpaint(self):

        self.parent.canvas.canvas.pixmap = self.parent.canvas.canvas.pixmap.copy(QRect(64, 32, 512, 512))

        self.parent.canvas.canvas.setPixmap(self.parent.canvas.canvas.pixmap)
        self.parent.canvas.canvas.update()

    def run_batch_outpaint(self, progress_callback=False):
        self.stopprocessing = False
        self.parent.callbackbusy = False
        self.sleepytime = 0.0
        self.parent.choice = "Outpaint"
        self.create_outpaint_batch()

    def wait_parent_busy(self):
        while self.parent.busy == True:
            time.sleep(0.25)

    def wait_callback_busy(self):
        while self.parent.callbackbusy == True:
            time.sleep(0.25)
            self.sleepytime += 0.25
        time.sleep(0.25)
        self.sleepytime += 0.25

    def wait_canvas_busy(self):
        while self.parent.canvas.canvas.busy == True:
            time.sleep(0.25)
            self.sleepytime += 0.25

    @Slot(object)
    def add_rect(self, data):
        print('addrect')
        self.parent.canvas.canvas.addrect_atpos(prompt=data["prompt"], x=data['x'], y=data['y'], image=data['image'],
                                                render_index=data['render_index'], order=data['order'],
                                                params=data['params'])
        self.parent.canvas.canvas.update()

    @Slot(object)
    def canvas_update(self):
        print('update')
        self.parent.canvas.canvas.update()


    def thread_save_add_rect(self, items, image, index , params):
        data_object = {
            'prompt':items["prompt"],
            'x':items['x'],
            'y':items['y'],
            'image':image,
            'render_index':index,
            'order':items["order"],
            'params':copy.deepcopy(params)
        }

        QMetaObject.invokeMethod(self.parent.canvas.canvas, "addrect_atpos_object", Qt.QueuedConnection,Q_ARG(QObject, data_object))



    def create_outpaint_batch(self, gobig_img_path=None, progress_callback=False):
        tilesize = 512
        self.parent.callbackbusy = True
        self.parent.busy = False
        self.parent.params = self.parent.sessionparams.update_params()
        self.parent.sessionparams.params.advanced = True

        offset = self.parent.widgets[self.current_widget].w.mask_offset.value()
        overlap = self.parent.widgets[self.current_widget].w.rect_overlap.value()
        # self.preview_batch_outpaint()

        if gobig_img_path is not None:
            tilesize = int(self.parent.widgets[self.current_widget].w.batch_upscale_tile_size.currentText())
            upscale_factor = self.parent.widgets[self.current_widget].w.batch_upscale_factor.value()
            pil_image = Image.open(gobig_img_path)
            width, height = pil_image.size
            target_h = int(int(height) * upscale_factor)
            target_w = int(int(width) * upscale_factor)
            self.parent.canvas.H.setValue(int(target_h))
            self.parent.canvas.W.setValue(int(target_w))
            pil_image = pil_image.resize((target_w, target_h),Image.Resampling.LANCZOS).convert("RGBA")
            qimage = ImageQt(pil_image)
            chops_x = int(qimage.width() / self.parent.canvas.canvas.w) + 1
            chops_y = int(qimage.width() / self.parent.canvas.canvas.h)
            self.parent.widgets[self.current_widget].w.rect_overlap.setValue(overlap)
            self.preview_batch_outpaint(chops_x=chops_x, chops_y=chops_y)


        rparams = self.parent.sessionparams.update_params()
        # print(self.tempsize_int)
        prompt_series = pd.Series([np.nan for a in range(self.tempsize_int)])
        # print(prompt_series)
        if rparams.keyframes == '':
            rparams.keyframes = "0"
        prom = rparams.prompts
        key = rparams.keyframes

        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))

        prompts = dict(zip(new_key, new_prom))

        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt
        animation_prompts = prompt_series.ffill().bfill()
        print(animation_prompts)
        x = 0
        for items in self.parent.canvas.canvas.tempbatch:
            if type(items) == list:
                for item in items:

                    if gobig_img_path is not None:
                        rect = QRect(item['x'], item['y'], self.parent.canvas.canvas.w, self.parent.canvas.canvas.h)
                        image = qimage.copy(rect)
                        index = None
                        self.hires_source = pil_image

                    else:
                        image = None
                        index = None
                        self.hires_source = None
                    offset = offset + tilesize
                    rparams.prompts = animation_prompts[x]
                    if rparams.seed_behavior == 'random':
                        rparams.seed = random.randint(0, 2 ** 32 - 1)
                    # print(f"seed bhavior:{rparams.seed_behavior} {rparams.seed}")

                    self.parent.canvas.canvas.addrect_atpos(prompt=item["prompt"], x=item['x'], y=item['y'], image=image,
                                                     render_index=index, order=item["order"],
                                                     params=copy.deepcopy(rparams))
                    print(animation_prompts[x])
                    if rparams.seed_behavior == 'iter':
                        rparams.seed += 1
                    self.wait_parent_busy()
                    x += 1
            elif type(items) == dict:


                if gobig_img_path is not None:
                    rect = QRect(items['x'], items['y'], self.parent.canvas.canvas.w, self.parent.canvas.canvas.h)
                    image = qimage.copy(rect)
                    index = None
                    self.hires_source = pil_image

                else:
                    image = None
                    index = None
                    self.hires_source = None
                offset = offset + tilesize
                rparams.prompts = animation_prompts[x]
                # print(rparams.prompts)
                if rparams.seed_behavior == 'random':
                    rparams.seed = random.randint(0, 2 ** 32 - 1)

                #self.thread_save_add_rect(items, image, index , copy.deepcopy(rparams))

                self.parent.canvas.canvas.addrect_atpos(prompt=items["prompt"], x=items['x'], y=items['y'], image=image,
                                                 render_index=index, order=items["order"],
                                                 params=copy.deepcopy(rparams))

                if rparams.seed_behavior == 'iter':
                    rparams.seed += 1

                x += 1
        self.parent.callbackbusy = False
        print('15 redraw')
        self.parent.canvas.canvas.redraw()

    def run_hires_batch(self, progress_callback=None):
        self.parent.sessionparams.update_params()
        self.parent.sessionparams.params.advanced = True
        # multi = self.parent.widgets[self.current_widget].w.multiBatch.isChecked()
        # batch_n = self.parent.widgets[self.current_widget].w.multiBatchvalue.value()
        multi = False
        batch_n = 1
        gs.stop_all = False
        self.parent.callbackbusy = False
        self.sleepytime = 0.0
        self.parent.choice = "Outpaint"

        for i in range(batch_n):
            self.wait_callback_busy()
            time.sleep(1)
            betterslices = []
            og_size = (512, 512)
            tiles = (self.parent.canvas.canvas.cols - 1) * (self.parent.canvas.canvas.rows)
            for x in range(int(tiles)):
                if gs.stop_all != True:
                    self.run_hires_step_x(x)
                    betterslices.append((self.parent.image.convert('RGBA'), self.parent.canvas.canvas.rectlist[x].x,
                                         self.parent.canvas.canvas.rectlist[x].y))
                else:
                    break

            source_image = self.hires_source
            alpha = Image.new("L", og_size, color=0xFF)
            alpha_gradient = ImageDraw.Draw(alpha)
            a = 0
            i = 0
            overlap = self.parent.widgets[self.current_widget].w.rect_overlap.value()
            shape = (og_size, (0, 0))
            while i < overlap:
                alpha_gradient.rectangle(shape, fill=a)
                a += 4
                i += 1
                shape = ((og_size[0] - i, og_size[1] - i), (i, i))
            mask = Image.new("RGBA", og_size, color=0)
            mask.putalpha(alpha)
            finished_slices = []
            for betterslice, x, y in betterslices:
                finished_slice = self.addalpha(betterslice, mask)
                finished_slices.append((finished_slice, x, y))
            # # Once we have all our images, use grid_merge back onto the source, then save
            final_output = self.grid_merge(
                source_image.convert("RGBA"), finished_slices
            ).convert("RGBA")
            final_output.save('output/test_hires.png')
            # base_filename = f"{base_filename}d"
            print(f"All time wasted: {self.sleepytime} seconds.")
            self.hires_source = final_output
            self.parent.deforum_ui.signals.prepare_hires_batch.emit('output/test_hires.png')

    def run_hires_step_x(self, x):
        print('x', x)
        self.parent.choice = 'Outpaint'
        image = self.parent.canvas.canvas.rectlist[x].image
        image.save('output/temp/temp.png', "PNG")
        self.parent.canvas.canvas.selected_item = self.parent.canvas.canvas.rectlist[x].id
        print('self.parent.canvas.canvas.selected_item',self.parent.canvas.canvas.selected_item)
        self.parent.render_index = x
        self.parent.deforum_ui.run_deforum_six_txt2img(hiresinit='output/temp/temp.png')

        self.wait_callback_busy()
        x += 1

        self.parent.busy = False
        return x

    def run_outpaint_step_x(self, x):

        # print("it should not do anything....")

        self.parent.busy = True
        self.parent.canvas.canvas.reusable_outpaint(self.parent.canvas.canvas.rectlist[x].id)
        self.wait_canvas_busy()

        self.parent.deforum_ui.run_deforum_outpaint(self.parent.canvas.canvas.rectlist[x].params)
        self.wait_callback_busy()
        x += 1

        self.parent.busy = False
        #return x

    def run_prepared_outpaint_batch(self, progress_callback=None):
        gs.stop_all = False
        self.parent.callbackbusy = False
        self.sleepytime = 0.0
        self.parent.choice = "Outpaint"
        self.parent.params.advanced = True

        # multi = self.parent.widgets[self.current_widget].w.multiBatch.isChecked()
        # batch_n = self.parent.widgets[self.current_widget].w.multiBatchvalue.value()

        multi = False
        batch_n = 1

        tiles = len(self.parent.canvas.canvas.rectlist)

        print(f"Tiles to Outpaint:{tiles}")

        if multi == True:
            print("multi outpaint batch")
            for i in range(batch_n):
                if i != 0:
                    filename = str(random.randint(1111111, 9999999))
                    self.parent.canvas.canvas.save_rects_as_json(filename=filename)
                    self.parent.canvas.canvas.save_canvas()
                    self.parent.canvas.canvas.rectlist.clear()
                    self.create_outpaint_batch()
                for x in range(tiles):
                    # print(x)
                    if gs.stop_all == False:
                        self.run_outpaint_step_x(x)
                    else:
                        break
        else:
            for x in range(tiles):
                if gs.stop_all == False:
                    print(f"running step {x}")
                    self.run_outpaint_step_x(x)
                else:
                    break
            # self.parent.canvas.canvas.save_canvas()

            print(f"All time wasted: {self.sleepytime} seconds.")


    def preview_batch_outpaint(self, chops_x=None, chops_y=None):
        if chops_x is None:
            self.parent.canvas.canvas.cols = self.parent.widgets[self.current_widget].w.batch_columns.value()
            self.parent.canvas.canvas.rows = self.parent.widgets[self.current_widget].w.batch_rows.value()
        else:
            self.parent.canvas.canvas.cols = chops_x
            self.parent.canvas.canvas.rows = chops_y

        print('self.parent.canvas.canvas.cols',self.parent.canvas.canvas.cols)
        print('self.parent.canvas.canvas.rows',self.parent.canvas.canvas.rows)

        self.parent.canvas.canvas.offset = self.parent.widgets[self.current_widget].w.rect_overlap.value()
        self.parent.canvas.canvas.maskoffset = self.parent.widgets[self.current_widget].w.mask_offset.value()
        randomize = self.parent.widgets[self.current_widget].w.randomize.isChecked()
        spiral = self.parent.widgets[self.current_widget].w.spiral.isChecked()
        reverse = self.parent.widgets[self.current_widget].w.reverse.isChecked()
        startOffsetX = self.parent.widgets[self.current_widget].w.start_offset_x.value()
        startOffsetY = self.parent.widgets[self.current_widget].w.start_offset_y.value()
        prompts = self.parent.widgets[self.current_widget].w.prompts.toPlainText()
        # keyframes = self.prompt.w.keyFrames.toPlainText()
        keyframes = ""

        self.parent.canvas.canvas.tempbatch = self.parent.canvas.canvas.create_tempBatch(prompts, keyframes, startOffsetX, startOffsetY, randomize)
        templist = []
        if spiral:
            print(self.parent.canvas.canvas.tempbatch)
            # self.parent.canvas.canvas.tempbatch = random_path(self.parent.canvas.canvas.tempbatch, self.parent.canvas.canvas.cols)
            self.parent.canvas.canvas.tempbatch = spiralOrder(self.parent.canvas.canvas.tempbatch)
        if reverse:
            self.parent.canvas.canvas.tempbatch.reverse()
        # print(len(self.parent.canvas.canvas.tempbatch))
        self.tempsize_int = self.parent.canvas.canvas.cols * self.parent.canvas.canvas.rows

        self.parent.canvas.canvas.draw_tempBatch(self.parent.canvas.canvas.tempbatch)
        self.parent.canvas.update()

    def outpaint_rect_overlap(self):
        self.parent.canvas.canvas.rectPreview = self.parent.widgets[self.current_widget].w.enable_overlap.isChecked()
        if self.parent.canvas.canvas.rectPreview == False:
            self.parent.canvas.canvas.newimage = True
            print('16 redraw')
            self.parent.canvas.canvas.redraw()
        elif self.parent.canvas.canvas.rectPreview == True:
            self.parent.canvas.canvas.visualize_rects()

    def addalpha(self, im, mask):
        imr, img, imb, ima = im.split()
        mmr, mmg, mmb, mma = mask.split()
        im = Image.merge(
            "RGBA", [imr, img, imb, mma]
        )  # we want the RGB from the original, but the transparency from the mask
        return im


    # Alternative method composites a grid of images at the positions provided
    def grid_merge(self, source, slices):
        source.convert("RGBA")
        for slice, posx, posy in slices:  # go in reverse to get proper stacking
            source.alpha_composite(slice, (posx, posy))
        return source
