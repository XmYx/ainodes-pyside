import copy
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PySide6 import QtCore
from PySide6.QtCore import Slot, QRect, Signal, QObject
from PySide6.QtGui import QPixmap, QIcon, QImage, QPainter, Qt, QPen
from PySide6.QtWidgets import QListWidgetItem

from backend.singleton import singleton
from frontend.ui_paint import spiralOrder

gs = singleton




class mySignals(QObject):
    add_rect = Signal(object)
    canvas_update = Signal()
    rect_ready_in_ui = Signal()

class Outpainting:

    def __init__(self, parent):
        self.parent = parent
        self.current_widget = parent.current_widget
        self.signals = mySignals()
        self.tile_size = 512
        self.batch_process = None
        self.batch_process_items = None
        self.gobig_pil_image = None  # had a typo here made by osi gobbig
        self.gobig_img_path = None
        self.last_batch_image = None


    @Slot(str)
    def run_create_outpaint_img2img_batch(self, input=None):
        if input != False:
            data = input
        else:
            data = self.parent.getfile()
        if data is not None:
            self.create_outpaint_batch(gobig_img_path=data)

    @Slot()
    def run_hires_batch_thread(self):
        self.parent.run_as_thread(self.run_hires_batch)

    @Slot()
    def run_prepared_outpaint_batch_thread(self):
        if self.parent.canvas.canvas.rectlist == []:
            self.create_outpaint_batch()
        self.parent.run_as_thread(self.run_prepared_outpaint_batch)

    @Slot()
    def preview_batch_outpaint_thread(self):
        self.preview_batch_outpaint()
        #self.parent.run_as_thread(self.preview_batch_outpaint)


    @Slot()
    def prepare_batch_outpaint_thread(self):
        #self.outpaint.run_batch_outpaint()
        self.parent.run_as_thread(self.run_batch_outpaint)

    def outpaint_offset_signal(self):

        value = int(self.parent.widgets[self.current_widget].w.mask_offset.value())
        self.parent.canvas.canvas.set_offset(value)


    def update_outpaint_parameters(self):
        #print('update outpaint')
        W = self.parent.widgets[self.current_widget].w.W.value()
        H = self.parent.widgets[self.current_widget].w.H.value()
        # W, H = map(lambda x: x - x % 64, (W, H))
        self.parent.widgets[self.current_widget].w.W.setValue(W)
        self.parent.widgets[self.current_widget].w.H.setValue(H)


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
                    try:
                        self.parent.sessionparams.params = items.params.__dict__
                        self.parent.update_ui_from_params()
                    except Exception as e:
                        print(f"Error, could not update  because of: {e}")

                    # todo still needed?
                    #if items.params != {}:
                    #    pass
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


        self.parent.choice = "Outpaint"
        self.create_outpaint_batch()


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

    def rect_ready_in_ui(self):
        self.batch_step_number += 1
        self.next_rect_from_batch()

    def next_rect_from_batch(self):
        if self.batch_process == 'create_outpaint_batch':
            if self.batch_process_items is not None and len(self.batch_process_items) > 0:
                item = self.batch_process_items[0]
                del self.batch_process_items[0]
                self.next_rect_from_batch_list(item)
            else:
                if self.parent.canvas.canvas.tempbatch_work is not None and len(self.parent.canvas.canvas.tempbatch_work) > 0:
                    self.batch_process_items = self.parent.canvas.canvas.tempbatch_work[0]
                    del self.parent.canvas.canvas.tempbatch_work[0]
                    if type(self.batch_process_items) == dict:
                        self.batch_process_items = [self.batch_process_items]
                    self.next_rect_from_batch()
                else:
                    self.batch_process = None
                    self.parent.canvas.canvas.draw_rects()
                    self.parent.canvas.canvas.redraw()


    def next_rect_from_batch_list(self, item):
        tilesize = self.tile_size
        if self.gobig_img_path is not None:
            rect = QRect(item['x'], item['y'], self.parent.canvas.canvas.w, self.parent.canvas.canvas.h)
            image = self.gobig_qimage.copy(rect)
            index = None
            self.hires_source = self.gobig_pil_image

        else:
            image = None
            index = None
            self.hires_source = None

        offset = self.parent.widgets[self.current_widget].w.mask_offset.value() + tilesize
        self.rparams.prompts = self.animation_prompts[self.batch_step_number]
        if self.rparams.seed_behavior == 'random':
            self.rparams.seed = random.randint(0, 2 ** 32 - 1)

        self.parent.canvas.canvas.addrect_atpos(prompt=item["prompt"], x=item['x'], y=item['y'], image=image,
                                                render_index=index, order=item["order"],
                                                params=copy.deepcopy(self.rparams), color=Qt.red)

        if self.rparams.seed_behavior == 'iter':
            self.rparams.seed += 1

        self.signals.rect_ready_in_ui.emit()

    def create_outpaint_batch(self, gobig_img_path=None, progress_callback=False):
        self.batch_process = 'create_outpaint_batch'
        self.gobig_img_path = gobig_img_path
        tilesize = self.tile_size

        self.parent.params = self.parent.sessionparams.update_params()
        self.parent.sessionparams.params.advanced = True

        offset = self.parent.widgets[self.current_widget].w.mask_offset.value()
        overlap = self.parent.widgets[self.current_widget].w.rect_overlap.value()
        # self.preview_batch_outpaint()

        if self.gobig_img_path is not None:
            overlap_tilesize = self.tile_size - overlap
            upscale_factor = self.parent.widgets[self.current_widget].w.batch_upscale_factor.value()
            self.gobig_pil_image = Image.open(self.gobig_img_path)
            width, height = self.gobig_pil_image.size
            target_h = int(int(height) * upscale_factor)
            target_w = int(int(width) * upscale_factor)
            self.parent.canvas.H.setValue(int(target_h))
            self.parent.canvas.W.setValue(int(target_w))
            self.gobig_pil_image = self.gobig_pil_image.resize((target_w, target_h),Image.Resampling.LANCZOS).convert("RGBA")
            self.gobig_qimage = ImageQt(self.gobig_pil_image)
            #chops_x = int(qimage.width() / self.parent.canvas.canvas.w) + 1
            #chops_y = int(qimage.height() / self.parent.canvas.canvas.h) + 1
            chops_x = int(target_w / overlap_tilesize) + 1
            chops_y = int(target_h / overlap_tilesize) #+ 1
            print('chops_x, chops_y', chops_x, chops_y)
            print('chops_y claculated size',chops_y * overlap_tilesize)
            print('chops_x claculated size',chops_x * overlap_tilesize)
            self.parent.widgets[self.current_widget].w.rect_overlap.setValue(overlap)
            self.preview_batch_outpaint(chops_x=chops_x, chops_y=chops_y)


        self.rparams = self.parent.sessionparams.update_params()
        # print(self.tempsize_int)
        prompt_series = pd.Series([np.nan for a in range(self.tempsize_int)])
        # print(prompt_series)
        if self.rparams.keyframes == '':
            self.rparams.keyframes = "0"
        prom = self.rparams.prompts
        key = self.rparams.keyframes

        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))

        prompts = dict(zip(new_key, new_prom))

        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt
        self.animation_prompts = prompt_series.ffill().bfill()
        self.batch_step_number = 0
        self.parent.canvas.canvas.tempbatch_work = copy.deepcopy(self.parent.canvas.canvas.tempbatch)
        self.next_rect_from_batch()

    def image_ready_in_ui(self):
        self.parent.canvas.canvas.redraw()
        self.parent.render_index += 1
        self.parent.run_as_thread(self.next_image_from_batch)


    def next_image_from_batch(self, progress_callback=None):
        if self.batch_process == 'run_prepared_outpaint_batch':
            if len(self.rectlist_work) > 0:
                if gs.stop_all is False:
                    print(f"running step {self.batch_step_number}")
                    self.run_outpaint_step_x()

            else:
                print('stopped by user action')
                self.batch_process = None

        if self.batch_process == 'run_hires_batch':
            print('run next hires image batch')
            if gs.stop_all is False:
                if len(self.parent.canvas.canvas.rectlist) > 0 and len(self.parent.canvas.canvas.rectlist) > self.parent.render_index:
                    self.run_hires_step_x()
                else:
                    og_size = (512, 512)
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
                    for betterslice, x, y in self.betterslices:
                        #betterslice.save(f'output/betterslice_{x}_{y}.png')
                        finished_slice = self.addalpha(betterslice, mask)
                        finished_slices.append((finished_slice, x, y))
                    # # Once we have all our images, use grid_merge back onto the source, then save
                    final_output = self.grid_merge(
                        source_image.convert("RGBA"), finished_slices
                    ).convert("RGBA")

                    # todo make this filename a dynamic name
                    final_output.save('output/test_hires.png')
                    # base_filename = f"{base_filename}d"

                    self.hires_source = final_output
                    self.parent.params.advanced = False
                    self.finish_batch()
                    self.parent.image_preview_signal(final_output.convert("RGB"))
            else:
                print('stopped by user action')

    def finish_batch(self):
        self.parent.canvas.canvas.rectlist = []
        self.batch_process = None
        self.last_batch_image = None

    def run_hires_batch(self, progress_callback=None):
        self.batch_process = 'run_hires_batch'
        self.parent.sessionparams.update_params()
        self.parent.sessionparams.params.advanced = True

        gs.stop_all = False

        self.parent.choice = "Outpaint"

        self.betterslices = []

        self.parent.render_index = 0
        self.next_image_from_batch()


    def run_hires_step_x(self):
        next_step = self.parent.canvas.canvas.rectlist[self.parent.render_index]
        self.parent.choice = 'Outpaint'
        image = next_step.image
        image.save('output/temp/temp.png', "PNG")
        self.parent.canvas.canvas.selected_item = next_step.id
        print('self.parent.canvas.canvas.selected_item',self.parent.canvas.canvas.selected_item)
        self.parent.deforum_ui.run_deforum_six_txt2img(hiresinit='output/temp/temp.png')

    def run_outpaint_step_x(self):
        next_step = self.rectlist_work[0]
        del self.rectlist_work[0]
        self.parent.canvas.canvas.reusable_outpaint(next_step.id)
        #self.wait_canvas_busy()
        self.parent.deforum_ui.run_deforum_outpaint(next_step.params)

    def run_prepared_outpaint_batch(self, progress_callback=None):
        self.batch_process = 'run_prepared_outpaint_batch'
        gs.stop_all = False

        self.parent.choice = "Outpaint"
        self.parent.params.advanced = True

        self.batch_step_number = 0
        tiles = len(self.parent.canvas.canvas.rectlist)
        self.rectlist_work = copy.deepcopy(self.parent.canvas.canvas.rectlist)
        print(f"Tiles to Outpaint:{tiles}")
        self.next_image_from_batch()

    def resize_canvas(self):
        tilesize = 512
        overlap = (self.parent.widgets[self.current_widget].w.rect_overlap.value())
        overlap = overlap - (overlap / 3)

        target_h = (self.parent.widgets[self.current_widget].w.batch_rows.value() * (tilesize - overlap)) + (2 * self.parent.widgets[self.current_widget].w.start_offset_x.value()) + overlap
        target_w = (self.parent.widgets[self.current_widget].w.batch_columns.value() * (tilesize - overlap)) + (2 * self.parent.widgets[self.current_widget].w.start_offset_y.value()) + overlap
        print('targetsize = ', target_w, target_h)
        self.parent.canvas.H.setValue(int(target_h))
        self.parent.canvas.W.setValue(int(target_w))
        self.parent.canvas.canvas.change_resolution()

    def preview_batch_outpaint(self, chops_x=None, chops_y=None, progress_callback=None):
        # we resize canvas if preview batch
        # no resize for img2img
        if chops_x is None:
            self.resize_canvas()
            self.parent.canvas.canvas.scene.update()

        if chops_x is None:
            self.parent.canvas.canvas.cols = self.parent.widgets[self.current_widget].w.batch_columns.value()
            self.parent.canvas.canvas.rows = self.parent.widgets[self.current_widget].w.batch_rows.value() -1
        else:
            self.parent.canvas.canvas.cols = chops_x
            self.parent.canvas.canvas.rows = chops_y

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
            self.parent.canvas.canvas.tempbatch = spiralOrder(self.parent.canvas.canvas.tempbatch)
        if reverse:
            self.parent.canvas.canvas.tempbatch.reverse()
        # print(len(self.parent.canvas.canvas.tempbatch))
        self.tempsize_int = self.parent.canvas.canvas.cols * self.parent.canvas.canvas.rows
        self.parent.canvas.canvas.draw_tempBatch(self.parent.canvas.canvas.tempbatch)

    def outpaint_rect_overlap(self):
        self.parent.canvas.canvas.rectPreview = self.parent.widgets[self.current_widget].w.enable_overlap.isChecked()
        if self.parent.canvas.canvas.rectPreview == False:
            self.parent.canvas.canvas.newimage = True
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
