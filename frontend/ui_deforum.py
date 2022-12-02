import os
import time
import random

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PySide6.QtCore import QObject, Signal, QPoint
from PySide6.QtGui import QMouseEvent

from backend.deforum.deforum_adapter import DeforumSix
from backend.hypernetworks.modules.images import GridAnnotation
from backend.singleton import singleton
import torchvision.transforms as T
from torchvision.utils import make_grid
from einops import rearrange
from fonts.ttf import Roboto
from backend.worker import Worker
from backend.shared import model_killer
gs = singleton


class Callbacks(QObject):
    txt2img_step = Signal()
    reenable_runbutton = Signal()
    txt2img_image_cb = Signal()
    deforum_step = Signal()
    deforum_image_cb = Signal()
    compviscallback = Signal()
    add_image_to_thumbnail_signal = Signal(str)
    setStatusBar = Signal(str)
    vid2vid_one_percent = Signal(int)
    prepare_hires_batch = Signal(str)


class Deforum_UI(QObject):
    def __init__(self, parent):
        # super(QObject, self).__init__()
        self.renderedFrames = None
        self.currentFrames = None
        self.onePercent = None
        self.updateRate = None
        self.update = None
        self.progress = None
        self.deforum = None
        self.parent = parent
        #self.deforum = DeforumGenerator()
        self.deforum_six = DeforumSix(self)
        self.signals = Callbacks()
        #self.deforum_six = DeforumSix()
    def run(self):
        params = self.parent.sessionparams.update_params()
        ##print(f"updated parameters to: {params}")
        self.deforum_six.run_deforum_six(W=int(params.W),
                                         H=int(params.H),
                                         seed=int(params.seed) if params.seed != '' else -1,
                                         sampler=str(params.sampler),
                                         steps=int(params.steps),
                                         scale=float(params.scale),
                                         ddim_eta=float(params.ddim_eta),
                                         save_settings=bool(params.save_settings),
                                         save_samples=bool(params.save_samples),
                                         show_sample_per_step=bool(params.show_sample_per_step),
                                         n_batch=int(params.n_batch),
                                         seed_behavior=params.seed_behavior,
                                         make_grid=params.makegrid,
                                         grid_rows=params.grid_rows,
                                         use_init=params.use_init,
                                         init_image=params.init_image,
                                         strength=float(params.strength),
                                         strength_0_no_init=params.strength_0_no_init,
                                         device=params.device,
                                         animation_mode=params.animation_mode,
                                         prompts=params.prompts,
                                         max_frames=params.max_frames,
                                         outdir=params.outdir,
                                         n_samples=params.n_samples,
                                         mean_scale=params.mean_scale,
                                         var_scale=params.var_scale,
                                         exposure_scale=params.exposure_scale,
                                         exposure_target=params.exposure_target,
                                         colormatch_scale=float(params.colormatch_scale),
                                         colormatch_image=params.colormatch_image,
                                         colormatch_n_colors=params.colormatch_n_colors,
                                         ignore_sat_weight=params.ignore_sat_weight,
                                         clip_name=params.clip_name,
                                         # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                                         clip_scale=params.clip_scale,
                                         aesthetics_scale=params.aesthetics_scale,
                                         cutn=params.cutn,
                                         cut_pow=params.cut_pow,
                                         init_mse_scale=params.init_mse_scale,
                                         init_mse_image=params.init_mse_image,
                                         blue_scale=params.blue_scale,
                                         gradient_wrt=params.gradient_wrt,  # ["x", "x0_pred"]
                                         gradient_add_to=params.gradient_add_to,  # ["cond", "uncond", "both"]
                                         decode_method=params.decode_method,  # ["autoencoder","linear"]
                                         grad_threshold_type=params.grad_threshold_type,
                                         # ["dynamic", "static", "mean", "schedule"]
                                         clamp_grad_threshold=params.clamp_grad_threshold,
                                         clamp_start=params.clamp_start,
                                         clamp_stop=params.clamp_stop,
                                         grad_inject_timing=1,
                                         # if self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text() == '' else self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                         cond_uncond_sync=params.cond_uncond_sync,
                                         step_callback=self.parent.tensor_preview_signal if params.show_sample_per_step else None,
                                         image_callback=self.parent.image_preview_signal,
                                         negative_prompts=params.negative_prompts if params.negative_prompts != False else None,
                                         hires=params.hires,
                                         prompt_weighting=params.prompt_weighting,
                                         normalize_prompt_weights=params.normalize_prompt_weights,
                                         lowmem=params.lowmem,
                                         )

    def run_deforum_six_txt2img(self, progress_callback=None, plotting=True):
        gs.stop_all = False
        id = None
        if self.parent.canvas.canvas.rectlist != []:
            for i in self.parent.canvas.canvas.rectlist:
                try:
                    i.stop()
                except:
                    pass
                id = i.id
        self.parent.canvas.canvas.stop_main_clock()

        if id is not None:
            self.parent.canvas.canvas.render_item = id

        gs.karras = self.parent.widgets[self.parent.current_widget].w.karras.isChecked()
        self.params = self.parent.sessionparams.update_params()
        self.parent.params = self.params

        ##print(self.params.translation_x)
        ##print(f"updated parameters to: {params}")
        model_killer(keep='sd')
        #print(gs.models)
        #if "inpaint" in gs.models:
        #    del gs.models["inpaint"]

        if self.params.with_inpaint == True: # todo what is this for?
            self.parent.params.advanced = True
        else:
            self.parent.params.advanced = False

        gs.T = self.parent.widgets[self.parent.current_widget].w.gradient_steps.value()
        gs.lr = self.parent.widgets[self.parent.current_widget].w.gradient_scale.value()
        gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients, self.parent.widgets[self.parent.current_widget].w.aesthetic_embedding.currentText())
        if gs.aesthetic_embedding_path == 'None':
            gs.aesthetic_embedding_path = None
        seed = random.randint(0, 2 ** 32 - 1)

        plotting = self.params.plotting

        if plotting:

            attrib2 = self.params.plotX
            attrib1 = self.params.plotY

            ploty_list_string = self.params.plotXLine
            plotx_list_string = self.params.plotYLine
            plotY = plotx_list_string.split(', ')
            plotX = ploty_list_string.split(', ')
            self.onePercent = 100 / (len(plotX) * len(plotY) * self.params.n_batch * self.params.n_samples * self.params.steps)

        else:
            plotX = [1]
            plotY = [1]
            self.onePercent = 100 / (self.params.n_batch * self.params.n_samples * self.params.steps)
        all_images = []
        for i in plotY:
            for j in plotX:
                if plotting:
                    self.params.__dict__[attrib1] = i
                    self.params.__dict__[attrib2] = j
                    if attrib1 == 'T': gs.T = int(i)
                    if attrib1 == 'lr': gs.lr = float(i)
                    if attrib2 == 'T': gs.T = int(j)
                    if attrib2 == 'lr': gs.lr = float(j)
                if self.params.init_image is not None:
                    if os.path.isdir(self.params.init_image) and self.params.animation_mode == 'None':
                        print('Batch Directory found')
                        self.params.max_frames = 2
                self.deforum_six.run_deforum_six(W=int(self.params.W),
                                                 H=int(self.params.H),
                                                 seed=int(self.params.seed) if self.params.seed != '' else seed,
                                                 sampler=str(self.params.sampler),
                                                 steps=int(self.params.steps),
                                                 scale=float(self.params.scale),
                                                 ddim_eta=float(self.params.ddim_eta),
                                                 save_settings=bool(self.params.save_settings),
                                                 save_samples=bool(self.params.save_samples),
                                                 show_sample_per_step=bool(self.params.show_sample_per_step),
                                                 n_batch=int(self.params.n_batch),
                                                 seed_behavior=self.params.seed_behavior,
                                                 make_grid=self.params.make_grid,
                                                 grid_rows=self.params.grid_rows,
                                                 use_init=self.params.use_init,
                                                 init_image=self.params.init_image,
                                                 strength=float(self.params.strength),
                                                 strength_0_no_init=self.params.strength_0_no_init,
                                                 device=self.params.device,
                                                 animation_mode=self.params.animation_mode,
                                                 prompts=self.params.prompts,
                                                 max_frames=self.params.max_frames,
                                                 outdir=self.params.outdir,
                                                 n_samples=self.params.n_samples,
                                                 mean_scale=self.params.mean_scale,
                                                 var_scale=self.params.var_scale,
                                                 exposure_scale=self.params.exposure_scale,
                                                 exposure_target=self.params.exposure_target,
                                                 colormatch_scale=float(self.params.colormatch_scale),
                                                 colormatch_image=self.params.colormatch_image,
                                                 colormatch_n_colors=self.params.colormatch_n_colors,
                                                 ignore_sat_weight=self.params.ignore_sat_weight,
                                                 clip_name=self.params.clip_name,
                                                 # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                                                 clip_scale=self.params.clip_scale,
                                                 aesthetics_scale=self.params.aesthetics_scale,
                                                 cutn=self.params.cutn,
                                                 cut_pow=self.params.cut_pow,
                                                 init_mse_scale=self.params.init_mse_scale,
                                                 init_mse_image=self.params.init_mse_image,
                                                 blue_scale=self.params.blue_scale,
                                                 gradient_wrt=self.params.gradient_wrt,  # ["x", "x0_pred"]
                                                 gradient_add_to=self.params.gradient_add_to,  # ["cond", "uncond", "both"]
                                                 decode_method=self.params.decode_method,  # ["autoencoder","linear"]
                                                 grad_threshold_type=self.params.grad_threshold_type,
                                                 # ["dynamic", "static", "mean", "schedule"]
                                                 clamp_grad_threshold=self.params.clamp_grad_threshold,
                                                 clamp_start=self.params.clamp_start,
                                                 clamp_stop=self.params.clamp_stop,
                                                 grad_inject_timing=1,
                                                 # if self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text() == '' else self.parent.widgets[self.parent.current_widget].w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                                 cond_uncond_sync=self.params.cond_uncond_sync,
                                                 step_callback=self.parent.tensor_preview_signal if self.params.show_sample_per_step is not False else None,
                                                 image_callback=self.parent.image_preview_signal,
                                                 negative_prompts=self.params.negative_prompts if self.params.negative_prompts is not False else None,
                                                 hires=self.params.hires,
                                                 prompt_weighting=self.params.prompt_weighting,
                                                 normalize_prompt_weights=self.params.normalize_prompt_weights,
                                                 lowmem=self.params.lowmem,
                                                 )
                if plotting:
                    all_images.append(T.functional.pil_to_tensor(self.parent.image))
        if plotting:
            ver_texts = []
            hor_texts = []
            for i in plotY:
                ver_texts.append([GridAnnotation(f"{attrib1}: {i}")])
            for j in plotX:
                hor_texts.append([GridAnnotation(f"{attrib2}: {j}")])
            ##print(hor_texts)
            grid = make_grid(all_images, nrow=len(plotX))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{time.strftime('%Y%m%d%H%M%S')}_{attrib1}_{attrib2}_grid_{self.params.seed}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))

            grid_image = draw_grid_annotations(grid_image, grid_image.size[0], grid_image.size[1], hor_texts, ver_texts, self.params.W,
                                               self.params.H, self.params)
            self.parent.image = grid_image
            self.parent.image_preview_signal(grid_image)
            grid_image.save(os.path.join(self.params.outdir, filename))
        #self.signals.reenable_runbutton.emit()
        #self.deforum_six = None
        return

    def run_deforum_outpaint(self, params=None, progress_callback=None):
        # self.deforum = DeforumGenerator()
        # self.deforum.signals = Callbacks()
        self.parent.params = self.parent.sessionparams.update_params()
        params = self.parent.params
        self.deforum_six = DeforumSix()
        self.progress = 0.0
        self.parent.update = 0
        self.onePercent = 100 / params.steps
        #self.updateRate = self.parent.sizer_count.w.previewSlider.value()
        self.updateRate = 1
        self.parent.currentFrames = []
        self.parent.renderedFrames = 0
        self.parent.sample_number = 1
        if params.n_samples == 1:
            makegrid = False
        else:
            makegrid = self.parent.animKeys.w.makeGrid.isChecked()
        #sampler_name = translate_sampler(self.parent.sampler.w.sampler.currentText())
        sampler_name = "ddim"
        init_image = "outpaint.png"
        gs.T = self.parent.widgets[self.parent.current_widget].w.gradient_steps.value()
        gs.lr = self.parent.widgets[self.parent.current_widget].w.gradient_scale.value() / 1000000000
        gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients, self.parent.widgets[self.parent.current_widget].w.aesthetic_embedding.currentText())
        #if params == None:
        params = self.parent.sessionparams.update_params()

        #if params is not None:
        ##print(params)
        steps = int(params.steps)
        H = int(params.H)
        W = int(params.W)
        seed = int(params.seed) if params.seed != "" else random.randint(0, 44444444)
        prompt = str(params.prompts)
        strength = float(params.strength)
        mask_blur = float(params.mask_blur)
        reconstruction_blur = float(params.recons_blur)
        scale = float(params.scale)
        ddim_eta = float(params.ddim_eta)
        with_inpaint = bool(params.with_inpaint)

        self.parent.sessionparams.params.advanced = True

        self.deforum_six.outpaint_txt2img(init_image=init_image,
                                          steps=steps,
                                          H=H,
                                          W=W,
                                          seed=seed,
                                          prompt=prompt,
                                          strength=strength,
                                          mask_blur=mask_blur,
                                          recons_blur=reconstruction_blur,
                                          scale=scale,
                                          ddim_eta=ddim_eta,
                                          image_callback=self.parent.image_preview_signal,
                                          step_callback=self.parent.tensor_preview_signal,
                                          with_inpaint=with_inpaint)

        # self.run_txt2img_lm(init_img=init_image, init_mask='outpaint_mask.png')

        self.signals.reenable_runbutton.emit()

    def deforum_outpaint_thread(self):

        self.parent.sessionparams.params = self.parent.sessionparams.update_params()
        self.choice = "Outpaint"
        worker = Worker(self.run_deforum_outpaint)
        self.parent.threadpool.start(worker)
class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None

def draw_grid_annotations(im, width, height, hor_texts, ver_texts, W, H, params):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center")

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (W + H) // 100
    line_spacing = fontsize // 2

    try:
        fnt = ImageFont.truetype(Roboto, fontsize)
    except Exception:
        fnt = ImageFont.truetype(Roboto, fontsize)

    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else W // 4

    cols = im.width // W
    rows = im.height // H

    ##print(f"DEBUG: {cols}, {rows}, of which at least one should be more then 1...")

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [W] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in
                        ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)
    #p_pad = len(params["prompts"][0]) * 1.75
    #d.multiline_text(((pad_left / 2) + p_pad, pad_top / 2), params["prompts"][0], font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="left")

    for col in range(cols):
        x = pad_left + W * col + W / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + H * row + H / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result
