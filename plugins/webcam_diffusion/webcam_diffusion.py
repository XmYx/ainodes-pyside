"""This is an example plugin file that you can use to build your own plugins for aiNodes.

Welcome to aiNodes. Please refer to PySide 6.4 documentation for UI functions.

Please also note the following features at your disposal by default:
DeforumSix
Hypernetworks
Singleton

at plugin loading time, the plugins initme function will be called automatically to make sure that
all defaults are set correctly, and that your new UI element is loaded, with its signals and slots connected.

Your plugin's parent is the MainWindow, and by default, it has a canvas loaded. You can access all of its functions,
such as addrect_atpos, and image_preview_func (make sure to set self.parent.image before doing so).

It is good to know, that if you are doing heavy lifting, you have to use its own QThreadPool, otherwise your gui freezes
while processing. To do so, just use the worker from backend.worker

        worker = Worker(self.parent.deforum_ui.run_deforum_six_txt2img)
        self.parent.threadpool.start(worker)

It is also worth mentioning, that ui should only be modified from the main thread, therefore when displaying an image,
set self.parent.image, then call self.parent.image_preview_signal, which will emit a signal to call
the image_preview_func from the main thread.
"""
import hashlib
import subprocess

import safetensors

from backend.aesthetics.aesthetic_clip import AestheticCLIP
from backend.deforum.deforum_adapter import load_vae_dict, vae_ignore_keys
from backend.deforum.six.model_load import make_linear_decode
from backend.deforum.six.seamless import configure_model_padding
from backend.hypernetworks.hypernetwork import apply_strength
from backend.singleton import singleton
from backend.torch_gc import torch_gc
from backend.hypernetworks import hypernetwork
import backend.hypernetworks.modules.sd_hijack
from backend.deforum.six.hijack import hijack_deforum

gs = singleton
import os, sys
import random
import traceback
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Signal, QObject, QThreadPool, Slot, QRunnable
from PySide6.QtWidgets import QTextEdit, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QLabel, QFileDialog, QCheckBox
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from backend.deforum.six.conditioning import threshold_by
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser, VDenoiser, DiscreteVDDPMDenoiser, OpenAIDenoiser
from backend.deforum.six.model_wrap import CFGDenoiserWithGrad
from ldm.util import instantiate_from_config
from PIL.ImageQt import ImageQt
from PIL import Image
from k_diffusion import sampling
from types import SimpleNamespace
from typing import Any, Callable, Optional

from torch import nn, autocast, optim
import torch
from backend.resizeRight import resizeright, interp_methods
torch.set_grad_enabled(False)

import cv2
import numpy as np

class aiNodesPlugin():
    def __init__(self, parent):
        self.parent = parent

    def initme(self):
        #cmd = ["pip", "install", "ffmpeg", "--upgrade"]
        #subprocess.Popen(cmd)
        cmd = ["pip", "install", "sk-video", "--upgrade"]
        subprocess.Popen(cmd)
        sshFile = "frontend/style/QTDark.stylesheet"
        self.widget = WebcamWidget()
        self.widget.setWindowTitle("Webcam Diffusion")
        with open(sshFile, "r") as fh:
            self.widget.setStyleSheet(fh.read())
        self.widget.show()

class WebcamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = OurSignals()
        self.signals.updateimagesignal.connect(self.update_image)
        self.signals.webcamupdate.connect(self.update_frame_func)
        # Set up the user interface
        self.capture_button = QtWidgets.QPushButton('Stop')
        self.capture_button.clicked.connect(self.stop_threads)
        self.continous = QtWidgets.QPushButton('Start Webcam Diffusion')
        self.continous.clicked.connect(self.start_continuous_capture)
        self.video_input = QtWidgets.QPushButton('Start Video Input Diffusion')
        self.video_input.clicked.connect(self.start_video_input)
        self.webcam_dropdown = QtWidgets.QComboBox()
        self.webcam_dropdown.addItems(self.get_available_webcams())
        self.webcam_dropdown.currentIndexChanged.connect(self.start_webcam)
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(512, 512)

        self.maskprompt = QTextEdit()
        self.prompt = QTextEdit()

        self.steps = QSpinBox()
        self.steps.setValue(12)
        self.steps.valueChanged.connect(self.make_sampler_schedule)

        self.strength = QDoubleSpinBox()
        self.strength.setValue(0.50)
        self.strength.setMaximum(1.00)
        self.strength.setMinimum(0.01)
        self.strength.setSingleStep(0.01)
        self.strength.valueChanged.connect(self.make_sampler_schedule)

        self.eta = QDoubleSpinBox()
        self.eta.setValue(0.0)
        self.eta.setMaximum(1.00)
        self.eta.setMinimum(0.00)
        self.eta.setSingleStep(0.01)

        self.rescalefactor = QDoubleSpinBox()
        self.rescalefactorlabel = QLabel("rescalefactor")
        self.rescalefactor.setValue(1.0)
        self.rescalefactor.setMinimum(1.0)
        self.rescalefactor.setMaximum(10.0)
        self.rescalefactor.setSingleStep(0.01)

        self.seed = QLineEdit()
        self.samplercombobox = QComboBox()
        self.samplercombobox.addItems(["euler", "dpm2", "dpm2_ancestral", "heun", "klms", "euler_ancestral",
                                        "dpm_fast", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_adaptive"])
        self.modelselect = QComboBox()
        self.modelselect.addItems(["Normal"])

        self.save_video_stream = QCheckBox("Save Video Stream")
        self.save_frames = QCheckBox("Save Individual Frames")
        # Set up the layout
        # Create a QGridLayout object
        layout = QtWidgets.QGridLayout()

        # Add the widgets to the grid layout
        layout.addWidget(QtWidgets.QLabel("Prompt:"), 0, 0)
        layout.addWidget(self.prompt, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Steps:"), 1, 0)
        layout.addWidget(self.steps, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Strength:"), 2, 0)
        layout.addWidget(self.strength, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Rescale factor:"), 3, 0)
        layout.addWidget(self.rescalefactor, 3, 1)
        layout.addWidget(QtWidgets.QLabel("Eta:"), 4, 0)
        layout.addWidget(self.eta, 4, 1)
        layout.addWidget(QtWidgets.QLabel("Seed:"), 5, 0)
        layout.addWidget(self.seed, 5, 1)
        layout.addWidget(QtWidgets.QLabel("Sampler:"), 6, 0)
        layout.addWidget(self.samplercombobox, 6, 1)
        layout.addWidget(QtWidgets.QLabel("Model:"), 7, 0)
        layout.addWidget(self.modelselect, 7, 1)
        layout.addWidget(QtWidgets.QLabel("Webcam:"), 8, 0)
        layout.addWidget(self.webcam_dropdown, 8, 1)
        layout.addWidget(self.capture_button, 9, 0)
        layout.addWidget(self.continous, 9, 1)
        layout.addWidget(self.save_video_stream, 10, 0)
        layout.addWidget(self.video_input, 10, 1)
        layout.addWidget(self.save_frames, 11, 0)
        #layout.addWidget(self.camera_label)
        self.setLayout(layout)
        self.threadpool = QThreadPool()
        # Start the webcam
        self.start_webcam()
        self.morphed_images = []
        #Show the preview window
        self.image_label = QtWidgets.QLabel()
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Create a QDialog and set the layout to the layout containing the image label
        self.image_dialog = QtWidgets.QWidget()
        prevlayout = QtWidgets.QVBoxLayout()
        self.fullscreen_button = QtWidgets.QPushButton('Fullscreen')
        self.fullscreen_button.clicked.connect(self.show_fullscreen)
        prevlayout.addWidget(self.fullscreen_button)
        prevlayout.addWidget(self.image_label)
        self.image_dialog.setLayout(prevlayout)
        sshFile = "frontend/style/QTDark.stylesheet"
        with open(sshFile, "r") as fh:
            self.image_dialog.setStyleSheet(fh.read())
        self.image_dialog.show()
        self.loadedmodel = None


    def show_fullscreen(self):
        if self.image_dialog.isFullScreen():
            self.image_dialog.showNormal()
        else:
            self.image_dialog.showFullScreen()
    def get_available_webcams(self):
        """Get a list of available webcams."""
        webcams = []
        for i in range(10):
            capture = cv2.VideoCapture(i)
            if capture.isOpened():
                webcams.append(f'Webcam {i}')
            capture.release()
        return webcams

    def start_webcam(self):
        """Start the webcam and display the video feed."""
        self.capture = cv2.VideoCapture(self.webcam_dropdown.currentIndex())

    def update_frame(self):
        """Update the camera preview label with the latest frame."""
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.webcamimage = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.signals.webcamupdate.emit()

    @Slot()
    def update_frame_func(self):
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(self.webcamimage))
    def stop_threads(self):
        self.run = False

    def start_video_input(self):

        inference = self.modelselect.currentText()
        if self.loadedmodel != "normal":
            #gs.models["sd"] = load_model_from_config("data/models/v1-5-pruned-emaonly.yaml",
            #                                    "data/models/v1-5-pruned-emaonly.ckpt")
            self.loadedmodel = "normal"
            self.midas_trafo = None
            self.sampler = None

        """Start the continuous capture in a separate thread."""
        self.run = True
        self.select_video_file()
        #self.continous_init_latent()
        torch.cuda.empty_cache()
        worker = Worker(self.process_video)
        self.threadpool.start(worker)
        self.index = 0
    def start_continuous_capture(self):

        inference = self.modelselect.currentText()
        if self.loadedmodel != "normal":
           # gs.models["sd"] = load_model_from_config("data/models/v1-5-pruned-emaonly.yaml",
           #                                     "data/models/v1-5-pruned-emaonly.ckpt")
            self.loadedmodel = "normal"
            self.midas_trafo = None
            self.sampler = None

        """Start the continuous capture in a separate thread."""
        self.run = True

        #self.continous_init_latent()
        torch.cuda.empty_cache()
        worker = Worker(self.continuous_capture)
        self.threadpool.start(worker)
        self.index = 0
    def make_sampler_schedule(self):
        steps = self.steps.value()
        eta = self.eta.value()
        if self.loadedmodel == 'normal':
            #self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
            self.sigmas = sampling.get_sigmas_karras(n=self.steps.value(), sigma_min=0.1, sigma_max=10, device="cuda")
            self.sigmas = self.sigmas[len(self.sigmas) - int(self.strength.value() * self.steps.value()) - 1:]

    def continuous_capture(self, progress_callback=None):
        """Capture images from the webcam continuously."""
        # State for interpolating between diffusion steps
        self.prepare_for_run()
        self.images = []
        self.index = 0
        self.lastinit = None
        self.seedint = 0
        if self.save_video_stream.isChecked():
            import skvideo.io
            skvideo.setFFmpegPath("ffmpeg.exe")
            output_path = "out_stream.mp4"
            frame_rate = 24
            writer = skvideo.io.FFmpegWriter(output_path, outputdict={'-r': str(frame_rate)})
        if self.loadedmodel == "normal":
            with autocast("cuda"):
                self.return_seedint()
                seed_everything(self.seedint)
                self.uc = gs.models["sd"].get_learned_conditioning(1 * [""])
                self.promptstring = self.prompt.toPlainText()
                self.c = gs.models["sd"].get_learned_conditioning(self.promptstring)
                self.sigmas = sampling.get_sigmas_karras(n=self.steps.value(), sigma_min=0.1, sigma_max=10, device="cuda")
                self.sigmas = self.sigmas[len(self.sigmas) - int(self.strength.value() * self.steps.value()) - 1:]
            self.model_wrap = CompVisDenoiser(gs.models["sd"], quantize=False)

            loss_fns_scales = [
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0]
            ]
            clamp_fn = threshold_by(threshold=0, threshold_type='dynamic',
                                    clamp_schedule=[0])
            grad_inject_timing_fn = make_inject_timing_fn(1, self.model_wrap, 10)
            self.cfg_model = CFGDenoiserWithGrad(self.model_wrap,
                                            loss_fns_scales,
                                            clamp_fn,
                                            None,
                                            None,
                                            True,
                                            decode_method=None,
                                            grad_inject_timing_fn=grad_inject_timing_fn,
                                            grad_consolidate_fn=None,
                                            verbose=False)
        _, frame = self.capture.read()
        self.args = SimpleNamespace()
        self.args.use_init = True
        self.args.scale = 7.5
        self.args.sampler = self.samplercombobox.currentText()
        self.args.n_samples = 1
        self.args.C = 4
        self.args.f = 8
        self.image_label.setScaledContents(True)
        self.args.log_weighted_subprompts = False
        self.args.normalize_prompt_weights = False
        #torch.backends.cudnn.benchmark = True
        frame_count = 0
        while self.run == True:
            with torch.autocast("cuda"):
                _, frame = self.capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Call the predict function and get the resulting image
                self.return_seedint()
                prompt = self.prompt.toPlainText()
                steps = self.steps.value()
                strength = self.strength.value()
                eta = self.eta.value()
                if prompt != self.prompt:
                    self.promptstring = self.prompt.toPlainText()
                    self.c = gs.models["sd"].get_learned_conditioning(self.promptstring)
                self.images = [self.img2img(frame, prompt,
                                            steps, 1, 7.5, self.seedint, eta, strength)]
                frame_count += 1
                if self.save_frames.isChecked():
                    filepath = f"video_out/webcam_out_frame_{frame_count}.png"
                    self.images[0].save(filepath)
                if self.save_video_stream.isChecked():
                    # Add the resulting image to the output video
                    writer.writeFrame(self.images[0])
                self.update_image_signal()
            if self.run == False:
                if self.save_video_stream.isChecked():
                    writer.close()
                break
        if self.save_video_stream.isChecked():
            try:
                writer.close()
            except:
                pass
    def prepare_for_run(self):
        check = self.load_model_from_config(config=None, ckpt=None)
        if check == -1:
            return check

        if gs.diffusion.selected_hypernetwork != 'None':
            hypernetwork.load_hypernetwork(gs.diffusion.selected_hypernetwork)
            hypernetwork.apply_strength(
                apply_strength)  # 1.0, "Hypernetwork strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
            gs.model_hijack.apply_circular(False)
            gs.model_hijack.clear_comments()

        # W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

        # if args.seamless == True and self.prev_seamless == False:

        # print("Running Seamless sampling...")
        seamless = False
        seamless_axes = ["x"]
        configure_model_padding(gs.models["sd"], seamless, seamless_axes)
        # self.prev_seamless = True
        """
        for key, value in root.__dict__.items():
            try:
                root.__dict__[key] = self.parent.params.__dict__[key]
            except:
                pass"""

        if gs.diffusion.selected_aesthetic_embedding != 'None' and gs.model_version == "1.5":
            gs.models["sd"].cond_stage_model.process_tokens.set_aesthetic_params(
                aesthetic_lr=gs.lr,
                aesthetic_weight=gs.aesthetic_weight,
                aesthetic_steps=gs.T,
                image_embs_name=gs.diffusion.selected_aesthetic_embedding,
                aesthetic_slerp=gs.slerp,
                aesthetic_imgs_text=gs.aesthetic_imgs_text,
                aesthetic_slerp_angle=gs.slerp_angle,
                aesthetic_text_negative=gs.aesthetic_text_negative)

    def select_video_file(self):
        self.filename = QFileDialog.getOpenFileName(caption='Select Init video', filter='Video (*.mp4 *.mov)')
        #print(self.filename[0])

    def process_video(self, video_path = None, progress_callback = None):
        self.prepare_for_run()
        video_path=self.filename[0]
        # Create a VideoCapture object for reading the video file
        capture = cv2.VideoCapture(video_path)
        # Read the first frame to get the video dimensions
        success, frame = capture.read()
        factor = self.rescalefactor.value()
        frame_height, frame_width, _ = frame.shape
        frame_height = frame_height // factor
        frame_width = frame_width // factor


        frame_rate = 24
        output_codec = "libx264"
        # Create a ffmpeg writer to write the output video

        if self.save_video_stream.isChecked():
            import skvideo.io
            skvideo.setFFmpegPath("ffmpeg.exe")
            output_path = "out_video.mp4"
            writer = skvideo.io.FFmpegWriter(output_path, outputdict={'-r': str(frame_rate)})
        os.makedirs("video_out", exist_ok=True)

        self.images = []
        self.index = 0
        self.lastinit = None
        self.seedint = 0
        self.run = True
        #if self.loadedmodel == "inpaint":
        #    gs.models["sd"] = None
        if self.loadedmodel == "normal":
            with autocast("cuda"):
                self.return_seedint()
                seed_everything(self.seedint)
                self.uc = gs.models["sd"].get_learned_conditioning(1 * [""])
                self.promptstring = self.prompt.toPlainText()
                self.c = gs.models["sd"].get_learned_conditioning(self.promptstring)
                self.sigmas = sampling.get_sigmas_karras(n=self.steps.value(), sigma_min=0.1, sigma_max=10, device="cuda")
                self.sigmas = self.sigmas[len(self.sigmas) - int(self.strength.value() * self.steps.value()) - 1:]
            #self.init_mask_model()
            self.model_wrap = CompVisDenoiser(gs.models["sd"], quantize=False)
            loss_fns_scales = [
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0],
                [None, 0.0]
            ]
            clamp_fn = threshold_by(threshold=0, threshold_type='dynamic',
                                    clamp_schedule=[0])
            grad_inject_timing_fn = make_inject_timing_fn(1, self.model_wrap, 10)
            self.cfg_model = CFGDenoiserWithGrad(self.model_wrap,
                                            loss_fns_scales,
                                            clamp_fn,
                                            None,
                                            None,
                                            True,
                                            decode_method=None,
                                            grad_inject_timing_fn=grad_inject_timing_fn,
                                            grad_consolidate_fn=None,
                                            verbose=False)
        self.args = SimpleNamespace()
        self.args.use_init = True
        self.args.scale = 7.5
        self.args.sampler = self.samplercombobox.currentText()
        self.args.n_samples = 1
        self.args.C = 4
        self.args.f = 8
        self.image_label.setScaledContents(True)
        self.args.log_weighted_subprompts = False
        self.args.normalize_prompt_weights = False
        # Initialize a counter variable to skip frames
        frame_count = 0
        frame_skip = 2
        # Iterate through all frames in the video
        while success:
            if self.run == True:
                with torch.autocast("cuda"):
                    # Convert the frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Call the predict function and get the resulting image
                    prompt = self.prompt.toPlainText()
                    steps = self.steps.value()
                    strength = self.strength.value()
                    eta = self.eta.value()
                    if prompt != self.prompt:
                        self.promptstring = self.prompt.toPlainText()
                        self.c = gs.models["sd"].get_learned_conditioning(self.promptstring)
                    self.seedint = ""
                    self.return_seedint()
                    result_image = self.img2img(frame, prompt, steps, 1, 7.5, self.seedint, eta, strength)
                    self.images = [result_image]
                    frame_count += 1
                    if self.save_frames.isChecked():
                        filepath = f"video_out/video_out_frame_{frame_count}.png"
                        self.images[0].save(filepath)
                    self.update_image_signal()
                    if self.save_video_stream.isChecked():
                        # Add the resulting image to the output video
                        writer.writeFrame(result_image)
                    # Increment the counter variable

                    # Skip the next `frame_skip` frames
                    for i in range(frame_skip):
                        success, frame = capture.read()
            else:
                if self.save_video_stream.isChecked():
                    writer.close()
                capture.release()
                break
        if self.save_video_stream.isChecked():
            writer.close()
        capture.release()
        # Close the ffmpeg writer and the VideoCapture object
    def return_seedint(self):
        self.seedint = self.seed.text() if self.seed.text() != '' else self.seedint
        self.seedint = int(self.seedint) + 1 if self.seedint != '' else random.randint(0, 4000000)
    def update_image_signal(self):
        self.signals.updateimagesignal.emit()
    @Slot()
    def update_image(self):
        if self.images != []:
            self.image_label.setPixmap(
                QtGui.QPixmap.fromImage(ImageQt(self.images[len(self.images) - 1])))

    def img2img(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        image = input_image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.*image - 1.
        image = image.half().to("cuda")
        t_enc = int(strength * steps)
        torch.cuda.manual_seed_all(seed)

        #with torch.autocast("cuda"):
        torch.backends.cudnn.benchmark = True
        self.init_latent = gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(image))
        factor = self.rescalefactor.value()
        if factor != 1.0:
            self.init_latent = resizeright.resize(self.init_latent, scale_factors=None,
                                         out_shape=[self.init_latent.shape[0], self.init_latent.shape[1], int(self.init_latent.shape[2] // factor),
                                                    int(self.init_latent.shape[3] // factor)],
                                         interp_method=interp_methods.lanczos3, support_sz=None,
                                         antialiasing=False, by_convs=True, scale_tolerance=None,
                                         max_numerator=10, pad_mode='reflect')
        self.args.W = self.init_latent.shape[3] * 8
        self.args.H = self.init_latent.shape[2] * 8
        self.args.steps = steps
        self.args.seed = seed
        self.args.sampler = self.samplercombobox.currentText()
        samples = self.sampler_fn(
            c=self.c,
            uc=self.uc,
            args=self.args,
            model_wrap=self.cfg_model,
            init_latent=self.init_latent,
            t_enc=t_enc,
            device="cuda",
            cb=None,
            verbose=False)
        x_samples = gs.models["sd"].decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
        image = Image.fromarray(x_sample.astype(np.uint8))
        return image

    def predict(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        do_full_sample = strength == 1.
        t_enc = min(int(strength * steps), steps-1)
        input_image = Image.fromarray(input_image)
        width, height = input_image.size
        result = self.paint(
            sampler=self.sampler,
            model=self.sampler.model,
            image=input_image,
            image_quad=input_image,
            prompt=prompt,
            t_enc=t_enc,
            seed=seed,
            scale=scale,
            num_samples=num_samples,
            callback=None,
            do_full_sample=do_full_sample
            )
        return result

    def sampler_fn(
            self,
            c: torch.Tensor,
            uc: torch.Tensor,
            args,
            model_wrap: CompVisVDenoiser,
            init_latent: Optional[torch.Tensor] = None,
            t_enc: Optional[torch.Tensor] = None,
            device=torch.device("cpu")
            if not torch.cuda.is_available()
            else torch.device("cuda"),
            cb: Callable[[Any], None] = None,
            verbose: Optional[bool] = False,
    ) -> torch.Tensor:
        shape = [args.C, args.H // args.f, args.W // args.f]

        x = (
                init_latent
                + torch.randn([args.n_samples, *shape], device=device) * self.sigmas[0]
        )
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigmas": self.sigmas,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": True,
            "callback": None,
        }
        if args.sampler in ["dpm_fast"]:
            min = self.sigmas[0].item()
            max = min
            for i in self.sigmas:
                if i.item() < min and i.item() != 0.0:
                    min = i.item()

            sampler_args = {
                "model": model_wrap,
                "x": x,
                "sigma_min": min,
                "sigma_max": max,
                "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
                "disable": True,
                "callback": None,
                "n": args.steps,
                "eta": 0.0,
                "s_noise": 1.0,
            }
        elif args.sampler in ["dpm_adaptive"]:
            min = self.sigmas[0].item()
            max = min
            for i in self.sigmas:
                if i.item() < min and i.item() != 0.0:
                    min = i.item()

            sampler_args = {
                "model": model_wrap,
                "x": x,
                "sigma_min": min,
                "sigma_max": max,
                "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
                "disable": True,
                "callback": None,
                "order": 3,
                "rtol": 0.05,
                "atol": 0.0078,
                "h_init": 0.05,
                "pcoeff": 0.0,
                "icoeff": 1.0,
                "dcoeff": 0.0,
                "eta": 0.0,
                "s_noise": 1.0,
            }

        elif args.sampler in ["dpmpp_sde", "dpmpp_2s_a"]:
            sampler_args = {
                "model": model_wrap,
                "x": x,
                "sigmas": self.sigmas,
                "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
                "disable": True,
                "callback": None,
                "eta": 1.0,
                "s_noise": 1.0,
            }

        sampler_map = {
            "klms": sampling.sample_lms,
            "dpm2": sampling.sample_dpm_2,
            "dpm2_ancestral": sampling.sample_dpm_2_ancestral,
            "heun": sampling.sample_heun,
            "euler": sampling.sample_euler,
            "euler_ancestral": sampling.sample_euler_ancestral,
            "dpm_fast": sampling.sample_dpm_fast,
            "dpm_adaptive": sampling.sample_dpm_adaptive,
            "dpmpp_2s_a": sampling.sample_dpmpp_2s_ancestral,
            "dpmpp_2m": sampling.sample_dpmpp_2m,
            "dpmpp_sde": sampling.sample_dpmpp_sde,
        }

        samples = sampler_map[args.sampler](**sampler_args)
        return samples

    def run_post_load_model_generation_specifics(self):

        # print("Loading Hypaaaa")
        gs.model_hijack = backend.hypernetworks.modules.sd_hijack.StableDiffusionModelHijack()

        print("hijacking??")
        gs.model_hijack.hijack(gs.models["sd"])
        gs.model_hijack.embedding_db.load_textual_inversion_embeddings()

        # gs.models["sd"].cond_stage_model = backend.aesthetics.modules.PersonalizedCLIPEmbedder()

        aesthetic = AestheticCLIP()
        aesthetic.process_tokens = gs.models["sd"].cond_stage_model.process_tokens
        gs.models["sd"].cond_stage_model.process_tokens = aesthetic

    def get_autoencoder_version(self):
        return "sd-v1"  # TODO this will be different for different models

    def transform_checkpoint_dict_key(self, k):
        chckpoint_dict_replacements = {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
        }
        for text, replacement in chckpoint_dict_replacements.items():
            if k.startswith(text):
                k = replacement + k[len(text):]

        return k

    def get_state_dict_from_checkpoint(self, pl_sd):
        pl_sd = pl_sd.pop("state_dict", pl_sd)
        pl_sd.pop("state_dict", None)

        sd = {}
        for k, v in pl_sd.items():
            new_key = self.transform_checkpoint_dict_key(k)

            if new_key is not None:
                sd[new_key] = v

        pl_sd.clear()
        pl_sd.update(sd)

        return pl_sd

    """
    512-base-ema.ckpt d635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824
    512-inpainting-ema.ckpt 2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3
    768-v-ema.ckpt bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f
    sd-v1-4.ckpt fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556
    sd-v1-5-inpainting.ckpt c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19
    v1-5-pruned-emaonly.ckpt cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516
    data/models/v2-1_512-ema-pruned.ckpt 88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36
    data/models/v2-1_768-ema-pruned.ckpt ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0    
    """

    def return_model_version(self, model):
        with open(model, 'rb') as file:
            # Read the contents of the file
            file_contents = file.read()

            # Calculate the SHA-256 hash
            sha256_hash = hashlib.sha256(file_contents).hexdigest()
            if sha256_hash == 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824':
                version = '2.0 512'
                config = '512-base-ema.yaml'
            elif sha256_hash == '2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3':
                version = '2.0 Inpaint'
                config = '512-inpainting-ema.yaml'
            elif sha256_hash == 'bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f':
                version = '2.0 768'
                config = '768-v-ema.yaml'
            elif sha256_hash == 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556':
                version = '1.4'
                config = 'sd-v1-4.yaml'
            elif sha256_hash == 'c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19':
                version = '1.5 Inpaint'
                config = 'sd-v1-5-inpainting.yaml'
            elif sha256_hash == 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516':
                version = '1.5 EMA Only'
                config = 'v1-5-pruned-emaonly.yaml'
            elif sha256_hash == '88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36':
                version = '2.1 512'
                config = 'v2-1_512-ema-pruned.yaml'
            elif sha256_hash == 'ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0':
                version = '2.1 768'
                config = 'v2-1_768-ema-pruned.yaml'
            else:
                version = 'unknown'
                config = None
            # Print the hash
            return config, version

    def load_model_from_config(self, config=None, ckpt=None, verbose=False):
        gs.force_inpaint = False
        if ckpt is None:
            ckpt = gs.system.sd_model_file
        # Open the file in binary mode

        # loads config.yaml with the name of the model
        # the config yaml has to be provided with pöropper naming,
        # otherwise it is not anymore possible to do all the magic with multiple versions of the model around
        # also config.yaml needs to have one entry at root model_version
        # model_version has to be explicid like 1.4 or 1.5 or 2.0
        # it is important that you give the right version hint based on the SD model version
        # if it is some custom model based on some version of SD we need to have the SD
        # version not the version of the custom model
        # if config is None:
        # config_yaml_name = os.path.splitext(ckpt)[0] + '.yaml'
        # if not os.path.exists(config_yaml_name):
        #    config_yaml_name = 'data/default_configs/v1-5.yaml'
        config, version = self.return_model_version(ckpt)
        if 'Inpaint' in version:
            gs.force_inpaint = True
            print("Forcing Inpaint")
        if config == None:
            config = os.path.splitext(ckpt)[0] + '.yaml'
        else:
            config = os.path.join('data/models', config)
        # print(config_yaml_name)
        # else:
        #    config_yaml_name = config
        # print(os.path.isfile(config_yaml_name))
        # if os.path.isfile(config_yaml_name):
        # config = config_yaml_name

        if "sd" not in gs.models:
            self.prev_seamless = False
            if verbose:
                print(f"Loading model from {ckpt} with config {config}")
            config = OmegaConf.load(config)

            # print(config.model['params'])

            if 'num_heads' in config.model['params']['unet_config']['params']:
                gs.model_version = '1.5'
            elif 'num_head_channels' in config.model['params']['unet_config']['params']:
                gs.model_version = '2.0'
            if config.model['params']['conditioning_key'] == 'hybrid-adm':
                gs.model_version = '2.0'
            if 'parameterization' in config.model['params']:
                gs.model_resolution = 768
            else:
                gs.model_resolution = 512
            # if not 'model_version' in config:
            #    print('you must provide a model_version in the config yaml or we can not figure how to tread your model')
            #    return -1
            print(f'{ckpt}: v{gs.model_version} found with resolution {gs.model_resolution}')

            # gs.model_version = config.model_version
            if verbose:
                print(gs.model_version)

            checkpoint_file = ckpt
            _, extension = os.path.splitext(checkpoint_file)
            map_location = "cpu"
            if extension.lower() == ".safetensors":
                pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
            else:
                pl_sd = torch.load(checkpoint_file, map_location=map_location)
            # pl_sd = torch.load(ckpt, map_location="cpu")

            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = self.get_state_dict_from_checkpoint(pl_sd)
            # sd = pl_sd["state_dict"]

            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
            model.half()
            gs.models["sd"] = model
            gs.models["sd"].cond_stage_model.device = "cuda"
            # gs.models["sd"].embedding_manager = EmbeddingManager(gs.models["sd"].cond_stage_model)
            # embedding_path = '001glitch-core.pt'
            # if embedding_path is not None:
            #    gs.models["sd"].embedding_manager.load(
            #        embedding_path
            #    )

            for m in gs.models["sd"].modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m._orig_padding_mode = m.padding_mode

            autoencoder_version = self.get_autoencoder_version()

            gs.models["sd"].linear_decode = make_linear_decode(autoencoder_version, "cuda")
            del pl_sd
            del sd
            del m, u
            del model
            torch_gc()

            if gs.model_version == '1.5' and not 'Inpaint' in version:
                self.run_post_load_model_generation_specifics()

            gs.models["sd"].eval()

            # todo make this 'cuda' a parameter
            gs.models["sd"].to("cuda")
            # todo why we do this here?
            from backend.aesthetics import modules
            #self.next_get_post_op()
            if gs.diffusion.selected_vae != 'None':
                self.load_vae(gs.diffusion.selected_vae)

    def load_vae(self, vae_file=None):
        global first_load, vae_dict, vae_list, loaded_vae_file
        # save_settings = False

        if os.path.isfile(vae_file):
            assert os.path.isfile(vae_file), f"VAE file doesn't exist: {vae_file}"
            print(f"Loading VAE weights from: {vae_file}")
            vae_ckpt = torch.load(vae_file, map_location='cpu')
            vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if
                          k[0:4] != "loss" and k not in vae_ignore_keys}
            load_vae_dict(gs.models["sd"], vae_dict_1)

            # If vae used is not in dict, update it
            # It will be removed on refresh though
            # vae_opt = get_filename(vae_file)
            # if vae_opt not in vae_dict:
            #    vae_dict[vae_opt] = vae_file
            #    vae_list.append(vae_opt)
        else:
            print(f"VAE file doesn't exist: {vae_file}")

        loaded_vae_file = vae_file

        """
        # Save current VAE to VAE settings, maybe? will it work?
        if save_settings:
            if vae_file is None:
                vae_opt = "None"
            # shared.opts.sd_vae = vae_opt
        """

        first_load = False

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, lock=False, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.lock = lock
        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class OurSignals(QObject):
    updateimagesignal = Signal()
    webcamupdate = Signal()


def make_inject_timing_fn(inject_timing, model, steps):
    """
    inject_timing (int or list of ints or list of floats between 0.0 and 1.0):
        int: compute every inject_timing steps
        list of floats: compute on these decimal fraction steps (eg, [0.5, 1.0] for 50 steps would be at steps 25 and 50)
        list of ints: compute on these steps
    model (CompVisDenoiser)
    steps (int): number of steps
    """
    all_sigmas = model.get_sigmas(steps)
    target_sigmas = torch.empty([0], device=all_sigmas.device)

    def timing_fn(sigma):
        is_conditioning_step = False
        if sigma in target_sigmas:
            is_conditioning_step = True
        return is_conditioning_step

    if inject_timing is None:
        timing_fn = lambda sigma: True
    elif isinstance(inject_timing,int) and inject_timing <= steps and inject_timing > 0:
        # Compute every nth step
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if (i+1) % inject_timing == 0]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,float) for t in inject_timing) and all(t>=0.0 and t<=1.0 for t in inject_timing):
        # Compute on these steps (expressed as a decimal fraction between 0.0 and 1.0)
        target_indices = [int(frac_step*steps) if frac_step < 1.0 else steps-1 for frac_step in inject_timing]
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i in target_indices]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,int) for t in inject_timing) and all(t>0 and t<=steps for t in inject_timing):
        # Compute on these steps
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i+1 in inject_timing]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)

    else:
        raise Exception(f"Not a valid input: inject_timing={inject_timing}\n" +
                        f"Must be an int, list of all ints (between step 1 and {steps}), or list of all floats between 0.0 and 1.0")
    return timing_fn
def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or opts.eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if opts.eta_noise_seed_delta > 0:
                torch.manual_seed(seed + opts.eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to("cuda") for n in sampler_noises]

    x = torch.stack(xs).to("cuda")
    return x


def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.

    torch.manual_seed(seed)
    return torch.randn(shape, device="cuda")


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.

    return torch.randn(shape, device="cuda")
def load_model_from_config(config, ckpt, verbose=False):
    config = OmegaConf.load(config)
    #print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        #print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        #print(u)
    #model = torch.compile(model)
    model.half().cuda()
    model.eval()
    del sd
    del pl_sd
    from backend.hypernetworks.modules.sd_hijack import apply_optimizations
    apply_optimizations()

    return model
