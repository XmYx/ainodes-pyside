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

from backend.singleton import singleton
gs = singleton
import os, sys
import random
import traceback
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Signal, QObject, QThreadPool, Slot, QRunnable
from PySide6.QtWidgets import QTextEdit, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QLabel
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

from torch import autocast
import torch
from backend.resizeRight import resizeright, interp_methods
torch.set_grad_enabled(False)

import cv2
import numpy as np

class aiNodesPlugin():
    def __init__(self, parent):
        self.parent = parent

    def initme(self):
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
        self.continous = QtWidgets.QPushButton('Start')
        self.continous.clicked.connect(self.start_continuous_capture)
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
        # Set up the layout
        layout = QtWidgets.QVBoxLayout()
        #layout.addWidget(self.maskprompt)
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.strength)
        layout.addWidget(self.rescalefactorlabel)
        layout.addWidget(self.rescalefactor)
        layout.addWidget(self.eta)
        layout.addWidget(self.seed)
        layout.addWidget(self.samplercombobox)
        layout.addWidget(self.modelselect)
        layout.addWidget(self.webcam_dropdown)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.continous)
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
        #if self.capture.isOpened():
        #    self.wtimer = QtCore.QTimer()
        #    self.wtimer.timeout.connect(self.update_frame)
        #    self.wtimer.start(8)

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

    def start_continuous_capture(self):

        inference = self.modelselect.currentText()
        if self.loadedmodel != "normal":
            self.model = load_model_from_config("data/models/v1-5-pruned-emaonly.yaml",
                                                "data/models/v1-5-pruned-emaonly.ckpt")
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
            self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)

    def continuous_capture(self, progress_callback=None):
        """Capture images from the webcam continuously."""
        # State for interpolating between diffusion steps
        self.images = []
        self.index = 0
        self.lastinit = None
        self.seedint = 0
        #if self.loadedmodel == "inpaint":
        #    self.model = None
        if self.loadedmodel == "normal":
            with autocast("cuda"):

                self.return_seedint()
                seed_everything(self.seedint)
                self.uc = self.model.get_learned_conditioning(1 * [""])
                self.promptstring = self.prompt.toPlainText()
                self.c = self.model.get_learned_conditioning(self.promptstring)
            #self.init_mask_model()
            self.model_wrap = CompVisDenoiser(self.model, quantize=False)

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
        #self.frame_to_mask_png(frame)
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
                    self.c = self.model.get_learned_conditioning(self.promptstring)
                self.images = [self.img2img(frame, prompt,
                                            steps, 1, 7.5, self.seedint, eta, strength)]
                #self.images = [result_image]
                self.update_image_signal()
            if self.run == False:
                break
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
        #seed_everything(seed)
        #torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        #with torch.autocast("cuda"):
        torch.backends.cudnn.benchmark = True
        self.init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(image))
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
        samples = sampler_fn(
            c=self.c,
            uc=self.uc,
            args=self.args,
            model_wrap=self.cfg_model,
            init_latent=self.init_latent,
            t_enc=t_enc,
            device="cuda",
            cb=None,
            verbose=False)
        x_samples = self.model.decode_first_stage(samples)
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
def sampler_fn(
    c: torch.Tensor,
    uc: torch.Tensor,
    args,
    model_wrap: CompVisDenoiser,
    init_latent: Optional[torch.Tensor] = None,
    t_enc: Optional[torch.Tensor] = None,
    device=torch.device("cpu")
    if not torch.cuda.is_available()
    else torch.device("cuda"),
    cb: Callable[[Any], None] = None,
    verbose: Optional[bool] = False,
) -> torch.Tensor:
    shape = [args.C, args.H // args.f, args.W // args.f]
    karras = True
    if karras == True:
        sigmas = sampling.get_sigmas_karras(n=args.steps, sigma_min=0.1, sigma_max=10, device="cuda")
    else:
        sigmas: torch.Tensor = model_wrap.get_sigmas(args.steps)
    #if gs.diffusion.discard_next_to_last_sigma == True:
    #    sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    sigmas = sigmas[len(sigmas) - t_enc - 1:]

    x = (
            init_latent
            + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
    )
    sampler_args = {
        "model": model_wrap,
        "x": x,
        "sigmas": sigmas,
        "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
        "disable": False,
        "callback": cb,
    }
    min = sigmas[0].item()
    max = min
    for i in sigmas:
        if i.item() < min and i.item() != 0.0:
            min = i.item()
    if args.sampler in ["dpm_fast"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
            "n":args.steps,
            "eta": 0.0,
            "s_noise": 1.0,
        }
    elif args.sampler in ["dpm_adaptive"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
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
            "sigmas": sigmas,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
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
