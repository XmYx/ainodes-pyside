import copy
import os
import random
import shutil

from PySide6 import QtUiTools
from PySide6.QtCore import QFile, QObject

from plugins.autotest.image_compare import img_compare
from backend.singleton import singleton
gs = singleton

class AutoTest(QObject):
    def __init__(self, *args, **kwargs):
        self.params = None
        loader = QtUiTools.QUiLoader()
        file = QFile("plugins/autotest/autotest.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class aiPixelsPlugin():
    def __init__(self, parent):
        self.parent = parent
        self.default_params = None
        self.autotest = AutoTest()
        self.get_settings()
        self.path = 'plugins/autotest'
        self.create_output_folder()

    def initme(self):
        print("Initializing Auto Test")
        self.connections()
        self.autotest.w.show()

    def run_it(self):
        self.parent.deforum_ui.deforum_six.run_deforum_six(W=int(self.params.W),
                                         H=int(self.params.H),
                                         seed=int(self.params.seed) if self.params.seed != '' else self.params.seed,
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

                                         keyframes=self.params.keyframes,

                                         dynamic_threshold=self.params.dynamic_threshold,
                                         static_threshold=self.params.static_threshold,
                                         # @markdown **Save & Display Settings**
                                         display_samples=self.params.display_samples,  # @param {type:"boolean"}
                                         save_sample_per_step=self.params.save_sample_per_step,  # @param {type:"boolean"}
                                         # normalize_prompt_weights=True,  # @param {type:"boolean"}
                                         log_weighted_subprompts=self.params.log_weighted_subprompts,  # @param {type:"boolean"}
                                         adabins=self.params.adabins,

                                         batch_name=self.params.batch_name,  # @param {type:"string"}
                                         filename_format=self.params.filename_format,
                                         # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]

                                         # Whiter areas of the mask are areas that change more
                                         use_mask=self.params.use_mask,  # @param {type:"boolean"}
                                         use_alpha_as_mask=self.params.use_alpha_as_mask,  # use the alpha channel of the init image as the mask
                                         mask_file=self.params.mask_file,  # @param {type:"string"}
                                         invert_mask=self.params.invert_mask,  # @param {type:"boolean"}
                                         # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                                         mask_brightness_adjust=self.params.mask_brightness_adjust,  # @param {type:"number"}
                                         mask_contrast_adjust=self.params.mask_contrast_adjust,  # @param {type:"number"}
                                         # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
                                         overlay_mask=self.params.overlay_mask,  # {type:"boolean"}
                                         # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
                                         mask_overlay_blur=self.params.mask_overlay_blur,  # {type:"number"}

                                         precision=self.params.precision,

                                         # prompt="",
                                         timestring=self.params.timestring,
                                         init_latent=self.params.init_latent,
                                         init_sample=self.params.init_sample,
                                         init_c=self.params.init_c,

                                         # Anim Args

                                         # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}

                                         border=self.params.border,  # @param ['wrap', 'replicate'] {type:'string'}
                                         angle=self.params.angle,  # @param {type:"string"}
                                         zoom=self.params.zoom,  # @param {type:"string"}
                                         translation_x=self.params.translation_x,  # @param {type:"string"}
                                         translation_y=self.params.translation_y,  # @param {type:"string"}
                                         translation_z=self.params.translation_z,  # @param {type:"string"}
                                         rotation_3d_x=self.params.rotation_3d_x,  # @param {type:"string"}
                                         rotation_3d_y=self.params.rotation_3d_y,  # @param {type:"string"}
                                         rotation_3d_z=self.params.rotation_3d_z,  # @param {type:"string"}
                                         flip_2d_perspective=self.params.flip_2d_perspective,  # @param {type:"boolean"}
                                         perspective_flip_theta=self.params.perspective_flip_theta,  # @param {type:"string"}
                                         perspective_flip_phi=self.params.perspective_flip_phi,  # @param {type:"string"}
                                         perspective_flip_gamma=self.params.perspective_flip_gamma,  # @param {type:"string"}
                                         perspective_flip_fv=self.params.perspective_flip_fv,  # @param {type:"string"}
                                         noise_schedule=self.params.noise_schedule,  # @param {type:"string"}
                                         strength_schedule=self.params.strength_schedule,  # @param {type:"string"}
                                         contrast_schedule=self.params.contrast_schedule,  # @param {type:"string"}
                                         # @markdown ####**Coherence:**
                                         color_coherence=self.params.color_coherence,
                                         diffusion_cadence=self.params.diffusion_cadence,  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                                         # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}

                                         # @markdown ####**3D Depth Warping:**
                                         use_depth_warping=self.params.use_depth_warping,  # @param {type:"boolean"}
                                         midas_weight=self.params.midas_weight,  # @param {type:"number"}
                                         near_plane=self.params.near_plane,
                                         far_plane=self.params.far_plane,
                                         fov=self.params.fov,  # @param {type:"number"}
                                         padding_mode=self.params.padding_mode,  # @param ['border', 'reflection', 'zeros'] {type:'string'}
                                         sampling_mode=self.params.sampling_mode,  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                                         save_depth_maps=self.params.save_depth_maps,  # @param {type:"boolean"}

                                         # @markdown ####**Video Input:**
                                         video_init_path=self.params.video_init_path,  # @param {type:"string"}
                                         extract_nth_frame=self.params.extract_nth_frame,  # @param {type:"number"}
                                         overwrite_extracted_frames=self.params.overwrite_extracted_frames,  # @param {type:"boolean"}
                                         use_mask_video=self.params.use_mask_video,  # @param {type:"boolean"}
                                         video_mask_path=self.params.video_mask_path,  # @param {type:"string"}

                                         # @markdown ####**Interpolation:**
                                         interpolate_key_frames=self.params.interpolate_key_frames,  # @param {type:"boolean"}
                                         interpolate_x_frames=self.params.interpolate_x_frames,  # @param {type:"number"}

                                         # @markdown ####**Resume Animation:**
                                         resume_from_timestring=self.params.resume_from_timestring,  # @param {type:"boolean"}
                                         resume_timestring=self.params.resume_timestring,
                                         # prev_sample=None,
                                         clear_latent=self.params.clear_latent,
                                         clear_sample=self.params.clear_sample,
                                         shouldStop=self.params.shouldStop,
                                         # keys={}
                                         cpudepth=self.params.cpudepth,
                                         # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                                         skip_video_for_run_all=self.params.skip_video_for_run_all,
                                         prompt=self.params.prompt,
                                         #use_hypernetwork=None,
                                         apply_strength=self.params.apply_strength,
                                         apply_circular=self.params.apply_circular)

    def get_settings(self):
        self.parent.sessionparams.create_diffusion_params()
        self.parent.sessionparams.create_system_params()
        self.default_params = self.parent.sessionparams.update_params()
        self.parent.deforum_ui.set_aesthetics()

    def connections(self):
        self.autotest.w.run_all_tests.clicked.connect(self.run_all_tests)
        self.autotest.w.run_this_test.clicked.connect(self.run_this_test)

    def get_new_seed(self):
        return random.randint(0, 2 ** 32 - 1)

    def clean_dir(self, dir):
        if os.path.isdir(dir):
            shutil.rmtree(dir)

    def create_output_folder(self):
        dir_name = os.path.join(self.path, 'testresults')
        if os.path.isdir(dir_name):
            pass
            #clean_dir(dir_name)
        self.image_out_dir = os.path.join(dir_name, 'image_out')
        os.makedirs(self.image_out_dir, exist_ok=True)

    def get_files(self, path):
        from os import walk
        return[os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']

    def reset_params(self):
        self.params = copy.deepcopy(self.default_params)

    def test_1(self):
        self.reset_params()
        self.params.prompts = 'Portrait of Emma Stone, detailed, concept art, Trending on Artstation'
        self.params.seed = 3493545648 #self.get_new_seed()
        self.params.outdir = os.path.join(self.image_out_dir, 'test_1')
        self.clean_dir(self.params.outdir)
        os.makedirs(self.params.outdir, exist_ok=True)
        self.params.save_settings = False
        print(self.params.seed)
        self.run_it()
        files = self.get_files(self.params.outdir)
        if len(files) > 0:
            img1 = files[0]
            img2 = 'plugins/autotest/expected/test_1/result_1.png'
            check = int(img_compare(img1, img2))
            print('check test 1', check)
            if check < 600000: #695998
                print(f'\033[91mfail test 1 check = {check} > 600000')
                print('\033[00m')
                return False
            else:
                print(f'\033[92mpass test 1 check = {check} > 600000')
                print('\033[92m pass test 1')
                print('\033[00m')
                return True

    def test_2(self):
        self.reset_params()
        self.params.seed = 1445889970 #self.get_new_seed()
        self.params.prompts = 'a corgi'
        self.params.outdir = os.path.join(self.image_out_dir, 'test_2')
        self.clean_dir(self.params.outdir)
        os.makedirs(self.params.outdir, exist_ok=True)
        self.params.save_settings = False
        self.run_it()
        files = self.get_files(self.params.outdir)
        if len(files) > 0:
            img1 = files[0]
            img2 = 'plugins/autotest/expected/test_2/result_1.png'
            check = img_compare(img1, img2)
            print('check test 2', check)
            if check < 400000: #695998
                print(f'\033[91mfail test 2 check = {check} < 400000')
                print('\033[00m')
                return False
            else:
                print(f'\033[92mpass test 2 check = {check} > 400000')
                print('\033[92m pass test 2')
                print('\033[00m')
                return True

    def test_3(self):
        self.reset_params()
        self.params.seed = 3493545648 #self.get_new_seed()
        self.params.use_init = True
        self.params.init_image = 'plugins/autotest/expected/test_1/result_1.png'
        self.params.strength = 0.1
        self.params.prompts = 'oil on canvas'
        self.params.outdir = os.path.join(self.image_out_dir, 'test_3')
        self.clean_dir(self.params.outdir)
        os.makedirs(self.params.outdir, exist_ok=True)
        self.params.save_settings = False
        self.run_it()
        files = self.get_files(self.params.outdir)
        if len(files) > 0:
            img1 = files[0]
            img2 = 'plugins/autotest/expected/test_3/result_1.png'
            check = img_compare(img1, img2)
            print('check test 3', check)
            if check < 650000:
                print('\033[91mfail test 3')
                print('\033[00m')
                return False
            else:
                print('\033[92m pass test 3')
                print('\033[00m')
                return True

    def test_4(self):
        self.reset_params()
        self.params.seed = 3493545648 #self.get_new_seed()
        self.params.use_init = True
        self.params.init_image = 'plugins/autotest/expected/test_2/result_1.png'
        self.params.strength = 0.9
        self.params.prompts = 'oil on canvas'
        self.params.outdir = os.path.join(self.image_out_dir, 'test_4')
        self.clean_dir(self.params.outdir)
        os.makedirs(self.params.outdir, exist_ok=True)
        self.params.save_settings = False
        self.run_it()
        files = self.get_files(self.params.outdir)
        if len(files) > 0:
            img1 = files[0]
            img2 = 'plugins/autotest/expected/test_4/result_1.png'
            check = img_compare(img1, img2)
            print('check test 3', check)
            if check < 650000:
                print('\033[91mfail test 4')
                print('\033[00m')
                return False
            else:
                print('\033[92m pass test 4')
                print('\033[00m')
                return True

    def test_5(self):
        self.reset_params()
        self.params.seed = 3493545648 #self.get_new_seed()
        #self.params.init_image = 'plugins/autotest/expected/test_1/result_1.png'
        self.params.prompts = 'oil on canvas'
        self.params.outdir = os.path.join(self.image_out_dir, 'test_4')
        self.params.max_frames = 2
        self.params.animation_mode ='2D'
        self.clean_dir(self.params.outdir)
        os.makedirs(self.params.outdir, exist_ok=True)
        self.params.save_settings = False
        self.run_it()
        files = self.get_files(self.params.outdir)
        if len(files) > 0:
            img1 = files[0]
            img2 = 'plugins/autotest/expected/test_3/result_1.png'
            check = img_compare(img1, img2)
            print('check test 3', check)
            if check < 650000:
                print('\033[91mfail test 5')
                print('\033[00m')
                return False
            else:
                print('\033[92m pass test 5')
                print('\033[00m')
                return True


    def run_all_tests(self):
        check = self.test_1()
        if check:
            check = self.test_2()
        if check:
            check = self.test_3()
        if check:
            check = self.test_4()


    def run_this_test(self):
        pass
