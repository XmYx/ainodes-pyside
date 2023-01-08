import random
from types import SimpleNamespace
from backend.sqlite import setting_db
from backend.singleton import singleton

gs = singleton


class SessionParams():
    def __init__(self, parent):
        self.system_params = None
        self.parent = parent
        self.max_history = None
        self.history = None
        self.params = None
        self.history = []
        self.history_index = 0
        self.max_history = 100
        self.session_name = "Test Session"
        self.session_id = "Test Id"
        self.session_mode = "still"

    def add_state_to_history(self):
        if len(self.history) == self.max_history:
            self.history.pop(0)
        self.history.append(self.params)
        self.history_index = len(self.history)

    def undo(self):
        if self.history_index > 0:
            self.params = self.history[self.history_index - 1]
            self.history_index -= 1
            self.update_ui_from_params()

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.params = self.history[self.history_index + 1]
            self.history_index += 1
            self.update_ui_from_params()


    def create_diffusion_params(self):
        self.params = {}
        for key, value in gs.diffusion.__dict__.items():
            self.params[key] = value
        return self.params


    def create_system_params(self):
        self.system_params = {}
        for key, value in gs.system.__dict__.items():
            self.system_params[key] = value
        return self.params


    def update_system_params(self):
        for key, value in self.system_params.items():
            try:
                current_widget = self.parent.system_setup.w
                type = str(getattr(current_widget, key))
                if 'QSpinBox' in type or 'QDoubleSpinBox' in type:
                    self.system_params[key] = getattr(current_widget, key).value()
                elif  'QTextEdit' in type or 'QLineEdit' in type:
                    self.system_params[key] = getattr(current_widget, key).text()
                elif 'QCheckBox' in type:
                    self.system_params[key] = getattr(current_widget, key).isChecked()
                elif 'QComboBox' in type:
                    self.system_params[key] = getattr(current_widget, key).currentText()
            except Exception as e:
                continue
            try:
                gs.system.__dict__[key] = value
            except:
                pass
        setting_db.save_settings()

    def update_diffusion_settings(self):
        for key in self.params.keys():
            if key in gs.diffusion.__dict__:
                gs.diffusion.__dict__[key] = self.params[key]
        gs.diffusion.seed = self.store_seed
        setting_db.save_settings()


    def update_params(self):

        widget = 'unicontrol'

        gs.T = 0
        gs.lr = 0

        # todo find out why this is no used self.parent.widgets[widget].w.aesthetic_embedding.currentText()
        # gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients_dir, self.parent.widgets[widget].w.aesthetic_embedding.currentText())
        if gs.aesthetic_embedding_path != "None":
            gs.T = self.parent.widgets[widget].w.gradient_steps.value()
            gs.lr = self.parent.widgets[widget].w.gradient_scale.value()
        else:
            gs.aesthetic_embedding_path = None

        if self.parent.widgets[widget].w.n_batch.value() == 1:
            make_grid = False
        else:
            make_grid = self.parent.widgets[widget].w.make_grid.isChecked()  # self.parent.widgets[widget].w.make_grid.isChecked()

        if self.parent.widgets[widget].w.toggle_negative_prompt.isChecked():
            negative_prompts = self.parent.widgets[widget].w.negative_prompts.toPlainText()
            print(f"Using negative prompts {negative_prompts}")
        else:
            negative_prompts = None

        if self.parent.widgets[widget].w.grad_inject_timing.text() == '':
            grad_inject_timing = 1
        elif self.parent.widgets[widget].w.grad_inject_timing.text() == 'None':
            grad_inject_timing = None
        else:
            grad_inject_timing = int(self.parent.widgets[widget].w.grad_inject_timing.text())


        negative_prompts = negative_prompts

        outdir = gs.system.txt2img_out_dir

        if self.parent.widgets[widget].w.max_frames.value() < 2:
            animation_mode = 'None'

            use_mask = self.parent.widgets[widget].w.use_mask.isChecked()
            use_alpha_as_mask = self.parent.widgets[widget].w.use_alpha_as_mask_2.isChecked()
            mask_file = self.parent.widgets[widget].w.mask_file.text()
            invert_mask = self.parent.widgets[widget].w.invert_mask_2.isChecked()
            mask_brightness_adjust = self.parent.widgets[widget].w.mask_brightness_adjust_2.value()
            mask_contrast_adjust = self.parent.widgets[widget].w.mask_contrast_adjust_2.value()
            mask_overlay_blur = self.parent.widgets[widget].w.mask_overlay_blur_2.value()
            overlay_mask = self.parent.widgets[widget].w.overlay_mask_2.isChecked()


        else:

            use_alpha_as_mask = self.parent.widgets[widget].w.use_alpha_as_mask.isChecked()
            mask_file = self.parent.widgets[widget].w.mask_file.text()
            invert_mask = self.parent.widgets[widget].w.invert_mask.isChecked()
            mask_brightness_adjust = self.parent.widgets[widget].w.mask_brightness_adjust.value()
            mask_contrast_adjust = self.parent.widgets[widget].w.mask_contrast_adjust.value()
            mask_overlay_blur = self.parent.widgets[widget].w.mask_overlay_blur.value()
            overlay_mask = self.parent.widgets[widget].w.overlay_mask.isChecked()

            if self.parent.widgets[widget].w.anim2D.isChecked():
                animation_mode = '2D'
                outdir = gs.system.txt2vid_single_frame_dir
                gs.system.pathmode = 'subfolders'
            if self.parent.widgets[widget].w.anim3D.isChecked():
                animation_mode = '3D'
                outdir = gs.system.txt2vid_single_frame_dir
                gs.system.pathmode = 'subfolders'
            if self.parent.widgets[widget].w.animVid.isChecked():
                animation_mode = 'Video Input'
                outdir = gs.system.txt2vid_single_frame_dir
                gs.system.pathmode = 'subfolders'

        if self.parent.widgets[widget].w.axis.currentText() == 'X':
            axis = {'x'}
        elif self.parent.widgets[widget].w.axis.currentText() == 'Y':
            axis = {'y'}
        elif self.parent.widgets[widget].w.axis.currentText() == 'Both':
            axis = {'x', 'y'}
        self.store_seed = self.parent.widgets[widget].w.seed.text()
        seed =  random.randint(0, 2 ** 32 - 1) if self.parent.widgets[widget].w.seed.text() == '' else int(
            self.parent.widgets[widget].w.seed.text())

        self.params = {             # todo make this a one step thing not two steps
            # Basic Params
            'mode': "",
            'sampler': translate_sampler(self.parent.widgets[widget].w.sampler.currentText()),
            'W': self.parent.widgets[widget].w.W.value(),
            'H': self.parent.widgets[widget].w.H.value(),
            'steps': self.parent.widgets[widget].w.steps.value(),
            'scale': self.parent.widgets[widget].w.scale.value(),
            'prompts': self.parent.widgets[widget].w.prompts.toPlainText(),
            'seed': seed,
            'advanced': False, # todo make variable
            'seamless': self.parent.widgets[widget].w.seamless.isChecked(),
            'axis': axis,
            # Advanced Params
            'animation_mode': animation_mode,
            'ddim_eta': self.parent.widgets[widget].w.ddim_eta.value(),
            'save_settings': self.parent.widgets[widget].w.save_settings.isChecked(),
            'save_samples': True,
            'show_sample_per_step': self.parent.widgets[widget].w.show_sample_per_step.isChecked(),
            'n_batch': self.parent.widgets[widget].w.n_batch.value(),
            'seed_behavior': self.parent.widgets[widget].w.seed_behavior.currentText(),
            'make_grid': make_grid,
            'grid_rows': 2,
            'use_init': self.parent.widgets[widget].w.use_init.isChecked(),
            'init_image': None if self.parent.widgets[widget].w.init_image.text() == '' else self.parent.widgets[widget].w.init_image.text(),
            'strength': self.parent.widgets[widget].w.strength.value(),
            'strength_0_no_init': self.parent.widgets[widget].w.strength_0_no_init.isChecked(),
            'device': 'cuda',
            'max_frames': self.parent.widgets[widget].w.max_frames.value() if animation_mode != 'None' else 1,
            'outdir': outdir,
            'n_samples': self.parent.widgets[widget].w.n_samples.value(),
            'mean_scale': self.parent.widgets[widget].w.mean_scale.value(),
            'var_scale': self.parent.widgets[widget].w.var_scale.value(),
            'exposure_scale': self.parent.widgets[widget].w.exposure_scale.value(),
            'exposure_target': self.parent.widgets[widget].w.exposure_target.value(),
            'colormatch_scale': self.parent.widgets[widget].w.colormatch_scale.value(),
            'colormatch_image': None,  # if self.parent.widgets[widget].w.colormatch_image.text() == '' else self.parent.widgets[widget].w.colormatch_image.text()
            'colormatch_n_colors': self.parent.widgets[widget].w.colormatch_n_colors.value(),
            'ignore_sat_weight': self.parent.widgets[widget].w.ignore_sat_weight.value(),
            'clip_name': self.parent.widgets[widget].w.clip_name.currentText(),  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
            'clip_scale': self.parent.widgets[widget].w.clip_scale.value(),
            'aesthetics_scale': self.parent.widgets[widget].w.aesthetics_scale.value(),
            'cutn': int(self.parent.widgets[widget].w.cutn.value()),
            'cut_pow': self.parent.widgets[widget].w.cut_pow.value(),
            'init_mse_scale': self.parent.widgets[widget].w.init_mse_scale.value(),
            'init_mse_image': None,  #if self.parent.widgets[widget].w.init_mse_image.text() == '' else self.parent.widgets[widget].w.init_mse_image.text()
            'blue_scale': self.parent.widgets[widget].w.blue_scale.value(),
            'gradient_wrt': self.parent.widgets[widget].w.gradient_wrt.currentText(),  # ["x", "x0_pred"]
            'gradient_add_to': self.parent.widgets[widget].w.gradient_add_to.currentText(),  # ["cond", "uncond", "both"]
            'decode_method': None if self.parent.widgets[widget].w.decode_method.currentText() == 'None' else self.parent.widgets[widget].w.decode_method.currentText(),  # ["autoencoder","linear"]
            'grad_threshold_type': self.parent.widgets[widget].w.grad_threshold_type.currentText(),  # ["dynamic", "static", "mean", "schedule"]
            'clamp_grad_threshold': self.parent.widgets[widget].w.clamp_grad_threshold.value(),
            'clamp_start': self.parent.widgets[widget].w.clamp_start.value(),
            'clamp_stop': self.parent.widgets[widget].w.clamp_stop.value(),
            'grad_inject_timing': grad_inject_timing,
            # if self.parent.unicontrol.w.grad_inject_timing.text() :: '' else self.parent.unicontrol.w.grad_inject_timing.text(), #it is a float an int or a list of floats
            'cond_uncond_sync': self.parent.widgets[widget].w.cond_uncond_sync.isChecked(),
            'negative_prompts': negative_prompts,
            'hires': self.parent.widgets[widget].w.hires.isChecked(),

            # Outpaint Parameters
            "with_inpaint": self.parent.widgets[widget].w.with_inpaint.isChecked(),
            "mask_blur": int(self.parent.widgets[widget].w.mask_blur.value()),
            "recons_blur": int(self.parent.widgets[widget].w.recons_blur.value()),

            # Animation Parameters

            "precision": 'autocast', # todo make variable
            "timestring": "",  # todo make variable
            "border": self.parent.widgets[widget].w.border.currentText(),
            "angle": self.parent.widgets[widget].w.angle.toPlainText(),
            "zoom": self.parent.widgets[widget].w.zoom.toPlainText(),
            "translation_x": self.parent.widgets[widget].w.translation_x.toPlainText(),
            "translation_y": self.parent.widgets[widget].w.translation_y.toPlainText(),
            "translation_z": self.parent.widgets[widget].w.translation_z.toPlainText(),
            "rotation_3d_x": self.parent.widgets[widget].w.rotation_3d_x.toPlainText(),
            "rotation_3d_y": self.parent.widgets[widget].w.rotation_3d_y.toPlainText(),
            "rotation_3d_z": self.parent.widgets[widget].w.rotation_3d_z.toPlainText(),
            "flip_2d_perspective": self.parent.widgets[widget].w.flip_2d_perspective.isChecked(),
            "perspective_flip_theta": self.parent.widgets[widget].w.perspective_flip_theta.toPlainText(),
            "perspective_flip_phi": self.parent.widgets[widget].w.perspective_flip_phi.toPlainText(),
            "perspective_flip_gamma": self.parent.widgets[widget].w.perspective_flip_gamma.toPlainText(),
            "perspective_flip_fv": self.parent.widgets[widget].w.perspective_flip_fv.toPlainText(),
            "noise_schedule": self.parent.widgets[widget].w.noise_schedule.toPlainText(),
            "strength_schedule": self.parent.widgets[widget].w.strength_schedule.toPlainText(),
            "contrast_schedule": self.parent.widgets[widget].w.contrast_schedule.toPlainText(),
            "diffusion_cadence": self.parent.widgets[widget].w.diffusion_cadence.value(),
            "color_coherence": self.parent.widgets[widget].w.color_coherence.currentText(),
            "use_depth_warping": self.parent.widgets[widget].w.use_depth_warping.isChecked(),
            "midas_weight": self.parent.widgets[widget].w.midas_weight.value(),
            "near_plane": self.parent.widgets[widget].w.near_plane.value(),
            "far_plane": self.parent.widgets[widget].w.far_plane.value(),
            "fov": self.parent.widgets[widget].w.fov.value(),
            "padding_mode": self.parent.widgets[widget].w.padding_mode.currentText(),
            "sampling_mode": self.parent.widgets[widget].w.sampling_mode.currentText(),
            "save_depth_maps": self.parent.widgets[widget].w.save_depth_maps.isChecked(),
            "use_mask_video": self.parent.widgets[widget].w.use_mask_video.isChecked(),
            "resume_from_timestring": self.parent.widgets[widget].w.resume_from_timestring.isChecked(),
            "resume_timestring": self.parent.widgets[widget].w.resume_timestring.text(),
            "clear_latent": self.parent.widgets[widget].w.clear_latent.isChecked(),
            "clear_sample": self.parent.widgets[widget].w.clear_sample.isChecked(),
            "shouldStop": False,
            "cpudepth": self.parent.widgets[widget].w.cpudepth.isChecked(),
            "skip_video_for_run_all": False, # todo make variable
            "prompt_weighting": self.parent.widgets[widget].w.prompt_weighting.isChecked(),
            "normalize_prompt_weights": self.parent.widgets[widget].w.normalized_prompts.isChecked(),
            "lowmem": self.parent.widgets[widget].w.lowmem.isChecked(),
            "plotting": self.parent.widgets[widget].w.toggle_plotting.isChecked(),
            "plotX": self.parent.widgets[widget].w.plotX.currentText(),
            "plotY": self.parent.widgets[widget].w.plotY.currentText(),
            "plotXLine": self.parent.widgets[widget].w.plotXLine.text(),
            "plotYLine": self.parent.widgets[widget].w.plotYLine.text(),
            "gradient_pass": self.parent.widgets[widget].w.gradient_pass.currentText(),
            "return_type": self.parent.widgets[widget].w.return_type.currentText(),
            "keyframes": self.parent.widgets[widget].w.keyframes.toPlainText(),
            "multi_dim_prompt": self.parent.widgets[widget].w.multi_dim_prompt.isChecked(),
            "multi_dim_seed_mode": self.parent.widgets[widget].w.multi_dim_seed_behavior.currentText(),
            "use_mask": use_mask,
            "use_alpha_as_mask": use_alpha_as_mask,
            "mask_file": mask_file,
            "invert_mask": invert_mask,
            "mask_brightness_adjust": mask_brightness_adjust,

            "mask_contrast_adjust": mask_contrast_adjust,
            "mask_overlay_blur": mask_overlay_blur,
            "overlay_mask": overlay_mask,
            # todo make this a variable controlled by ui
            "dynamic_threshold": None,
            "static_threshold": None,
            "display_samples": False,
            "save_sample_per_step": False,
            "log_weighted_subprompts": False,
            "adabins": self.parent.widgets[widget].w.adabins.isChecked(),
            "batch_name": 'batch_name_' + str(seed),
            "filename_format": "{timestring}_{index}_{prompt}.png",

            "init_latent": None,
            "init_sample": None,
            "init_c": None,
            "video_init_path": self.parent.widgets[widget].w.video_init_path.text(),
            "extract_nth_frame": 1,
            "overwrite_extracted_frames": True,
            "video_mask_path": self.parent.widgets[widget].w.video_mask_path.text(),
            "interpolate_key_frames": False,
            "interpolate_x_frames": 4,
            "prompt": self.parent.widgets[widget].w.prompts.toPlainText(),
            "apply_strength": 0,
            "apply_circular": False,
            "karras_sigma_min": self.parent.widgets[widget].w.karras_sigma_min.value(),
            "karras_sigma_max": self.parent.widgets[widget].w.karras_sigma_max.value(),
            "pathmode": self.parent.widgets[widget].w.pathmode.currentText(),
            "discard_next_to_last_sigma": self.parent.widgets[widget].w.discard_next_to_last_sigma.isChecked(),
        }
        self.update_diffusion_settings()
        self.params = SimpleNamespace(**self.params)
        print(f'sampler: {self.params.sampler} steps {self.params.steps}\nscale: {self.params.scale}\nddim_eta: {self.params.ddim_eta}')
        return self.params


def translate_sampler(sampler):
    if sampler == "LMS":
        sampler = "klms"
    elif sampler == "DPM 2":
        sampler = "dpm2"
    elif sampler == "DPM 2 Ancestral":
        sampler = "dpm2_ancestral"
    elif sampler == "Heun":
        sampler = "heun"
    elif sampler == "Euler":
        sampler = "euler"
    elif sampler == "Euler Ancestral":
        sampler = "euler_ancestral"
    elif sampler == "DPM Fast":
        sampler = "dpm_fast"
    elif sampler == "DPM Adaptive":
        sampler = "dpm_adaptive"
    elif sampler == "DPMPP 2S Ancestral":
        sampler = "dpmpp_2s_a"
    elif sampler == "DPMPP 2M":
        sampler = "dpmpp_2m"
    elif sampler == "DPMPP SDE":
        sampler = "dpmpp_sde"
    else:
        sampler = sampler
    return sampler
