import random
from types import SimpleNamespace

from backend.singleton import singleton

gs = singleton


class SessionParams():
    def __init__(self, parent):
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


    def create_params(self):
        self.params = {}
        for key, value in gs.diffusion.__dict__.items():
            self.params[key] = value

        return self.params

    def update_params(self):
        mode = ""
        widget = 'unicontrol'
        steps = self.parent.widgets[widget].w.steps.value()
        H = self.parent.widgets[widget].w.H.value()
        W = self.parent.widgets[widget].w.W.value()
        seed = random.randint(0, 2 ** 32 - 1) if self.parent.widgets[widget].w.seed.text() == '' else int(
            self.parent.widgets[widget].w.seed.text())
        prompt = self.parent.widgets[widget].w.prompts.toPlainText()
        mask_blur = int(self.parent.widgets[widget].w.mask_blur.value())
        recons_blur = int(self.parent.widgets[widget].w.recons_blur.value())
        scale = self.parent.widgets[widget].w.scale.value()
        ddim_eta = self.parent.widgets[widget].w.ddim_eta.value()
        with_inpaint = self.parent.widgets[widget].w.with_inpaint.isChecked()
        gs.T = 0
        gs.lr = 0


        # todo find out why this is no used self.parent.widgets[widget].w.aesthetic_embedding.currentText()
        # gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients_dir, self.parent.widgets[widget].w.aesthetic_embedding.currentText())
        if gs.aesthetic_embedding_path != "None":
            gs.T = self.parent.widgets[widget].w.gradient_steps.value()
            gs.lr = self.parent.widgets[widget].w.gradient_scale.value()
        else:
            gs.aesthetic_embedding_path = None

        if gs.aesthetic_embedding_path != "None": # todo whats the difference ?
            gs.T = self.parent.widgets[widget].w.gradient_steps.value()
            gs.lr = self.parent.widgets[widget].w.gradient_scale.value()
            # print(f"Aesthetic Gradients: {gs.aesthetic_embedding_path} \nSteps: {gs.T} \nScale: {gs.lr}\n\nGL HF\n\n")
            # print(f"Expected Tensor Value: {(gs.T * gs.lr) + -0.3}")
        else:
            gs.aesthetic_embedding_path = None


        if self.parent.widgets[widget].w.n_samples.value() == 1:
            make_grid = False
        else:
            make_grid = self.parent.widgets[widget].w.make_grid.isChecked()  # self.parent.widgets[widget].w.make_grid.isChecked()
        outdir = gs.system.txt2img_out_dir
        sampler_name = translate_sampler(self.parent.widgets[widget].w.sampler.currentText())
        use_init = self.parent.widgets[widget].w.use_init.isChecked()
        strength = self.parent.widgets[widget].w.strength.value()
        seed_behavior = self.parent.widgets[widget].w.seed_behavior.currentText()
        n_batch = self.parent.widgets[widget].w.n_batch.value()
        n_samples = self.parent.widgets[widget].w.n_samples.value()
        show_sample_per_step = self.parent.widgets[widget].w.show_sample_per_step.isChecked()
        save_settings = self.parent.widgets[widget].w.save_settings.isChecked()
        strength_0_no_init = self.parent.widgets[widget].w.strength_0_no_init.isChecked()

        # self.parent.deforum.sampler_name = sampler_name
        print(f'sampler: {sampler_name} steps {steps}\nscale: {scale}\nddim_eta: {ddim_eta}')

        decode_method = None if self.parent.widgets[widget].w.decode_method.currentText() == 'None' else self.parent.widgets[widget].w.decode_method.currentText()

        if self.parent.widgets[widget].w.toggle_negative_prompt.isChecked():
            negative_prompts = self.parent.widgets[widget].w.negative_prompts.toPlainText()
            print(f"Using negative prompts {negative_prompts}")
        else:
            negative_prompts = None

        mean_scale = self.parent.widgets[widget].w.mean_scale.value()
        var_scale = self.parent.widgets[widget].w.var_scale.value()
        exposure_scale = self.parent.widgets[widget].w.exposure_scale.value()
        exposure_target = self.parent.widgets[widget].w.exposure_target.value()
        colormatch_scale = self.parent.widgets[widget].w.colormatch_scale.value()
        # To Do: Image selector, and line editor PopUp window
        colormatch_image = None  # if self.parent.widgets[widget].w.colormatch_image.text() == '' else self.parent.widgets[widget].w.colormatch_image.text()
        colormatch_n_colors = self.parent.widgets[widget].w.colormatch_n_colors.value()
        ignore_sat_weight = self.parent.widgets[widget].w.ignore_sat_weight.value()
        clip_name = self.parent.widgets[widget].w.clip_name.currentText()  # @param ['ViT-L/14' 'ViT-L/14@336px' 'ViT-B/16' 'ViT-B/32']
        clip_scale = self.parent.widgets[widget].w.clip_scale.value()
        aesthetics_scale = self.parent.widgets[widget].w.aesthetics_scale.value()
        cutn = int(self.parent.widgets[widget].w.cutn.value())
        cut_pow = self.parent.widgets[widget].w.cut_pow.value()

        init_mse_scale = self.parent.widgets[widget].w.init_mse_scale.value()
        init_mse_image = None #if self.parent.widgets[widget].w.init_mse_image.text() == '' else self.parent.widgets[widget].w.init_mse_image.text()
        blue_scale = self.parent.widgets[widget].w.blue_scale.value()

        gradient_wrt = self.parent.widgets[widget].w.gradient_wrt.currentText()  # ["x" "x0_pred"]
        gradient_add_to = self.parent.widgets[widget].w.gradient_add_to.currentText()  # ["cond" "uncond" "both"]
        decode_method = decode_method  # ["autoencoder""linear"]
        grad_threshold_type = self.parent.widgets[widget].w.grad_threshold_type.currentText()  # ["dynamic" "static" "mean" "schedule"]
        clamp_grad_threshold = self.parent.widgets[widget].w.clamp_grad_threshold.value()
        clamp_start = self.parent.widgets[widget].w.clamp_start.value()
        clamp_stop = self.parent.widgets[widget].w.clamp_stop.value()
        grad_inject_timing = 1 if self.parent.widgets[widget].w.grad_inject_timing.text() == '' else int(self.parent.widgets[widget].w.grad_inject_timing.text()) #it is a float an int or a list of floats
        cond_uncond_sync = self.parent.widgets[widget].w.cond_uncond_sync.isChecked()
        negative_prompts = negative_prompts
        prompts = self.parent.widgets[widget].w.prompts.toPlainText()
        hires = self.parent.widgets[widget].w.hires.isChecked()

        mask_overlay_blur = self.parent.widgets[widget].w.mask_overlay_blur.value()
        precision = 'autocast'
        timestring = ""
        border = self.parent.widgets[widget].w.border.currentText()
        angle = self.parent.widgets[widget].w.angle.toPlainText()
        zoom = self.parent.widgets[widget].w.zoom.toPlainText()
        translation_x = self.parent.widgets[widget].w.translation_x.toPlainText()
        translation_y = self.parent.widgets[widget].w.translation_y.toPlainText()
        translation_z = self.parent.widgets[widget].w.translation_z.toPlainText()
        rotation_3d_x = self.parent.widgets[widget].w.rotation_3d_x.toPlainText()
        rotation_3d_y = self.parent.widgets[widget].w.rotation_3d_y.toPlainText()
        rotation_3d_z = self.parent.widgets[widget].w.rotation_3d_z.toPlainText()
        flip_2d_perspective = self.parent.widgets[widget].w.flip_2d_perspective.isChecked()
        perspective_flip_theta = self.parent.widgets[widget].w.perspective_flip_theta.toPlainText()
        perspective_flip_phi = self.parent.widgets[widget].w.perspective_flip_phi.toPlainText()
        perspective_flip_gamma = self.parent.widgets[widget].w.perspective_flip_gamma.toPlainText()
        perspective_flip_fv = self.parent.widgets[widget].w.perspective_flip_fv.toPlainText()
        noise_schedule = self.parent.widgets[widget].w.noise_schedule.toPlainText()
        strength_schedule = self.parent.widgets[widget].w.strength_schedule.toPlainText()
        contrast_schedule = self.parent.widgets[widget].w.contrast_schedule.toPlainText()
        diffusion_cadence = self.parent.widgets[widget].w.diffusion_cadence.value()
        color_coherence = self.parent.widgets[widget].w.color_coherence.currentText()
        use_depth_warping = self.parent.widgets[widget].w.use_depth_warping.isChecked()
        midas_weight = self.parent.widgets[widget].w.midas_weight.value()
        near_plane = self.parent.widgets[widget].w.near_plane.value()
        far_plane = self.parent.widgets[widget].w.far_plane.value()
        fov = self.parent.widgets[widget].w.fov.value()
        padding_mode = self.parent.widgets[widget].w.padding_mode.currentText()
        sampling_mode = self.parent.widgets[widget].w.sampling_mode.currentText()
        save_depth_maps = self.parent.widgets[widget].w.save_depth_maps.isChecked()
        use_mask_video = self.parent.widgets[widget].w.use_mask_video.isChecked()
        resume_from_timestring = self.parent.widgets[widget].w.resume_from_timestring.isChecked()
        resume_timestring = self.parent.widgets[widget].w.resume_timestring.text()
        clear_latent = self.parent.widgets[widget].w.clear_latent.isChecked()
        clear_sample = self.parent.widgets[widget].w.clear_sample.isChecked()
        shouldStop = False
        cpudepth = self.parent.widgets[widget].w.cpudepth.isChecked()
        skip_video_for_run_all = False

        init_image = self.parent.widgets[widget].w.init_image.text()
        prompt_weighting = self.parent.widgets[widget].w.prompt_weighting.isChecked()
        normalize_prompt_weights = self.parent.widgets[widget].w.normalized_prompts.isChecked()
        outdir = gs.system.txt2img_out_dir

        if self.parent.widgets[widget].w.max_frames.value() < 2:
            animation_mode = 'None'
        else:
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

        advanced = False
        max_frame = self.parent.widgets[widget].w.max_frames.value() if animation_mode != 'None' else 1
        lowmem = self.parent.widgets[widget].w.lowmem.isChecked()
        seamless = self.parent.widgets[widget].w.seamless.isChecked()
        if self.parent.widgets[widget].w.axis.currentText() == 'X':
            axis = {'x'}
        elif self.parent.widgets[widget].w.axis.currentText() == 'Y':
            axis = {'y'}
        elif self.parent.widgets[widget].w.axis.currentText() == 'Both':
            axis = {'x', 'y'}
        plotting = self.parent.widgets[widget].w.plotting.isChecked()
        plotX = self.parent.widgets[widget].w.plotX.currentText()
        plotY = self.parent.widgets[widget].w.plotY.currentText()
        plotXLine = self.parent.widgets[widget].w.plotXLine.text()
        plotYLine = self.parent.widgets[widget].w.plotYLine.text()
        gradient_pass = self.parent.widgets[widget].w.gradient_pass.currentText()
        return_type = self.parent.widgets[widget].w.return_type.currentText()
        keyframes = self.parent.widgets[widget].w.keyframes.toPlainText()

        self.params = {             # todo make this a one step thing not two steps
            # Basic Params
            'mode': mode,
            'sampler': sampler_name,
            'W': W,
            'H': H,
            'steps': steps,
            'scale': scale,
            'prompts': prompts,
            'seed': seed,
            'advanced': advanced,
            'seamless': seamless,
            'axis': axis,
            # Advanced Params
            'animation_mode': animation_mode,
            'ddim_eta': ddim_eta,
            'save_settings': save_settings,
            'save_samples': True,
            'show_sample_per_step': show_sample_per_step,
            'n_batch': n_batch,
            'seed_behavior': seed_behavior,
            'make_grid': make_grid,
            'grid_rows': 2,
            'use_init': use_init,
            'init_image': None if init_image == '' else init_image,
            'strength': strength,
            'strength_0_no_init': strength_0_no_init,
            'device': 'cuda',
            'max_frames': max_frame,
            'outdir': outdir,
            'n_samples': n_samples,
            'mean_scale': mean_scale,
            'var_scale': var_scale,
            'exposure_scale': exposure_scale,
            'exposure_target': exposure_target,
            'colormatch_scale': colormatch_scale,
            'colormatch_image': colormatch_image,
            'colormatch_n_colors': colormatch_n_colors,
            'ignore_sat_weight': ignore_sat_weight,
            'clip_name': clip_name,  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
            'clip_scale': clip_scale,
            'aesthetics_scale': aesthetics_scale,
            'cutn': cutn,
            'cut_pow': cut_pow,
            'init_mse_scale': init_mse_scale,
            'init_mse_image': init_mse_image,
            'blue_scale': blue_scale,
            'gradient_wrt': gradient_wrt,  # ["x", "x0_pred"]
            'gradient_add_to': gradient_add_to,  # ["cond", "uncond", "both"]
            'decode_method': decode_method,  # ["autoencoder","linear"]
            'grad_threshold_type': grad_threshold_type,  # ["dynamic", "static", "mean", "schedule"]
            'clamp_grad_threshold': clamp_grad_threshold,
            'clamp_start': clamp_start,
            'clamp_stop': clamp_stop,
            'grad_inject_timing': grad_inject_timing,
            # if self.parent.unicontrol.w.grad_inject_timing.text() :: '' else self.parent.unicontrol.w.grad_inject_timing.text(), #it is a float an int or a list of floats
            'cond_uncond_sync': cond_uncond_sync,
            'negative_prompts': negative_prompts,
            'hires': hires,

            # Outpaint Parameters
            "with_inpaint": with_inpaint,
            "mask_blur": mask_blur,
            "recons_blur": recons_blur,

            # Animation Parameters
            "mask_overlay_blur": mask_overlay_blur,
            "precision": precision,
            "timestring": timestring,
            "border": border,
            "angle": angle,
            "zoom": zoom,
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_z": translation_z,
            "rotation_3d_x": rotation_3d_x,
            "rotation_3d_y": rotation_3d_y,
            "rotation_3d_z": rotation_3d_z,
            "flip_2d_perspective": flip_2d_perspective,
            "perspective_flip_theta": perspective_flip_theta,
            "perspective_flip_phi": perspective_flip_phi,
            "perspective_flip_gamma": perspective_flip_gamma,
            "perspective_flip_fv": perspective_flip_fv,
            "noise_schedule": noise_schedule,
            "strength_schedule": strength_schedule,
            "contrast_schedule": contrast_schedule,
            "diffusion_cadence": diffusion_cadence,
            "color_coherence": color_coherence,
            "use_depth_warping": use_depth_warping,
            "midas_weight": midas_weight,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "fov": fov,
            "padding_mode": padding_mode,
            "sampling_mode": sampling_mode,
            "save_depth_maps": save_depth_maps,
            "use_mask_video": use_mask_video,
            "resume_from_timestring": resume_from_timestring,
            "resume_timestring": resume_timestring,
            "clear_latent": clear_latent,
            "clear_sample": clear_sample,
            "shouldStop": shouldStop,
            "cpudepth": cpudepth,
            "skip_video_for_run_all": skip_video_for_run_all,
            "prompt_weighting": prompt_weighting,
            "normalize_prompt_weights": normalize_prompt_weights,
            "lowmem": lowmem,
            "plotting": plotting,
            "plotX": plotX,
            "plotY": plotY,
            "plotXLine": plotXLine,
            "plotYLine": plotYLine,
            "gradient_pass": gradient_pass,
            "return_type": return_type,
            "keyframes": keyframes,
        }

        self.params = SimpleNamespace(**self.params)
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
