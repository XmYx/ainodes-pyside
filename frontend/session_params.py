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
        print(self.params)

        return self.params

    def update_params(self):
        mode = ""
        steps = self.parent.unicontrol.w.steps.value()
        H = self.parent.unicontrol.w.H.value()
        W = self.parent.unicontrol.w.W.value()
        seed = random.randint(0, 2 ** 32 - 1) if self.parent.unicontrol.w.seed.text() == '' else int(
            self.parent.unicontrol.w.seed.text())
        prompt = self.parent.unicontrol.w.prompts.toPlainText()
        strength = self.parent.unicontrol.w.strength.value()
        mask_blur = int(self.parent.unicontrol.w.mask_blur.value())
        recons_blur = int(self.parent.unicontrol.w.reconstruction_blur.value())
        scale = self.parent.unicontrol.w.scale.value()
        ddim_eta = self.parent.unicontrol.w.ddim_eta.value()
        with_inpaint = self.parent.unicontrol.w.use_inpaint.isChecked()
        gs.T = 0
        gs.lr = 0


        # todo find out why this is no used self.parent.unicontrol.w.aesthetic_embedding.currentText()
        # gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients, self.parent.unicontrol.w.aesthetic_embedding.currentText())
        if gs.aesthetic_embedding_path != "None":
            gs.T = self.parent.unicontrol.w.gradient_steps.value()
            gs.lr = self.parent.unicontrol.w.gradient_scale.value()
        else:
            gs.aesthetic_embedding_path = None

        if gs.aesthetic_embedding_path != "None": # todo whats the difference ?
            gs.T = self.parent.unicontrol.w.gradient_steps.value()
            gs.lr = self.parent.unicontrol.w.gradient_scale.value()
            # print(f"Aesthetic Gradients: {gs.aesthetic_embedding_path} \nSteps: {gs.T} \nScale: {gs.lr}\n\nGL HF\n\n")
            # print(f"Expected Tensor Value: {(gs.T * gs.lr) + -0.3}")
        else:
            gs.aesthetic_embedding_path = None


        if self.parent.unicontrol.w.n_samples.value() == 1:
            makegrid = False
        else:
            makegrid = self.parent.unicontrol.w.makegrid.isChecked()  # self.parent.unicontrol.w.make_grid.isChecked()
        outdir = gs.system.txt2imgOut
        sampler_name = translate_sampler(self.parent.unicontrol.w.sampler.currentText())
        use_init = self.parent.unicontrol.w.use_init.isChecked()
        strength = self.parent.unicontrol.w.strength.value()
        seed_behavior = self.parent.unicontrol.w.seed_behavior.currentText()
        n_batch = self.parent.unicontrol.w.n_batch.value()
        n_samples = self.parent.unicontrol.w.n_samples.value()
        show_sample_per_step = self.parent.unicontrol.w.show_sample_per_step.isChecked()
        save_settings = self.parent.unicontrol.w.save_settings.isChecked()
        strength_0_no_init = self.parent.unicontrol.w.strength_0_no_init.isChecked()

        # self.parent.deforum.sampler_name = sampler_name
        print(f'sampler: {sampler_name} steps {steps}\nscale: {scale}\nddim_eta: {ddim_eta}')

        decode_method = None if self.parent.unicontrol.w.decode_method.currentText() == 'None' else self.parent.unicontrol.w.decode_method.currentText()

        if self.parent.unicontrol.w.toggle_negative_prompt.isChecked():
            negative_prompts = self.parent.unicontrol.w.negative_prompts.toPlainText()
            print(f"Using negative prompts {negative_prompts}")
        else:
            negative_prompts = None

        mean_scale = self.parent.unicontrol.w.mean_scale.value()
        var_scale = self.parent.unicontrol.w.var_scale.value()
        exposure_scale = self.parent.unicontrol.w.exposure_scale.value()
        exposure_target = self.parent.unicontrol.w.exposure_target.value()
        colormatch_scale = self.parent.unicontrol.w.colormatch_scale.value()
        # To Do: Image selector, and line editor PopUp window
        colormatch_image = None  # if self.parent.unicontrol.w.colormatch_image.text() == '' else self.parent.unicontrol.w.colormatch_image.text()
        colormatch_n_colors = self.parent.unicontrol.w.colormatch_n_colors.value()
        ignore_sat_weight = self.parent.unicontrol.w.ignore_sat_weight.value()
        clip_name = self.parent.unicontrol.w.clip_name.currentText()  # @param ['ViT-L/14' 'ViT-L/14@336px' 'ViT-B/16' 'ViT-B/32']
        clip_scale = self.parent.unicontrol.w.clip_scale.value()
        aesthetics_scale = self.parent.unicontrol.w.aesthetics_scale.value()
        cutn = int(self.parent.unicontrol.w.cutn.value())
        cut_pow = self.parent.unicontrol.w.cut_pow.value()

        init_mse_scale = 0
        init_mse_image = None #if self.parent.unicontrol.w.init_mse_image.text() == '' else self.parent.unicontrol.w.init_mse_image.text()
        blue_scale = 0

        gradient_wrt = self.parent.unicontrol.w.gradient_wrt.currentText()  # ["x" "x0_pred"]
        gradient_add_to = self.parent.unicontrol.w.gradient_add_to.currentText()  # ["cond" "uncond" "both"]
        decode_method = decode_method  # ["autoencoder""linear"]
        grad_threshold_type = self.parent.unicontrol.w.grad_threshold_type.currentText()  # ["dynamic" "static" "mean" "schedule"]
        clamp_grad_threshold = self.parent.unicontrol.w.clamp_grad_threshold.value()
        clamp_start = self.parent.unicontrol.w.clamp_start.value()
        clamp_stop = self.parent.unicontrol.w.clamp_stop.value()
        grad_inject_timing = 0  if self.parent.unicontrol.w.grad_inject_timing.text() == '' else self.parent.unicontrol.w.grad_inject_timing.text() #it is a float an int or a list of floats
        cond_uncond_sync = self.parent.unicontrol.w.cond_uncond_sync.isChecked()
        negative_prompts = negative_prompts
        prompts = self.parent.unicontrol.w.prompts.toPlainText()
        hires = self.parent.unicontrol.w.hires.isChecked()

        mask_overlay_blur = self.parent.unicontrol.w.mask_overlay_blur.value()
        precision = 'autocast'
        timestring = ""
        border = self.parent.unicontrol.w.border.currentText()
        angle = self.parent.unicontrol.w.angle.toPlainText()
        zoom = self.parent.unicontrol.w.zoom.toPlainText()
        translation_x = self.parent.unicontrol.w.translation_x.toPlainText()
        translation_y = self.parent.unicontrol.w.translation_y.toPlainText()
        translation_z = self.parent.unicontrol.w.translation_z.toPlainText()
        rotation_3d_x = self.parent.unicontrol.w.rotation_3d_x.toPlainText()
        rotation_3d_y = self.parent.unicontrol.w.rotation_3d_y.toPlainText()
        rotation_3d_z = self.parent.unicontrol.w.rotation_3d_z.toPlainText()
        flip_2d_perspective = self.parent.unicontrol.w.flip_2d_perspective.isChecked()
        perspective_flip_theta = self.parent.unicontrol.w.perspective_flip_theta.toPlainText()
        perspective_flip_phi = self.parent.unicontrol.w.perspective_flip_phi.toPlainText()
        perspective_flip_gamma = self.parent.unicontrol.w.perspective_flip_gamma.toPlainText()
        perspective_flip_fv = self.parent.unicontrol.w.perspective_flip_fv.toPlainText()
        noise_schedule = self.parent.unicontrol.w.noise_schedule.toPlainText()
        strength_schedule = self.parent.unicontrol.w.strength_schedule.toPlainText()
        contrast_schedule = self.parent.unicontrol.w.contrast_schedule.toPlainText()
        diffusion_cadence = self.parent.unicontrol.w.diffusion_cadence.value()
        color_coherence = self.parent.unicontrol.w.color_coherence.currentText()
        use_depth_warping = self.parent.unicontrol.w.use_depth_warping.isChecked()
        midas_weight = self.parent.unicontrol.w.midas_weight.value()
        near_plane = self.parent.unicontrol.w.near_plane.value()
        far_plane = self.parent.unicontrol.w.far_plane.value()
        fov = self.parent.unicontrol.w.fov.value()
        padding_mode = self.parent.unicontrol.w.padding_mode.currentText()
        sampling_mode = self.parent.unicontrol.w.sampling_mode.currentText()
        save_depth_maps = self.parent.unicontrol.w.save_depth_maps.isChecked()
        use_mask_video = self.parent.unicontrol.w.use_mask_video.isChecked()
        resume_from_timestring = self.parent.unicontrol.w.resume_from_timestring.isChecked()
        resume_timestring = self.parent.unicontrol.w.resume_timestring.text()
        clear_latent = self.parent.unicontrol.w.clear_latent.isChecked()
        clear_sample = self.parent.unicontrol.w.clear_sample.isChecked()
        shouldStop = False
        cpudepth = self.parent.unicontrol.w.cpudepth.isChecked()
        skip_video_for_run_all = False
        advanced = False
        init_image = self.parent.unicontrol.w.init_image.text()
        prompt_weighting = self.parent.unicontrol.w.prompt_weighting.isChecked()
        normalize_prompt_weights = self.parent.unicontrol.w.normalized_prompts.isChecked()
        animation_mode = 'None'
        use_inpaint = self.parent.unicontrol.w.use_inpaint.isChecked()
        lowmem = self.parent.unicontrol.w.lowmem.isChecked()

        plotting = self.parent.unicontrol.w.plotting.isChecked()
        plotX = self.parent.unicontrol.w.plotX.currentText()
        plotY = self.parent.unicontrol.w.plotY.currentText()
        plotXLine = self.parent.unicontrol.w.plotXLine.text()
        plotYLine = self.parent.unicontrol.w.plotYLine.text()
        self.params = {
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

            # Advanced Params
            'animation_mode':'None',
            'ddim_eta': ddim_eta,
            'save_settings': save_settings,
            'save_samples': True,
            'show_sample_per_step': show_sample_per_step,
            'n_batch': n_batch,
            'seed_behavior': seed_behavior,
            'makegrid': makegrid,
            'grid_rows': 2,
            'use_init': use_init,
            'init_image': None if init_image == '' else init_image,
            'strength': strength,
            'strength_0_no_init': strength_0_no_init,
            'device': 'cuda',
            'max_frames': 1,
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
            "use_inpaint": use_inpaint,
            "lowmem": lowmem,
            "plotting": plotting,
            "plotX": plotX,
            "plotY": plotY,
            "plotXLine": plotXLine,
            "plotYLine": plotYLine
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
    else:
        sampler = sampler
    return sampler
