import ldm.modules.diffusionmodules.util
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
from backend.deforum.six.hijack import hijack_codes

import time


ddim_decode = ldm.models.diffusion.ddim.DDIMSampler.decode

plms_sampling_safe = ldm.models.diffusion.plms.PLMSSampler.plms_sampling


def undo_hijack():
    ldm.models.diffusion.ddim.DDIMSampler.decode = ddim_decode
    ldm.models.diffusion.plms.PLMSSampler.plms_sampling = plms_sampling_safe

def deforum_hijack():
    ldm.models.diffusion.ddim.DDIMSampler.decode = hijack_codes.decode
    ldm.models.diffusion.plms.PLMSSampler.plms_sampling = hijack_codes.plms_sampling
