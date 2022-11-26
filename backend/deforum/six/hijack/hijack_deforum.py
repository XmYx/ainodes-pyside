import ldm_deforum.modules.diffusionmodules.util
import ldm_deforum.models.diffusion.ddim
import ldm_deforum.models.diffusion.plms
from backend.deforum.six.hijack import hijack_codes

ddim_decode = ldm_deforum.models.diffusion.ddim.DDIMSampler.decode

plms_sampling_safe = ldm_deforum.models.diffusion.plms.PLMSSampler.plms_sampling


def undo_hijack():
    ldm_deforum.models.diffusion.ddim.DDIMSampler.decode = ddim_decode
    ldm_deforum.models.diffusion.plms.PLMSSampler.plms_sampling = plms_sampling_safe

def deforum_hijack():
    print('hijack util')
    ldm_deforum.models.diffusion.ddim.DDIMSampler.decode = hijack_codes.decode
    ldm_deforum.models.diffusion.plms.PLMSSampler.plms_sampling = hijack_codes.plms_sampling
