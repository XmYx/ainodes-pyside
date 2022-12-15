from PIL import Image, ImageFont, ImageDraw, PngImagePlugin

from backend.singleton import singleton
gs = singleton

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
upscaler_for_img2img= None
sd_upscalers = []

def resize_image(resize_mode, im, width, height):
    def resize(im, w, h):
        if upscaler_for_img2img is None or upscaler_for_img2img == "None" or im.mode == 'L':
            return im.resize((w, h), resample=LANCZOS)

        scale = max(w / im.width, h / im.height)

        if scale > 1.0:
            upscalers = [x for x in sd_upscalers if x.name == upscaler_for_img2img]
            assert len(upscalers) > 0, f"could not find upscaler named {upscaler_for_img2img}"

            upscaler = upscalers[0]
            im = upscaler.scaler.upscale(im, scale, upscaler.data_path)

        if im.width != w or im.height != h:
            im = im.resize((w, h), resample=LANCZOS)

        return im

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res
