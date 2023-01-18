#Import required Image library
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps


def add_watermark(path='', watermark='watermark', font_size=36, rotation=0.0, pos_x=100, pos_y=100, fill=100, destination=''):


    #Create an Image Object from an Image
    im = Image.open(path)
    width, height = im.size

    draw = ImageDraw.Draw(im)
    text = watermark

    font = ImageFont.truetype('arial.ttf', font_size)

    txt=Image.new('L', (width,font_size))
    d = ImageDraw.Draw(txt)
    d.text( (0, 0), watermark, font=font, fill=fill)
    watermark_image=txt.rotate(angle=rotation, expand=True)
    im.paste( ImageOps.colorize(watermark_image, (0,0,0), (255,150,0)), (pos_x, pos_y),  watermark_image)

    if destination != '':
        (dirname, filename) = os.path.split(path)
        (base, ext) = os.path.splitext(filename)
        path = os.path.join(destination, base + '.watermarked.png')
    else:
        (dirname, filename) = os.path.split(path)
        (base, ext) = os.path.splitext(filename)
        path = os.path.join(dirname, base + '.watermarked.png')

    print(path)

    #Save watermarked image
    im.save(path)
