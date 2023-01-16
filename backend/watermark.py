#Import required Image library
from PIL import Image, ImageDraw, ImageFont, ImageOps


def add_watermark(path='', watermark='watermark', font_size=36, rotation=0.0, pos_x=100, pos_y=100, fill=100):


    #Create an Image Object from an Image
    im = Image.open(path)
    width, height = im.size

    draw = ImageDraw.Draw(im)
    text = watermark

    font = ImageFont.truetype('arial.ttf', font_size)

    txt=Image.new('L', (width,100))
    d = ImageDraw.Draw(txt)
    d.text( (0, 0), watermark,  font=font, fill=fill)
    w=txt.rotate(rotation,  expand=1)
    im.paste( ImageOps.colorize(w, (0,0,0), (255,150,0)), (pos_x, pos_y),  w)

    #Save watermarked image
    im.save(path + '.watermarked.png')
