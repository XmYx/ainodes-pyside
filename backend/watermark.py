#Import required Image library
from PIL import Image, ImageDraw, ImageFont, ImageOps


def add_watermark(path='', watermark='watermark', font_size=36):


    #Create an Image Object from an Image
    im = Image.open(path)
    width, height = im.size

    draw = ImageDraw.Draw(im)
    text = watermark

    font = ImageFont.truetype('arial.ttf', font_size)

    txt=Image.new('L', (width,100))
    d = ImageDraw.Draw(txt)
    d.text( (0, 0), watermark,  font=font, fill=150)
    w=txt.rotate(17.5,  expand=1)
    im.paste( ImageOps.colorize(w, (0,0,0), (255,150,0)), (100,120),  w)

    #Save watermarked image
    im.save(path + '.watermarked.png')
