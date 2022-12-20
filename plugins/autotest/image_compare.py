import os


import cv2
import numpy as np
from PIL import Image, ImageChops

def img_compare(file1, file2):
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    image = ImageChops.difference(img1, img2)
    tmp_img = 'plugins/autotest/tmp/tmp.png'
    if os.path.isfile(tmp_img):
        os.remove(tmp_img)
    image.save(tmp_img)
    image = cv2.imread(tmp_img)
    return np.sum(image < 1)
