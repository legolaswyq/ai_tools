import os 
import glob 
from PIL import Image 
import numpy as np
import cv2

def isp_it2(img):
    # bayer_norm = (img/255).astype(np.float32)
    # black_level = 6/255
    # white_level = 131/255
    # bayer_norm = (bayer_norm - black_level)/(white_level - black_level)

    # black_level = 0
    # white_level = 131 
    # bayer_norm = img - black_level / (white_level)
    # bayer_norm = bayer_norm*white_level
    # bayer_norm = np.clip(bayer_norm, 0.0, 1.0)
    # # red channel
    # bayer_norm[1::2, 1::2] = bayer_norm[1::2, 1::2]
    # # blue channel
    # bayer_norm[::2, ::2] = bayer_norm[::2, ::2]*1.25
    # bayer_norm = bayer_norm*2.5
    # bayer_norm = np.clip(bayer_norm, 0, 1)
    # bayer = np.array(bayer_norm*255, dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
    return img_rgb

bayers = glob.glob("/home/walter/nas_cv/walter_stuff/raw_data/bayer/black_products/*.bayer")
# for bayer in bayers:

bayer = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/black_products/data_black_products__1_1696286481098_5180768.bayer"
bayer = bayers[10]

data = open(bayer, 'rb')
bayer_np = np.fromfile(data, dtype=np.uint8)
bayer_np = bayer_np.reshape(324, 324)
img_isp = isp_it2(bayer_np)
img_gammad = np.array(255*(img_isp/255)**0.65, dtype='uint8')
# rotated_img = cv2.rotate(img_gammad, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotated_img = cv2.rotate(img_gammad, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("img", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()