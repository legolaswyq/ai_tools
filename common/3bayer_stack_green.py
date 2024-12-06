import os
import glob
from PIL import Image
import numpy as np
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage



def read_bayer_to_np(bayer):
    with open(bayer, 'rb') as f:
        bayer_np = np.fromfile(f, dtype=np.uint8)
        bayer_np = bayer_np.reshape(324, 324)

    return bayer_np


def stack(bayer1, bayer2, bayer3, cam, filename):
    bayer1_np = read_bayer_to_np(bayer1)
    bayer2_np = read_bayer_to_np(bayer2)
    bayer3_np = read_bayer_to_np(bayer3)
    stack = np.stack([bayer1_np[::2, 1::2], bayer2_np[::2, 1::2], bayer3_np[::2, 1::2]], axis=2)
    stack = stack * 7
    img = cv2.cvtColor(stack, cv2.COLOR_RGB2BGR)
    if cam == 1 or cam == 2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if cam == 0: 
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    pardir = os.path.dirname(filename)
    os.makedirs(pardir, exist_ok=True)
    cv2.imwrite(filename, img)


bayer_1 = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/black_products/data_black_products__0_1696286485581_5185594.bayer"
bayer_2 = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/black_products/data_black_products__0_1696286485687_5185617.bayer"
bayer_3 = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/black_products/data_black_products__0_1696286485790_5185640.bayer"

filename = "/home/walter/git/ai_tools/test.jpg"
stack(bayer_1, bayer_2, bayer_3, 0, filename)








