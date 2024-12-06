import argparse
import os
import cv2
import numpy as np
import glob
import uuid
import re 
import multiprocessing as mp
import random
import imgaug.augmenters as iaa

def img_augmentation_sequence():
    """
    usage: 
    seq.augment_images(imgs)
    imgs is a list of image[np.array]
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-90, 90),
            shear=(-8, 8)
        ),
    ], random_order=True)

    return seq


def process_per_barcode(barcode_dir, save_root, num_sample, seq):
    crops = glob.glob(f"{barcode_dir}/**/*.jpg", recursive=True)
    barcode = barcode_dir.split("/")[-1]
    print(barcode)
    for i in range(num_sample):
        crop_3 = random.choices(crops, k=3)
        crop_3_np = [np.array(cv2.imread(x)) for x in crop_3]
        aug_3_np = seq.augment_images(crop_3_np)
        img_0 = cv2.resize(aug_3_np[0], (150, 150), interpolation=cv2.INTER_CUBIC)
        img_1 = cv2.resize(aug_3_np[1], (150, 150), interpolation=cv2.INTER_CUBIC)
        img_2 = cv2.resize(aug_3_np[2], (150, 150), interpolation=cv2.INTER_CUBIC)
        s_img = np.hstack((img_0,img_1,img_2))
        filename = uuid.uuid4()
        save_dir = os.path.join(save_root, barcode)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir , f"{filename}.jpg"), s_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        


crop_root = "/home/walter/nas_cv/offline_OD_exp/faster-rcnn-new-office-new-onboard-from_scratch-new_office_new_onboard-19999-infer/crop_imgs_epoch_19999/"
save_root = "/home/walter/big_daddy/new_office_new_onboard_random_3_view"
os.makedirs(save_root, exist_ok=True)
seq = img_augmentation_sequence()
barcodes = os.listdir(crop_root)

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

for bar in barcodes:
    barcode_dir = os.path.join(crop_root, bar)
    pool.apply_async(process_per_barcode, args=(barcode_dir, save_root, 1000, seq,))



pool.close()
pool.join()