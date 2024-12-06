import cv2 
import numpy as np
import multiprocessing as mp
import glob 
import os 
import re

def apply_gamma(image, gamma):
    # Generate a lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply the lookup table to the image
    corrected_image = cv2.LUT(image, table)
    return corrected_image


def process_image(img_path, gamma, brightness_factor, dst):
    img = cv2.imread(img_path)
    gamma_corrected_image = apply_gamma(img, gamma)
    brightened_image = np.clip(gamma_corrected_image * brightness_factor, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, brightened_image)


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

gamma = 0.67
brightness_factor = 0.67

dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/od_jpg_ed_data"
save_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/worser_jpg"
imgs = glob.glob(f"{dir}/*/*/*.jpg")
print(len(imgs))

for img in imgs:
    basename = os.path.basename(img)
    dst = re.sub(dir, save_dir, img)
    dirname = os.path.dirname(dst)
    os.makedirs(dirname, exist_ok=True)
    pool.apply_async(process_image, args=(img, gamma , brightness_factor, dst))

pool.close()
pool.join()
