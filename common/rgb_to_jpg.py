import numpy as np
import cv2
import os
import glob
import re
import multiprocessing as mp

def isp_it2(img):
    bayer_norm = (img/255).astype(np.float32)
    black_level = 6/255
    white_level = 68/255
    bayer_norm = (bayer_norm - black_level)/(white_level - black_level)
    bayer_norm = bayer_norm*white_level
    bayer_norm = np.clip(bayer_norm,0.0,1.0)
    #red channel
    bayer_norm[1::2,1::2] = bayer_norm[1::2,1::2]
    #blue channel
    bayer_norm[::2,::2] = bayer_norm[::2,::2]*1.25
    bayer_norm = bayer_norm*2.5 # was 2.5
    bayer_norm = np.clip(bayer_norm,0,1)
    bayer = np.array(bayer_norm*255,dtype=np.uint8)
    img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
    return img_rgb


def rgb_to_jpg(filePath):
    filename = os.path.basename(filePath)
    h, w = [int(x) for x in filename.split('_')[-2].split("x")]
    file = open(filePath,'rb')
    rgb_image_from_file = np.fromfile(file, dtype=np.uint8)
    rgb_image = rgb_image_from_file.reshape(w,h,3)
    rgb_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2RGB)
    return rgb_image


def process_per_barcode(barcode, src_dir, save_dir):
    src = os.path.join(src_dir, barcode)
    files = glob.glob(f"{src}/**/*.rgb", recursive=True)
    for file in files:
        # basename = os.path.basename(file)
        # save_name = re.sub(".rgb", ".jpg", basename)
        # dst_dir = os.path.join(save_dir, barcode)
        # os.makedirs(dst_dir, exist_ok=True)
        # dst = os.path.join(dst_dir, save_name)
        dst = re.sub(src_dir, save_dir, file)
        dst = re.sub(".rgb", ".jpg", dst)
        pardir = os.path.dirname(dst)
        os.makedirs(pardir, exist_ok=True)
        rgb = rgb_to_jpg(file)
        cv2.imwrite(dst, rgb)


src_dir = "/home/walter/big_daddy/imagr-nnr-120324_pi-83_classifier_events"
barcodes = os.listdir(src_dir)
save_dir = "/home/walter/big_daddy/imagr-nnr-120324_pi-83_classifier_events_crop"
os.makedirs(save_dir, exist_ok=True)

cpus = mp.cpu_count()
pool = mp.Pool(processes=cpus)
for barcode in barcodes:
    pool.apply_async(process_per_barcode, (barcode, src_dir, save_dir))

pool.close()
pool.join()


# process_per_barcode(barcodes[0], src_dir, save_dir)