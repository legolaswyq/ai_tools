import argparse
import os
import cv2
import numpy as np
import glob
import uuid
import re 
import multiprocessing as mp


def process_per_barcode(file, save_root, crop_dir):
    barcode = os.path.basename(file).split(".")[0]
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            crop_0 = os.path.join(crop_dir, barcode, "cam0", data[0])
            crop_1 = os.path.join(crop_dir, barcode, "cam1", data[1])
            crop_2 = os.path.join(crop_dir, barcode, "cam2", data[2])
            crop_0 = re.sub(".bayer", ".jpg", crop_0)
            crop_1 = re.sub(".bayer", ".jpg", crop_1)
            crop_2 = re.sub(".bayer", ".jpg", crop_2)
            if os.path.exists(crop_0) and os.path.exists(crop_1) and os.path.exists(crop_2):
                img_0 = cv2.imread(crop_0)
                img_1 = cv2.imread(crop_1)
                img_2 = cv2.imread(crop_2)
                img_0 = cv2.resize(img_0, (150, 150), interpolation=cv2.INTER_CUBIC)
                img_1 = cv2.resize(img_1, (150, 150), interpolation=cv2.INTER_CUBIC)
                img_2 = cv2.resize(img_2, (150, 150), interpolation=cv2.INTER_CUBIC)
                s_img=np.hstack((img_0,img_1,img_2))
                filename = uuid.uuid4()
                save_dir = os.path.join(save_root, barcode)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir , f"{filename}.jpg"), s_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

txt_file_dir = "/home/walter/big_daddy/imagr_instore_v5/matching_txt"
txt_files = glob.glob(f"{txt_file_dir}/*.txt")
print(len(txt_files))
# txt_files = ["/home/walter/big_daddy/new_office_classifier_txt/Thu-Feb-29-1626_brown_bag.txt"]

# crop_dir = "/home/walter/nas_cv/offline_OD_exp/faster-rcnn-new-office-new-onboard-from_scratch-new_office_new_onboard-19999-infer/crop_imgs_epoch_19999"
seg_dir = "/home/walter/big_daddy/imagr_instore_v5/seg_isp3_maskrcnn"
save_root = "/home/walter/big_daddy/imagr_instore_v5_3view_mask_rcnn_seg"

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)
for file in txt_files:
    pool.apply_async(process_per_barcode, args=(file, save_root, seg_dir,))

pool.close()
pool.join()


# for file in txt_files:
#     process_per_barcode(file, save_root, seg_dir)
