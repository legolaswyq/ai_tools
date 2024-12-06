import os 
import glob 
import cv2
import numpy as np
import pprint


def get_info(filename):
    if "/" in filename:
        filename = os.path.basename(filename)
    
    p_barcode = filename.split("_")[0]
    cam_id = filename.split("_")[2]
    timestamp = filename.split("_")[3]

    return p_barcode, int(timestamp), cam_id, filename


# 9310155005517_1708567610320_0_1708567621006_92747.bayer
src_dir = "/home/walter/big_daddy/imagr_instore_v5/jpgs"
txt_file_save_dir = "/home/walter/big_daddy/imagr_instore_v5/matching_txt"
os.makedirs(txt_file_save_dir, exist_ok=True)
barcodes = os.listdir(src_dir)
# barcodes = ["Thu-Feb-29-1626_brown_bag"]
# print(len(barcodes))


for barcode in barcodes:
    txtfile_savepath = os.path.join(txt_file_save_dir, f"{barcode}.txt")
    bayers = glob.glob(f"{src_dir}/{barcode}/*/*.jpg")
    print(len(bayers))
    find_pair = []
    cam0, cam1, cam2 = [], [], []
    for bayer in list(sorted(bayers, key=lambda x: os.path.basename(x).split("_")[3])):
        _, _, cam_id, basename = get_info(bayer)
        if cam_id == "0":
            cam0.append(basename)
        elif cam_id == "1":
            cam1.append(basename)
        elif cam_id == "2":
            cam2.append(basename)
        if len(cam0) >= 1 and len(cam1) >= 1 and len(cam2) >= 1:
            str_to_write = f"{cam0.pop()} {cam1.pop()} {cam2.pop()}"
            find_pair.append(str_to_write)
            cam0, cam1, cam2 = [], [], []
    
    for pair in find_pair:
        first_basename , second_basename, third_basename = pair.split()
        with open(txtfile_savepath, "a") as f:
            f.write(f"{first_basename} {second_basename} {third_basename}\n")