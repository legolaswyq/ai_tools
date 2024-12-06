import os 
import glob 
import cv2 
from PIL import Image 
# import tensorflow as tf
import numpy as np
import re

crop_dir = "/home/walter/big_daddy/imagr_instore_v3/crops"
edgetpu_seg_dir = "/home/walter/big_daddy/imagr_instore_v3/edge_jaccard_0_threshold_crop_original"
maskrcnn_seg_dir = "/home/walter/nas_cv/offline_seg_exp/mask_rcnn_imagr_instore_v3_jpgs_isp3_from_scratch_19999_infer_new/seg_imgs"

no_process = "/home/walter/git/ai_tools/seg_anno_tool/no_process.txt"
bad_barcode = "/home/walter/git/ai_tools/seg_anno_tool/bad_barcode.txt"


def resize_with_pad(image, target_height, target_width):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Calculate target aspect ratio
    target_aspect_ratio = target_width / target_height

    # Calculate resize ratio
    if aspect_ratio > target_aspect_ratio:
        # Resize based on width
        resize_ratio = target_width / width
        resized_height = int(height * resize_ratio)
        resized_width = target_width
    else:
        # Resize based on height
        resize_ratio = target_height / height
        resized_width = int(width * resize_ratio)
        resized_height = target_height

    # Resize the image
    resized_image = cv2.resize(image, (resized_width, resized_height))

    # Pad the image
    top_pad = (target_height - resized_height) // 2
    bottom_pad = target_height - resized_height - top_pad
    left_pad = (target_width - resized_width) // 2
    right_pad = target_width - resized_width - left_pad
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

    return padded_image


def delete_barcode(filename, barcode_to_delete):
    # Read the content of the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove the specified barcode
    lines = [line.strip() for line in lines if line.strip() != barcode_to_delete]

    # Write the modified content back to the file
    with open(filename, 'w') as file:
        file.write('\n'.join(lines))


target_size = (128, 128)
barcodes = []
with open(no_process, "r") as f:
    barcodes = f.readlines()

barcodes = [barcode.strip() for barcode in barcodes]


for barcode in barcodes:
    crops = glob.glob(f"{os.path.join(crop_dir, barcode)}/**/*.jpg", recursive=True)
    for crop in crops:
        
        crop_img = cv2.imread(crop)
        resize_img = resize_with_pad(crop_img, 128, 128)

        mask_crop = re.sub(crop_dir, maskrcnn_seg_dir, crop)
        
        if not os.path.exists(mask_crop):
            print("no maskrcnn")
            continue

        mask_crop_img = cv2.imread(mask_crop)
        resize_mask_img = resize_with_pad(mask_crop_img, 128, 128)

        
        edgetpu_crop = re.sub(crop_dir, edgetpu_seg_dir, crop)
        if not os.path.exists(edgetpu_crop):
            print("no edgetpu")
            continue
        edgetpu_crop_img = cv2.imread(edgetpu_crop)
        resize_edgetpu_img = resize_with_pad(edgetpu_crop_img, 128, 128)

        stack = np.hstack((resize_img, resize_mask_img, resize_edgetpu_img))
        cv2.imshow(f"{barcode}", stack)
        key = cv2.waitKey(0)

        # just jump to next barcode 
        if key == ord('n'):
            break

        if key == ord('s'):
            pass


    delete_barcode(no_process, barcode)

    
    
        