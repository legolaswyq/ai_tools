from pycoral.utils import edgetpu
from pycoral.adapters import common
from PIL import Image
import numpy as np
import time
import os 
import re
import glob 


def resize_crop_pad(img, is_label=False):
    """
    img -> PIL.Image
    return -> PIL.Image
    """
    target_width, target_height = 128, 128
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    if original_aspect_ratio > target_aspect_ratio:
        scale_factor = target_width / original_width
    else:
        scale_factor = target_height / original_height

    resized_image = img.resize((int(original_width * scale_factor), int(original_height * scale_factor)), Image.BILINEAR)
    if is_label:
        padded_image = Image.new("L", (target_width, target_height))
    else:
        padded_image = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    paste_x = (target_width - resized_image.width) /imagr_instore_v5_3view_mask_rcnn_seg

def apply_single_channel_mask(image, mask):
    mask_3_channels = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.where(mask_3_channels, image, 0)
    return masked_image


def crop_masked_region(masked_image, mask):
    """crop out the mask region from original image 

    Args:
        masked_image (np.array): original image
        mask (np.array): single channel mask 

    Returns:
        np.array: crop
    """
    # Find bounding box of the mask
    non_zero_indices = np.argwhere(mask)
    min_y, min_x = np.min(non_zero_indices, axis=0)
    max_y, max_x = np.max(non_zero_indices, axis=0)
    # Crop out the region containing the mask
    cropped_masked_image = masked_image[min_y:max_y+1, min_x:max_x+1]
    return cropped_masked_image


def binarize_img_np(img_np):
    """
    img -> np.array
    return -> np.array
    """
    binary_img_array = np.where(img_np > 0, 1, 0)
    return binary_img_array.astype(np.uint8)




model = "/home/walter/git/barcode_pipeline/barcode_detector/tflite/train_with_maskrcnn_data.tflite"

interpreter = edgetpu.make_interpreter(model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

src_dir = "/home/walter/big_daddy/imagr_instore_v5/crops"
save_dir = "/home/walter/big_daddy/imagr_instore_v5/edgetpu_seg"
os.makedirs(save_dir, exist_ok=True)


def per_barcode(barcode, src_dir, save_dir):
    src = os.path.join(src_dir, barcode)
    files = glob.glob(f"{src}/**/*.jpg", recursive=True)
    for file in files:
        # basename = os.path.basename(file)
        # dst_dir = os.path.join(save_dir, barcode)
        # os.makedirs(dst_dir, exist_ok=True)
        # dst = os.path.join(dst_dir, basename)

        dst = re.sub(src_dir, save_dir, file)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        img = Image.open(file)
        resize_pad = resize_crop_pad(img)
        resize_pad_np = np.array(resize_pad, dtype=np.float32) / 255.
        resize_pad_np = np.expand_dims(resize_pad_np, axis=0)
        interpreter.set_tensor(input_details[0]['index'], resize_pad_np)
    
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        mask = np.squeeze(output)
        mask = binarize_img_np(mask)
        masked_image = apply_single_channel_mask(resize_pad, mask)
        crop = crop_masked_region(np.array(masked_image), mask)
        crop_pil = Image.fromarray(crop)
        crop_pil.save(dst)
        


for barcode in os.listdir(src_dir):
    per_barcode(barcode, src_dir, save_dir)

