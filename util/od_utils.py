import os
import tensorflow as tf
import re
import glob
from PIL import Image

class FindLebel():
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

    def get_label_by(self, img_path):
        label_path = re.sub(self.img_dir, self.label_dir, img_path)
        label_path = re.sub(".jpg", ".txt", label_path)

        return label_path if os.path.exists(label_path) else None


def xywh2xyxy(bbox):
    "bbox is a list contains x,y,w,h"
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    return [xmin, ymin, xmax, ymax]


def read_yolo_anno(anno_path):
    """yolo anno [label, x, y, w, h] in normalized form 

    Args:
        anno_path (path): path to the yolo anno file

    Returns:
        dict: return a dict of yolo anno 
    """
    cls_ids = []
    bboxes = []
    with open(anno_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            data = [float(x) for x in data]
            cls_ids.append(int(data[0])) 
            x, y, w, h = data[1], data[2], data[3], data[4]
            bbox = xywh2xyxy([x, y, w, h])
            bboxes.append(bbox)
            
    return cls_ids, bboxes



def resize_crop_pad(image, target_width, target_height):
    # Open image and get its dimensions
    img = Image.open(image)
    original_width, original_height = img.size

    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # Determine scaling factors for resizing
    if original_aspect_ratio > target_aspect_ratio:
        # Scale according to width
        scale_factor = target_width / original_width
    else:
        # Scale according to height
        scale_factor = target_height / original_height

    # Resize the image while maintaining aspect ratio
    resized_image = img.resize((int(original_width * scale_factor), int(original_height * scale_factor)), Image.ANTIALIAS)

    # Create a new image with the target size and fill it with zeros
    padded_image = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))

    # Calculate position to paste the resized image
    paste_x = (target_width - resized_image.width) // 2
    paste_y = (target_height - resized_image.height) // 2

    # Paste the resized image onto the padded image
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image