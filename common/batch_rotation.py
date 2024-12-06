import os 
import glob
import re 
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2 
import multiprocessing as mp

def read_img_by_label(label_path, label_base_dir, img_base_dir):
    img_path = re.sub(label_base_dir, img_base_dir, label_path)
    img_path = re.sub(".txt", ".jpg", img_path)
    img = cv2.imread(img_path)
    return img


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


def xyxy2xywh(bbox):
    # xyxy to x_center, y_center, w, h
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    w = round(x2 - x1, 6)
    h = round(y2 - y1, 6)
    x_center = round((x1 + x2) / 2, 6)
    y_center = round((y1 + y2) / 2, 6)
    return [x_center, y_center, w, h]


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


def scale_bboxes(bboxes, img):
    scaled_bboxes = []
    height, width, channel = img.shape
    for bbox in bboxes:
        scaled_bboxes.append(BoundingBox(int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)))
    
    return scaled_bboxes 


def get_camera_id_by_label_path(label_path):
    dirname = os.path.dirname(label_path)
    return dirname.split("/")[-1]


def rotate_img_label(img, bboxes, cam_id):
    degree = 0
    if cam_id == "cam1" or cam_id == "cam2":
        degree = 90
    elif cam_id == "cam0":
        degree = 270

    seq = iaa.Sequential([
        iaa.Affine(rotate=degree)
    ])

    image_aug, bbs_aug = seq(image=img, bounding_boxes=bboxes)
    
    return image_aug, bbs_aug


def save_rotated_img_label(r_img, r_bboxes, label_path, label_base, label_save, image_base, image_save):
    img_path = re.sub(label_base, image_base, label_path)
    img_path = re.sub(".txt", ".jpg", img_path)
    img_dst = re.sub(image_base, image_save, img_path)
    label_dst = re.sub(label_base, label_save, label_path)

    img_dst_dir = os.path.dirname(img_dst)
    os.makedirs(img_dst_dir, exist_ok=True)
    cv2.imwrite(img_dst, r_img)

    label_dst_dir = os.path.dirname(label_dst)
    os.makedirs(label_dst_dir, exist_ok=True)
    with open(label_dst, 'w') as f:
        for r_bbox in r_bboxes:
            bbox = [r_bbox.x1, r_bbox.y1, r_bbox.x2, r_bbox.y2]
            height, width, channel = r_img.shape
            bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            x, y, w, h = xyxy2xywh(bbox)
            line = f"0 {x} {y} {w} {h}\n"
            f.write(line)


label_base_dir = "/home/walter/Downloads/stack_green"
img_base_dir = "/home/walter/nas_cv/walter_stuff/raw_data/raw_stack"

img_save_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/stack_green"
label_save_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/labels/stack_green"


labels = glob.glob(f"{label_base_dir}/*/*/*.txt")
print(len(labels))
# for label in labels:
#     print(label)
#     cls_ids, bboxes = read_yolo_anno(label)
#     cam = get_camera_id_by_label_path(label)
#     img = read_img_by_label(label, label_base_dir, img_base_dir)
#     scaled_bboxes = scale_bboxes(bboxes, img)
#     image_aug, bbs_aug = rotate_img_label(img, scaled_bboxes, cam)
#     save_rotated_img_label(image_aug, bbs_aug, label, label_base_dir, label_save_dir, img_base_dir, img_save_dir)




def process(label):
    cls_ids, bboxes = read_yolo_anno(label)
    cam = get_camera_id_by_label_path(label)
    img = read_img_by_label(label, label_base_dir, img_base_dir)
    if bboxes:
        scaled_bboxes = scale_bboxes(bboxes, img)
        image_aug, bbs_aug = rotate_img_label(img, scaled_bboxes, cam)
        save_rotated_img_label(image_aug, bbs_aug, label, label_base_dir, label_save_dir, img_base_dir, img_save_dir)
    else:
        image_aug, bbs_aug = rotate_img_label(img, scaled_bboxes, cam)



# MAX_NUM_CORES = mp.cpu_count()
# pool = mp.Pool(MAX_NUM_CORES-1 or 1)
# print(pool)

# for label in labels:
#     print(label)
#     pool.apply_async(process, args=(label,))

# pool.close()
# pool.join()