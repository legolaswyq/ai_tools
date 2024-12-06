import os 
import cv2

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


def scale_bbox_by_img(bboxes, img):
    s_bboxes = []
    height, width, channel = img.shape
    for bbox in bboxes:
        s_bbox = [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)]
        s_bboxes.append(s_bbox)
    return s_bboxes


label_path = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/labels/isp_gamma/train/cam0/imagr-260923_reiko_9310072032740_08bc0543-69a1-4694-2f21-eb710ddef109_0_1695685864224_8363551.txt"
img_path = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/isp_gamma/train/cam0/imagr-260923_reiko_9310072032740_08bc0543-69a1-4694-2f21-eb710ddef109_0_1695685864224_8363551.jpg"

img = cv2.imread(img_path)
cls_ids, bboxes = read_yolo_anno(label_path)
print(cls_ids)
print(bboxes)
s_bboxes = scale_bbox_by_img(bboxes, img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.equalizeHist(img)


for b in s_bboxes:
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255,0,0), 1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
