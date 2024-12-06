import os
import tensorflow as tf
import hashlib
import re
import glob
import argparse


class FindLebel():
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir

    def get_label_by(self, img_path):
        label_path = re.sub(self.img_dir, self.label_dir, img_path)
        label_path = re.sub(".jpg", ".txt", label_path)
        if os.path.exists(label_path):
            return label_path
        else: 
            return None


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
            # SSD item label is 1
            cls_ids.append(int(data[0]) + 1) 
            x, y, w, h = data[1], data[2], data[3], data[4]
            bbox = xywh2xyxy([x, y, w, h])
            bboxes.append(bbox)
            
    return cls_ids, bboxes


def load_xyxy_label(file):
    """load xmin, ymin, xmax, ymax range from [0, 255] label
    return class ids and bboxs

    Args:
        file (path): path to the txt file 

    Returns:
        list, list: list of class ids and list of bbox
    """
    bboxs = []
    cls_ids = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.split()
            data = list(map(lambda x: float(x), data))
            cls_ids.append(int(data[0]))
            xmin = data[1]
            ymin = data[2]
            xmax = data[3]
            ymax = data[4]
            bboxs.append([xmin, ymin, xmax, ymax])
    return cls_ids, bboxs 





def generate_example(img_path, label_finder: FindLebel, im_w=324, im_h=324, cls_id=1, cls_text=None):
    """
    cls_label is 1 for SSD Mobilenetv2 for item, 0 for background
    img_path: path to jpg image 
    bboxs: list of bbox [xmin, ymin, xmax, ymax] range from [0, 255]

    Returns:
        tf.train.Example: tfrecord example contains all the info needed for training 
    """
    assert os.path.exists(img_path)
    
    label_path = label_finder.get_label_by(img_path)
    if not label_path:
        return 
    cls_ids, bboxes = read_yolo_anno(label_path)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_texts = []
    

    encoded_jpg = open(img_path, 'rb').read()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    for bbox in bboxes:
        xmins.append(bbox[0])
        ymins.append(bbox[1])
        xmaxs.append(bbox[2])
        ymaxs.append(bbox[3])
    
    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_h])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[im_w])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_texts)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=cls_ids)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example


def create_tfrecord(imgs_path,  tfrecord_name, label_finder: FindLebel):
    with tf.io.TFRecordWriter(tfrecord_name) as writer:
        for img_path in imgs_path:
            example = generate_example(img_path, label_finder)
            if example:
                writer.write(example.SerializeToString())



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images_dir', required=True, help='path to the images dir')
    parser.add_argument('-l', '--labels_dir', required=True, help='path to the labels dir')
    parser.add_argument('-t', '--tfrecord_save_dir', required=True, help='path to save the tfrecord')
    parser.add_argument('-n', '--tfrecord_name_prefix', default="tfrecord", help='prefix name for tfrecord')
    args = parser.parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir
    label_finder = FindLebel(img_dir=images_dir, label_dir=labels_dir)

    tfrecord_save_dir = args.tfrecord_save_dir
    os.makedirs(tfrecord_save_dir, exist_ok=True)
    tfrecord_name_prefix = args.tfrecord_name_prefix

    # get all the images in the image dir 
    imgs = glob.glob(f"{images_dir}/*.jpg")

    # split the images to create a tfrecord with 1000 or less imaegs 
    splits = int(len(imgs) / 1000) + 1
    for i in range(splits):
        tfrecord_name = f"{tfrecord_save_dir}/{tfrecord_name_prefix}_{i:04d}.tfrecord"
        start = i * 1000
        end = min(len(imgs), start + 1000)
        create_tfrecord(imgs[start:end], tfrecord_name, label_finder)




if __name__ == "__main__":
    main()