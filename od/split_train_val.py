import os 
import glob 
import re
import shutil
import random
import multiprocessing as mp

def process(img, dst_img_dir, dst_label_dir):
    copy_file_to_new_dir(img, dst_img_dir)
    label = re.sub("images", "labels", img)
    label = re.sub("\.jpg", ".txt", label)
    copy_file_to_new_dir(label, dst_label_dir)


def copy_file_to_new_dir(src, dst_dir):
    filename = os.path.basename(src)
    dst = os.path.join(dst_dir, filename)
    shutil.copyfile(src, dst)


def copy_imgs_and_labels(imgs, dst_img_dir, dst_label_dir):

    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)
    print(pool)

    for img in imgs:
        pool.apply_async(process, args=(img, dst_img_dir, dst_label_dir))

    pool.close()
    pool.join()



random.seed(42)

split_test = True
img_src_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/OB_data_sample1_252_9_5_skip30"

label_src_dir = re.sub(r"images", "labels", img_src_dir)

img_train_dir = f"{img_src_dir}_train"
img_val_dir = f"{img_src_dir}_val"
os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)

label_train_dir = re.sub(r"images", "labels", img_train_dir)
label_val_dir = re.sub(r"images", "labels", img_val_dir)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)


imgs = glob.glob(f"{img_src_dir}/*.jpg")
random.shuffle(imgs)
num_imgs = len(imgs)
print(f"num of total images: {num_imgs}")


if split_test:
    img_test_dir = f"{img_src_dir}_test"
    os.makedirs(img_test_dir, exist_ok=True)
    label_test_dir = re.sub(r"images", "labels", img_test_dir)
    os.makedirs(label_test_dir, exist_ok=True)

    split_num = int(num_imgs * 0.8)
    val_num = int(num_imgs * 0.1)
    train_imgs = imgs[:split_num]
    val_imgs = imgs[split_num: split_num+val_num]
    test_imgs = imgs[split_num+val_num:]

    copy_imgs_and_labels(train_imgs, img_train_dir, label_train_dir)
    copy_imgs_and_labels(val_imgs, img_val_dir, label_val_dir)
    copy_imgs_and_labels(test_imgs, img_test_dir, label_test_dir)
else:
    split_num = int(num_imgs * 0.8)
    train_imgs = imgs[:split_num]
    val_imgs = imgs[split_num:]

    copy_imgs_and_labels(train_imgs, img_train_dir, label_train_dir)
    copy_imgs_and_labels(val_imgs, img_val_dir, label_val_dir)