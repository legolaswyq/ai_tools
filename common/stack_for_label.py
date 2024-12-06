import os
import glob
import numpy as np
import random
import cv2 


def read_bayer_to_np(bayer):
    with open(bayer, 'rb') as f:
        bayer_np = np.fromfile(f, dtype=np.uint8)
        bayer_np = bayer_np.reshape(324, 324)

    return bayer_np


def isp_it2(img):
    bayer_norm = (img/255).astype(np.float32)
    black_level = 6/255
    white_level = 68/255
    bayer_norm = (bayer_norm - black_level)/(white_level - black_level)
    bayer_norm = bayer_norm*white_level
    bayer_norm = np.clip(bayer_norm, 0.0, 1.0)
    # red channel
    bayer_norm[1::2, 1::2] = bayer_norm[1::2, 1::2]
    # blue channel
    bayer_norm[::2, ::2] = bayer_norm[::2, ::2]*1.25
    bayer_norm = bayer_norm*2.5
    bayer_norm = np.clip(bayer_norm, 0, 1)
    bayer = np.array(bayer_norm*255, dtype=np.uint8)
    img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
    return img_rgb


def bayer_to_rgb(bayer):
    bayer_np = read_bayer_to_np(bayer)
    img_rgb = isp_it2(bayer_np)
    return img_rgb


def stack_rgb_green_channel(bayer1, bayer2, bayer3, cam, filename, filename_2):
    rgb1_np = bayer_to_rgb(bayer1)
    rgb2_np = bayer_to_rgb(bayer2)
    rgb3_np = bayer_to_rgb(bayer3)
    stack = np.stack([rgb1_np[:,:,1], rgb2_np[:,:,1], rgb3_np[:,:,1]], axis=2)
    # stack = stack * 7
    # img = cv2.cvtColor(stack, cv2.COLOR_RGB2BGR)
    if cam == 1 or cam == 2:
        img = cv2.rotate(stack, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb3_np = cv2.rotate(rgb3_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    if cam == 0: 
        img = cv2.rotate(stack, cv2.ROTATE_90_CLOCKWISE)
        rgb3_np = cv2.rotate(rgb3_np, cv2.ROTATE_90_CLOCKWISE)
    
    pardir = os.path.dirname(filename)
    os.makedirs(pardir, exist_ok=True)
    cv2.imwrite(filename, img)

    
    pardir = os.path.dirname(filename_2)
    os.makedirs(pardir, exist_ok=True)
    cv2.imwrite(filename_2, rgb3_np)


def stack_bayer_green_channel(bayer1, bayer2, bayer3, cam, filename):
    bayer1_np = read_bayer_to_np(bayer1)
    bayer2_np = read_bayer_to_np(bayer2)
    bayer3_np = read_bayer_to_np(bayer3)
    # stack = np.stack([bayer1_np[::2, 1::2], bayer2_np[::2, 1::2], bayer3_np[::2, 1::2]], axis=2)
    stack = np.stack([bayer1_np, bayer2_np, bayer3_np], axis=2)
    # stack = stack * 7
    # img = cv2.cvtColor(stack, cv2.COLOR_RGB2BGR)
    print(cam)
    if cam == 1 or cam == 2:
        img = cv2.rotate(stack, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if cam == 0: 
        img = cv2.rotate(stack, cv2.ROTATE_90_CLOCKWISE)
    
    pardir = os.path.dirname(filename)
    os.makedirs(pardir, exist_ok=True)
    cv2.imwrite(filename, img)


random.seed(42)
cam_id = "cam0"
src_dir = f"/home/walter/nas_cv/event_detection_datasets/all_cameras_v5/{cam_id}"

label_train = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/for_label/train/{cam_id}"
label_test = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/for_label/test/{cam_id}"
train_save_dir = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/stack_green/train/{cam_id}"
test_save_dir = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/stack_green/test/{cam_id}"
single_jpg_train_dir = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/single/train/{cam_id}"
single_jpg_test_dir = f"/home/walter/nas_cv/walter_stuff/raw_data/jpg/sg_single_view/single/test/{cam_id}"

events = glob.glob(f"{src_dir}/something/*")
num_events = len(events)

random.shuffle(events)
split_index = int(0.8 * num_events)

train_events = events[:split_index]
test_events = events[split_index:]


counter = 1

for event in train_events:
    bayers = glob.glob(f"{event}/*.bayer")
    bayers = sorted(bayers)
    if len(bayers) >= 3:
        for i in range(len(bayers) - 2):
            filename = f"{counter:06d}.jpg"
            counter += 1
            train_dst = os.path.join(train_save_dir, filename)
            label_dst = os.path.join(label_train, filename)
            filename_2 = os.path.join(single_jpg_train_dir, filename)
            stack_rgb_green_channel(bayers[i], bayers[i+1], bayers[i+2], 0, label_dst, filename_2)
            stack_bayer_green_channel(bayers[i], bayers[i+1], bayers[i+2], 0, train_dst)

counter = 1

for event in test_events:
    bayers = glob.glob(f"{event}/*.bayer")
    bayers = sorted(bayers)
    if len(bayers) >= 3:
        for i in range(len(bayers) - 2):
            filename = f"{counter:06d}.jpg"
            counter += 1
            train_test_dst = os.path.join(test_save_dir, filename)
            label_test_dst = os.path.join(label_test, filename)
            filename_2 = os.path.join(single_jpg_test_dir, filename)
            stack_rgb_green_channel(bayers[i], bayers[i+1], bayers[i+2], 0, label_test_dst, filename_2)
            stack_bayer_green_channel(bayers[i], bayers[i+1], bayers[i+2], 0, train_test_dst)