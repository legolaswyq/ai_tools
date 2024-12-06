import os
import glob
import numpy as np
import random
import cv2 
import re 
import shutil
import multiprocessing as mp


def find_valid_event(src_dir, cam):
    valid_events = []
    events = glob.glob(f"{src_dir}/{cam}/something/*")
    for event in events:
        bayers = glob.glob(f"{event}/*.bayer")
        if len(bayers) > 3:
            valid_events.append(event)
    return valid_events



def find_smallest_num_of_valid_events(src_dir):
    cams = os.listdir(src_dir)
    smallest = np.inf
    for cam in cams:
        counter = 0
        events = glob.glob(f"{src_dir}/{cam}/something/*")
        for event in events:
            bayers = glob.glob(f"{event}/*.bayer")
            if len(bayers) > 3:
                counter += 1
        if counter < smallest:
            smallest = counter

    return smallest

def copy_bayers_by_events(events, save_dir, train_test, cam):
    for event in events:
        bayers = glob.glob(f"{event}/*.bayer")
        if len(bayers) > 3:
            event_uuid = os.path.basename(event)
            event_save_dir = os.path.join(save_dir, train_test, cam, event_uuid)
            os.makedirs(event_save_dir, exist_ok=True)
            for bayer in bayers:
                basename = os.path.basename(bayer)
                dst = os.path.join(event_save_dir, basename)
                shutil.copy(bayer, dst)




random.seed(42)
MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)

src_dir = f"/home/walter/nas_cv/event_detection_datasets/all_cameras_v5"
save_dir = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/ed_events"

smallest = find_smallest_num_of_valid_events(src_dir)
num_train = int(smallest * 0.8)

cams = os.listdir(src_dir)
for cam in cams:
    counter = 0
    events = find_valid_event(src_dir, cam)
    random.shuffle(events)
    train_events = events[0:num_train]
    test_events = events[num_train:smallest]
    

#     # copy_bayers_by_events(train_events, save_dir, "train", cam)
#     # copy_bayers_by_events(test_events, save_dir, "test", cam)

    pool.apply_async(copy_bayers_by_events, args=(train_events, save_dir, "train", cam))
    pool.apply_async(copy_bayers_by_events, args=(test_events, save_dir, "test", cam))

pool.close()
pool.join()
