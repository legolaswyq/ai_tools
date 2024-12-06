import os
import glob
import numpy as np
import random
import cv2 
import re
import pprint
import multiprocessing as mp
import tqdm


def read_bayer_to_np(bayer):
    with open(bayer, 'rb') as f:
        bayer_np = np.fromfile(f, dtype=np.uint8)
        bayer_np = bayer_np.reshape(324, 324)
        # bayer_np = bayer_np * 3
        # bayer_np = np.clip(bayer_np, 0, 255)
    return bayer_np


def stack_3_bayers(bayer1, bayer2, bayer3, filename):
    bayer1_np = read_bayer_to_np(bayer1)
    bayer2_np = read_bayer_to_np(bayer2) 
    bayer3_np = read_bayer_to_np(bayer3) 
    stack = np.stack([bayer1_np, bayer2_np, bayer3_np], axis=2)
    stack = stack * 3
    
    if "cam1" in bayer1 or "cam2" in bayer1:
        img = cv2.rotate(stack, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "cam0" in bayer1: 
        img = cv2.rotate(stack, cv2.ROTATE_90_CLOCKWISE)
    
    pardir = os.path.dirname(filename)
    os.makedirs(pardir, exist_ok=True)
    cv2.imwrite(filename, img)


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


def isp_gamma(bayer, filename):
    bayer_np = read_bayer_to_np(bayer)
    img_isp = isp_it2(bayer_np)
    img_gamma = np.array(255*(img_isp/255)**0.65, dtype='uint8')
    if "cam1" in bayer or "cam2" in bayer:
        img = cv2.rotate(img_gamma, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "cam0" in bayer: 
        img = cv2.rotate(img_gamma, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(filename, img)


def apply_gamma(image, gamma):
    # Generate a lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply the lookup table to the image
    corrected_image = cv2.LUT(image, table)
    return corrected_image


def improve_quality(bayer, filename):
    bayer_np = read_bayer_to_np(bayer)
    img_isp = isp_it2(bayer_np)
    # img_gamma = np.array(255*(img_isp/255)**0.65, dtype='uint8')
    gamma_corrected_image = apply_gamma(img_isp, 1.5)
    brightened_image = np.clip(gamma_corrected_image * 1.5, 0, 255).astype(np.uint8)
    if "cam1" in bayer or "cam2" in bayer:
        img = cv2.rotate(brightened_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "cam0" in bayer: 
        img = cv2.rotate(brightened_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(filename, img)


def process_event(event, bayer_event_dir, stack_bayer_save_dir, isp_gamma_save_dir, improve_quality_save_dir):
    bayers = glob.glob(f"{event}/*.bayer")
    bayers = sorted(bayers)

    for i in range(len(bayers) - 2):
        stack_bayer_name = re.sub(bayer_event_dir, stack_bayer_save_dir, bayers[i+1])
        stack_bayer_name = os.path.join(os.path.dirname(os.path.dirname(stack_bayer_name)), os.path.basename(stack_bayer_name))
        
        stack_bayer_name = re.sub(".bayer", ".jpg", stack_bayer_name)
        stack_pdir = os.path.dirname(stack_bayer_name)
        os.makedirs(stack_pdir, exist_ok=True)
        stack_3_bayers(bayers[i], bayers[i+1], bayers[i+2], stack_bayer_name)

        isp_gamma_name = re.sub(bayer_event_dir, isp_gamma_save_dir, bayers[i+1])
        isp_gamma_name = os.path.join(os.path.dirname(os.path.dirname(isp_gamma_name)), os.path.basename(isp_gamma_name))
        isp_gamma_name = re.sub(".bayer", ".jpg", isp_gamma_name)
        isp_gamma_pdir = os.path.dirname(isp_gamma_name)
        os.makedirs(isp_gamma_pdir, exist_ok=True)
        isp_gamma(bayers[i+1], isp_gamma_name)

        improve_quality_name = re.sub(bayer_event_dir, improve_quality_save_dir, bayers[i+1])   
        improve_quality_name = os.path.join(os.path.dirname(os.path.dirname(improve_quality_name)), os.path.basename(improve_quality_name))
        improve_quality_name = re.sub(".bayer", ".jpg", improve_quality_name)
        improve_quality_pdir = os.path.dirname(improve_quality_name)
        os.makedirs(improve_quality_pdir, exist_ok=True)
        improve_quality(bayers[i+1], improve_quality_name)

random.seed(42)

bayer_event_dir = "/home/walter/nas_cv/walter_stuff/raw_data/bayer/ed_events"
stack_bayer_save_dir = "/home/walter/nas_cv/walter_stuff/raw_data/jpg/ed_events/stack_all"
isp_gamma_save_dir = "/home/walter/nas_cv/walter_stuff/raw_data/jpg/ed_events/isp_gamma"
improve_quality_save_dir = "/home/walter/nas_cv/walter_stuff/raw_data/jpg/ed_events/improve_quality"

bayer_events = glob.glob(f"{bayer_event_dir}/*/*/*")
MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)

for event in bayer_events:
    print(event)
    pool.apply_async(process_event, args=(event, bayer_event_dir, stack_bayer_save_dir, isp_gamma_save_dir, improve_quality_save_dir))
    

pool.close()
pool.join()