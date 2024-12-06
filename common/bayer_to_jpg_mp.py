# process raw bayer hm01b0 files

import numpy as np
import cv2
import os
import glob
import re
import multiprocessing as mp
import argparse


def isp_it(img):

    bayer_norm = (img/255).astype(np.float32)

    # red channel
    bayer_norm[1::2, 1::2] = bayer_norm[1::2, 1::2]
    # blue channel
    bayer_norm[::2, ::2] = bayer_norm[::2, ::2]*1.25
    bayer_norm = bayer_norm*3

    bayer_norm = np.clip(bayer_norm, 0, 1)
    bayer = np.array(bayer_norm*255, dtype=np.uint8)
    # was BG2RGB  which equates to the same thing but is wrong
    img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)

    return img_rgb


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


def isp_it3(img):
    bayer_norm = (img/255).astype(np.float32)
    black_level = 6/255
    white_level = 68/255
    bayer_norm = (bayer_norm - black_level)/(white_level - black_level)
    bayer_norm=bayer_norm*white_level
    bayer_norm=np.clip(bayer_norm,0.0,1.0)
    #red channel
    bayer_norm[1::2,1::2]=bayer_norm[1::2,1::2]
    #blue channel
    bayer_norm[::2,::2]=bayer_norm[::2,::2]*1.25
    bayer_norm=bayer_norm*2.5
    # clip the image to remove false color
    bayer_norm_clipped=np.clip(bayer_norm,None,0.18)
    #bayer_norm= np.clip(bayer_norm,0,1)
    bayer=np.array(bayer_norm_clipped*255,dtype=np.uint8)
    img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)  #was BG2RGB  which equates to the same thing but is wrong
    return img_rgb

def convert_bayer_save_jpg(bayer, root, output, cam_pos):
    # perform isp and gammad
    data = open(bayer, 'rb')
    bayer_np = np.fromfile(data, dtype=np.uint8)
    bayer_np = bayer_np.reshape(324, 324)
    img_isp = isp_it3(bayer_np)
    img_gammad = np.array(255*(img_isp/255)**0.65, dtype='uint8')

    # prepare jpg name and save path
    bayer_name = os.path.basename(bayer)
    jpg_savename = re.sub(".bayer", ".jpg", bayer_name)

    parent_dir = os.path.dirname(bayer)
    save_dir = re.sub(root, output, parent_dir)
    os.makedirs(save_dir, exist_ok=True)
    dst = os.path.join(save_dir, jpg_savename)

    # rotate base on camera position
    cam_position = bayer_name.split("_")[int(cam_pos)]

    # for shopping rotation 
    # if cam_position == "1" or cam_position == "2":
    #     rotated_img = cv2.rotate(
    #         img_gammad, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # else:
    #     rotated_img = cv2.rotate(img_gammad, cv2.ROTATE_90_CLOCKWISE)

    # for onboarding ratation 
    if cam_position == "0":
        rotated_img = cv2.rotate(
            img_gammad, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_img = cv2.rotate(
            img_gammad, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # save img
    cv2.imwrite(dst, rotated_img, [cv2.IMWRITE_JPEG_QUALITY, 100])





def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True, help='root path to the source bayers')
    parser.add_argument('-o', '--output', required=True, help='input dir')
    parser.add_argument('-c', '--cam_pos', required=True, help='cam position index on the file name')
    parser.add_argument('-g', '--glob_pattern', required=True, help='glob_pattern')
    '''
    9300633119010_1692655960542_4302751_6_00_0130.jpg
    The cam position is 3
    barcode_timestamp_time_camera_round_seq.jpg
    bayer_name.split("_")[cam_pos]
    '''
    args = parser.parse_args()

    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    root = args.root
    output = args.output
    os.makedirs(output, exist_ok=True)
    cam_pos = args.cam_pos
    glob_pattern = args.glob_pattern

    bayers = glob.glob(f"{root}{glob_pattern}")
    for bayer in bayers:
        pool.apply_async(convert_bayer_save_jpg, args=(
            bayer, root, output, cam_pos))


    pool.close()
    pool.join()


if __name__ == '__main__':  
    main()