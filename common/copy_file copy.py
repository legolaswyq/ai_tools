import shutil 
import glob 
import os 
import re 
import pprint
import numpy as np
import os 
import multiprocessing as mp
import random 


def process(bar, src_dir, save_dir):
    src_dir_barcode = os.path.join(src_dir, bar)
    
    cam0 = glob.glob(f"{src_dir_barcode}/*/*/cam0/*.bayer", recursive=True)
    cam1 = glob.glob(f"{src_dir_barcode}/*/*/cam1/*.bayer", recursive=True)
    cam2 = glob.glob(f"{src_dir_barcode}/*/*/cam2/*.bayer", recursive=True)
    cam0_20 = random.sample(cam0, 20)
    cam1_20 = random.sample(cam1, 20)
    cam2_20 = random.sample(cam2, 20)

    cam0_save = os.path.join(save_dir, "bayer", "cam0")
    os.makedirs(cam0_save, exist_ok=True)
    cam1_save = os.path.join(save_dir, "bayer", "cam1")
    os.makedirs(cam1_save, exist_ok=True)
    cam2_save = os.path.join(save_dir, "bayer", "cam2")
    os.makedirs(cam2_save, exist_ok=True)

    for bayer in cam0_20:
        basename = os.path.basename(bayer)
        dst = os.path.join(cam0_save, basename)
        shutil.copy(bayer, dst)

    for bayer in cam1_20:
        basename = os.path.basename(bayer)
        dst = os.path.join(cam1_save, basename)
        shutil.copy(bayer, dst)

    for bayer in cam2_20:
        basename = os.path.basename(bayer)
        dst = os.path.join(cam2_save, basename)
        shutil.copy(bayer, dst)

    # for rgb in rgbs:
    #     dst = re.sub(src_dir, save_dir, rgb)
    #     pardir = os.path.dirname(dst)
    #     os.makedirs(pardir, exist_ok=True)
    #     shutil.copy(rgb, dst)


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

datadirs = ["imagr_instore_500_rand_mvp_260224", "imagr_instore_500_rand_mvp_240224", "imagr_instore_500_rand_mvp_290224", "imagr_instore_500_rand_pi-128_240224", "imagr_instore_500_rand_pi-83_240224", "imagr_instore_500_rand_pi-83_260224"]

src_dir = "/home/walter/big_daddy/nigel/hm01b0_data/data_collection_exp/imagr_instore_500_rand_pi-83_260224"
save_dir = "/home/walter/big_daddy/20240313_od_data_sample_from_imagr_instore_755"
os.makedirs(save_dir, exist_ok=True)
barcodes = os.listdir(src_dir)

# process("Sat-Feb-24-1310_9400597032889", src_dir, save_dir)

for bar in barcodes:
    print(bar)
    pool.apply_async(process, args=(bar, src_dir, save_dir))

pool.close()
pool.join()

    
