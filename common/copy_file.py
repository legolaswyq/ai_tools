import shutil 
import glob 
import os 
import re 
import pprint
import numpy as np
import random
import multiprocessing as mp

def process(bar, src_dir, save_dir):
    src_dir_barcode = os.path.join(src_dir, bar)
    # save_dir_barcode = os.path.join(save_dir, bar)
    # os.makedirs(save_dir_barcode, exist_ok=True)
    rgbs = glob.glob(f"{src_dir_barcode}/*/*.jpg")
    print(len(rgbs))
    samples = random.sample(rgbs, 10)
    for rgb in samples:
        basename = os.path.basename(rgb)
        dst = os.path.join(save_dir, basename)
        print(rgb)
        print(dst)
        shutil.copy(rgb, dst)

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

src_dir = "/home/walter/big_daddy/imagr_instore_v3/jpgs_isp3"
save_dir = "/home/walter/nas_cv/walter_stuff/raw_data/jpg/finetune_FasterRcnn"
os.makedirs(save_dir, exist_ok=True)
barcodes = os.listdir(src_dir)



# process(barcodes[0], src_dir, save_dir)
for bar in barcodes:
    print(bar)
    pool.apply_async(process, args=(bar, src_dir, save_dir))

pool.close()
pool.join()

    
