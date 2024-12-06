import numpy as np 
from PIL import Image 
from io import BytesIO
import os 
import glob
import re
import multiprocessing as mp

def process(bar, src_dir, save_dir):
    rgbs = glob.glob(f"{os.path.join(src_dir, bar)}/*.rgb")
    for rgb in rgbs:
        basename = os.path.basename(rgb)
        savename = re.sub(".rgb", ".jpg", basename)
        dst = os.path.join(save_dir, bar, savename)
        dirname = os.path.dirname(dst)
        os.makedirs(dirname, exist_ok=True)
        data = open(rgb, 'rb')
        bayer_np = np.fromfile(data, dtype=np.uint8)
        try:
            bayer_np = bayer_np.reshape(320, 320, 3)
        except:
            continue
        img = Image.fromarray(bayer_np)
        img.save(dst)

MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)


src_dir = "/home/walter/big_daddy/walter_stuff/od_data/rgb_test"
save_dir = "/home/walter/big_daddy/walter_stuff/od_data/jpg_test"
os.makedirs(save_dir, exist_ok=True)
bars = os.listdir(src_dir)

for bar in bars:
    print(bar)
    # pool.apply_async(process, args=(bar, src_dir, save_dir))
    process(bar, src_dir, save_dir)

pool.close()
pool.join()