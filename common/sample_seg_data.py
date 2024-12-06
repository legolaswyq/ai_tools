import os 
import glob 
import shutil
import random 
import multiprocessing as mp



# barcode = barcodes[0]

def process(barcode, src_dir, save_dir):
    crops = glob.glob(f"{src_dir}/{barcode}/*/*.jpg")
    sample_crops = random.sample(crops, 10)
    for sample_crop in sample_crops:
        basename = os.path.basename(sample_crop)
        dst = os.path.join(save_dir, basename)
        shutil.copyfile(sample_crop, dst)


src_dir = "/home/walter/big_daddy/new_office_classifier_crop"
save_dir = "/home/walter/big_daddy/seg_training/images"
os.makedirs(save_dir, exist_ok=True)
barcodes = os.listdir(src_dir)


MAX_NUM_CORES = mp.cpu_count()
pool = mp.Pool(MAX_NUM_CORES-1 or 1)
print(pool)

for barcode in barcodes:
    print(barcode)
    pool.apply_async(process, args=(barcode, src_dir, save_dir))

pool.close()
pool.join()