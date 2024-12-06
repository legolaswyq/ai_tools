import os 
import glob 


test = "/home/walter/big_daddy/imagr_instore_v3/jpgs"

bayers = glob.glob(f"{test}/**/*.jpg", recursive=True)

for bayer in bayers:
    file_size_bytes = os.path.getsize(bayer)
    # print(file_size_bytes)
    if file_size_bytes < 50000:
        print(bayer)
        print(file_size_bytes)
