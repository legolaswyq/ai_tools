import os 
import glob 
import shutil 
from collections import defaultdict
import pprint
import shutil 

src_dir = "/home/walter/big_daddy/nigel/hm01b0_data/data_collection_exp/imagr_instore_500_rand_pi-83_260224"
to_move = "/home/walter/big_daddy/nigel/temp"
os.makedirs(to_move, exist_ok=True)
barcode_groups = defaultdict(list)

targets = os.listdir(src_dir)
for target in list(sorted(targets)):
    time = target.split("_")[0]
    barcode = target.split("_")[1]
    barcode_groups[barcode].append((target, time))

unique_filenames = []
duplicated_filenames = []
for barcode, entries in barcode_groups.items():
    latest_entry = max(entries, key=lambda x: x[1])
    unique_filenames.append(latest_entry[0])
    if len(entries) > 1:
        duplicated_filenames.append(entries)

# for filename in unique_filenames:
#     print(filename)


for target in targets:
    if target not in unique_filenames:
        src = os.path.join(src_dir, target)
        dst = os.path.join(to_move, target)
        shutil.move(src, dst)


print(len(targets))
print(len(unique_filenames))