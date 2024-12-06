import os 
import glob 
import re
import numpy as np
from PIL import Image

src_dir = "/home/walter/big_daddy/nigel/hm01b0_data/imagr_nnr_events/pil_crops"
barcodes = os.listdir(src_dir)
for barcode in barcodes:
    barcode_dir = os.path.join(src_dir, barcode)
    events = os.listdir(barcode_dir)
    for event in events:
        event_dir = os.path.join(barcode_dir, event)
        ods = glob.glob(f"{event_dir}/*.od")
        for od in ods:
            basename = os.path.basename(od).split(".")[0]
            print(basename)
            rgb = re.sub(".od", ".rgb", od)
            rgb_np = np.fromfile(rgb, dtype=np.uint8).reshape((320,320,3))
            rgb_pil = Image.fromarray(rgb_np)
            
            with open(od, 'r') as f:
                lines = f.readlines()
                print(lines)
                for line in lines:
                    infos = line.split(",")
                    print(infos)
                    print(infos[1].strip()[6:])
                    print(infos[2].strip()[6:])
                    print(infos[3].strip()[6:])
                    print(infos[4].strip()[6:])
                    ymin = int(float(infos[1].strip()[6:]) * 320)
                    xmin = int(float(infos[2].strip()[6:]) * 320)
                    ymax = int(float(infos[3].strip()[6:]) * 320)
                    xmax = int(float(infos[4].strip()[6:]) * 320)
                
                
            crop = rgb_pil.crop([xmin, ymin, xmax, ymax])
            crop.save(os.path.join(event_dir, f"{basename}.jpg"))

            
        
    