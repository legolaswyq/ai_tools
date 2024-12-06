import glob 
import os 
import numpy as np 
from PIL import Image 
import shutil 
import re 

out = "/home/walter/big_daddy/nigel/hm01b0_data/imagr_nnr_events/pil_crops"
shopping_event_dir = "/home/walter/big_daddy/nigel/hm01b0_data/imagr_nnr_events/shopping_events"
barcodes = os.listdir(shopping_event_dir)

for barcode in barcodes:
    barcode_dir = os.path.join(shopping_event_dir, barcode)
    events_dir = os.path.join(shopping_event_dir, barcode, "classifier_events")
    events = os.listdir(events_dir)
    for event in events:
        save_dir = os.path.join(out, barcode, event)
        os.makedirs(save_dir, exist_ok=True)
        event_dir = os.path.join(events_dir, event)
        files = os.listdir(event_dir)
        event_ids = [f for f in files if '.' not in f]
        for event_id in event_ids:
            dets = glob.glob(f"{barcode_dir}/*/{event_id}/*.od")
            for det in dets:
                basename = os.path.basename(det)
                rgb = re.sub(".od", ".rgb", det)
                rgb_basename = os.path.basename(rgb)
                shutil.copy2(det, os.path.join(save_dir, basename))
                shutil.copy2(rgb, os.path.join(save_dir, rgb_basename))
           