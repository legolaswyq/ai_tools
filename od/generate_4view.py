import os 
import glob 
import numpy as np
import random
from PIL import Image

def form_fourview(images, seq, image_size=320):
    imgs_np = []
    for image in images:
        img = Image.open(image)
        img = img.resize((int(image_size/2) , int(image_size/2)))
        img_np = np.array(img)
        imgs_np.append(img_np)
    
    if seq:
        imgs_np.extend(seq.augment_images(imgs_np))

    while len(imgs_np) < 4:
        imgs_np.extend(imgs_np)

    print(len(imgs_np))
    imgs = random.sample(imgs_np, k=4)

    fourview_np = np.vstack(
        [np.hstack([imgs[0], imgs[1]]),
        np.hstack([imgs[2], imgs[3]])]
    )
    
    return Image.fromarray(fourview_np)

out = "/home/walter/big_daddy/nigel/hm01b0_data/imagr_nnr_events/pil_fourview"
src_dir = "/home/walter/big_daddy/nigel/hm01b0_data/imagr_nnr_events/pil_crops"
barcodes = os.listdir(src_dir)
for barcode in barcodes:
    barcode_dir = os.path.join(src_dir, barcode)
    events = os.listdir(barcode_dir)
    for event in events:
        event_dir = os.path.join(barcode_dir, event)
        crops = glob.glob(f"{event_dir}/*.jpg")
        fourview = form_fourview(crops, None)
        os.makedirs(f"{out}/{barcode}", exist_ok=True)
        fourview.save(f"{out}/{barcode}/{event}.jpg")
        