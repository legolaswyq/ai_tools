import os 
import glob 
import shutil 


src = "/home/walter/Downloads/event_barcodes"
save_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset"

imgs = glob.glob(f"{src}/*/*.jpg")
labels = glob.glob(f"{src}/*/*.txt")


for img in imgs:
    saveDir = os.path.join(save_dir, "images", "od_events")
    basename = os.path.basename(img)
    os.makedirs(saveDir, exist_ok=True)
    dst = os.path.join(saveDir, basename)
    shutil.copy(img, dst)


for label in labels:
    saveDir = os.path.join(save_dir, "labels", "od_events")
    basename = os.path.basename(label)
    os.makedirs(saveDir, exist_ok=True)
    dst = os.path.join(saveDir, basename)
    shutil.copy(label, dst)
