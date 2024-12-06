import os 
import glob 
import re 
import shutil

src = "/home/walter/Downloads/od-events-testset"
basename = os.path.basename(src)


img_dir = f"{src}/images/{basename}"
label_dir = f"{src}/labels/{basename}"
os.makedirs(img_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

imgs = glob.glob(f"{src}/*.jpg")
labels = glob.glob(f"{src}/*.txt")

print(len(imgs))
print(len(labels))

for img in imgs:
    dst = os.path.join(img_dir, os.path.basename(img))
    shutil.move(img, dst)

for label in labels:
    dst = os.path.join(label_dir, os.path.basename(label))
    shutil.move(label, dst)