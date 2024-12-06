import os 
from PIL import Image 
import glob 
import numpy as np

# img = "/home/walter/big_daddy/walter_stuff/onboard/crops/9556001171337/1/9556001171337_1696278414505_1706603_1_00_0126.jpg"

# img = Image.open(img)
# img.show()
# resized_img = img.resize([150,150])
# BICUBIC_resized_img = img.resize([150,150], resample=Image.BICUBIC)
# resized_img.show()
# BICUBIC_resized_img.show()


img_dir = "/home/walter/big_daddy/walter_stuff/onboard/crops/9300667029415"
imgs = glob.glob(f"{img_dir}/*/*.jpg")
areas = []
for img in imgs:
    image = Image.open(img)
    width, height = image.size
    area = width * height
    print(area)
    areas.append(area)

areas = np.array(areas)
median = np.median(areas)


out = "/home/walter/median_crops/9300667029415"
os.makedirs(out)
for img in imgs:
    image = Image.open(img)
    width, height = image.size
    area = width * height
    basename = os.path.basename(img)
    dst = os.path.join(out, basename)
    if area >= median:
        image.save(dst)