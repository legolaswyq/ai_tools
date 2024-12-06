import os 
import glob 
import shutil 
import re 

src = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/improve_quality/test"
stack_bayer_src = "/home/walter/nas_cv/walter_stuff/raw_data/jpg/ed_events/isp_gamma/test"

dst = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/isp_gamma/test"
os.makedirs(dst, exist_ok=True)

imgs = glob.glob(f"{src}/*/*.jpg")
for img in imgs:
    stack_bayer = re.sub(src, stack_bayer_src, img)
    new_dst = re.sub(stack_bayer_src, dst, stack_bayer)
    new_dst_dir = os.path.dirname(new_dst)
    os.makedirs(new_dst_dir, exist_ok=True)
    print(stack_bayer)
    print(new_dst)
    print(os.path.exists(stack_bayer))
    shutil.copy(stack_bayer, new_dst)