import os 
import shutil 
import re


def copy_file_to_new_dir(src, dst_dir):
    filename = os.path.basename(src)
    dst = os.path.join(dst_dir, filename)
    shutil.copyfile(src, dst)