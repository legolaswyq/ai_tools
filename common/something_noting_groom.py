import os 
import glob 
import cv2 
import numpy as np
import pprint
import multiprocessing as mp
import shutil

def isp_it2(img):
    bayer_norm = (img/255).astype(np.float32)
    black_level = 6/255
    white_level = 68/255
    bayer_norm = (bayer_norm - black_level)/(white_level - black_level)
    bayer_norm=bayer_norm*white_level
    bayer_norm=np.clip(bayer_norm,0.0,1.0)
    #red channel
    bayer_norm[1::2,1::2]=bayer_norm[1::2,1::2]
    #blue channel
    bayer_norm[::2,::2]=bayer_norm[::2,::2]*1.25
    bayer_norm=bayer_norm*4
    bayer_norm= np.clip(bayer_norm,0,1)
    bayer=np.array(bayer_norm*255,dtype=np.uint8)
    img_rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
    return img_rgb


def extract_timestamp(filename):
    parts = os.path.splitext(filename)[0].split('_')  # Split the file name by underscores and remove extension
    timestamp_str = parts[5]  
    return int(timestamp_str)


def copyfile(filepath, dst_dir, bayer_dir):
    basename = os.path.basename(filepath)
    dst = os.path.join(dst_dir, basename)
    shutil.copyfile(filepath, dst)
    dst_bayer = os.path.join(bayer_dir, basename)
    shutil.copyfile(filepath, dst_bayer)

dataset_name = "eastridge_new_world_210224_mvp"
src_dir = f"/home/walter/big_daddy/offsite_data_collection/{dataset_name}"
save_root = f"/home/walter/big_daddy/offsite_data_ED_datasets/{dataset_name}/ED_data"
bayer_dir = f"/home/walter/big_daddy/offsite_data_ED_datasets/{dataset_name}/bayer"
temp_dir = f"/home/walter/big_daddy/offsite_data_ED_datasets/{dataset_name}/temp"
discard_dir = f"/home/walter/big_daddy/offsite_data_ED_datasets/{dataset_name}/discard"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(save_root, exist_ok=True)
os.makedirs(bayer_dir, exist_ok=True)
os.makedirs(discard_dir, exist_ok=True)

run_first = False

if run_first:
    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)
    print(pool)
    bayers = glob.glob(f"{src_dir}/**/*.bayer", recursive=True)
    bayers = list(sorted(bayers))
    for i in range(0, len(bayers), 10):
        print(bayers[i])
        pool.apply_async(copyfile, args=(bayers[i],temp_dir,bayer_dir))

    pool.close()
    pool.join()
else:
    bayers = glob.glob(f"{temp_dir}/*.bayer", recursive=True)
    cam0, cam1, cam2 = [], [], []
    for bayer in list(sorted(bayers)):
        filename = os.path.basename(bayer)
        timestamp_event_id = "_".join(filename.split('_')[0:2])
        print(timestamp_event_id)
        camid = filename.split('_')[2]
        
        if camid == '0':
            cam0.append(bayer)
        elif camid == '1':
            cam1.append(bayer)
        elif camid == '2':
            cam2.append(bayer)
        else:
            print('Camera ID not recognized')

        if timestamp_event_id.startswith("nothing"):
            cam_id = camid
            nothing_dir = os.path.join(save_root, f"cam{cam_id}", "nothing")
            os.makedirs(nothing_dir, exist_ok=True)
            dst_dir = os.path.join(nothing_dir,timestamp_event_id)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir,filename)
            print(bayer)
            print(dst)
            shutil.move(bayer,dst)

    print(len(cam0), len(cam1), len(cam2))

    def is_directory_empty(directory):
        return not any(os.listdir(directory))

    def per_cam(bayer_list, cam_id, rotation, discard_dir):

        restore_list = []
        something_dir = os.path.join(save_root, cam_id, "something")
        nothing_dir = os.path.join(save_root, cam_id, "nothing")
        os.makedirs(something_dir, exist_ok=True)
        os.makedirs(nothing_dir, exist_ok=True)

        while len(bayer_list) > 0:
            bayer_file = bayer_list.pop()
            filename = os.path.basename(bayer_file)
            timestamp_event_id = "_".join(filename.split('_')[0:2])
            bayer_8bit=np.fromfile(bayer_file,dtype=np.uint8)
            try:
                bayer_8bit=bayer_8bit.reshape(324,324)
            except:
                print(f"remove file {bayer_file} with size: {os.path.getsize(bayer_file)}")
                os.remove(bayer_file)
                continue
            linear_img=isp_it2(bayer_8bit)
            img_gammad=np.array(255*(linear_img/255)**0.65,dtype='uint8')
            img_gammad=np.rot90(img_gammad, rotation)
            cv2.imshow('image',img_gammad)
            key=cv2.waitKey(0)
            if key==ord('n'):
                dst_dir = os.path.join(nothing_dir,timestamp_event_id)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir,filename)
                restore_list.append(dst)
                print(bayer_file)
                print(dst)
                shutil.move(bayer_file,dst)
            elif key==ord('g'):
                dst_dir = os.path.join(something_dir,timestamp_event_id)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir,filename)
                restore_list.append(dst)
                print(dst)
                shutil.move(bayer_file,dst)
            elif key==ord('b'):
                if len(restore_list) > 0:
                    file = restore_list.pop()
                    filename = os.path.basename(file)
                    parent_dir = os.path.dirname(file)
                    print(parent_dir)
                    dst = os.path.join(temp_dir, filename)
                    shutil.move(file, dst)
                    if is_directory_empty(parent_dir):
                        os.rmdir(parent_dir)
                    bayer_list.append(dst)
                else:
                    break
            elif key==ord('q'):
                print('goodbye')
                print(bayer_file)
                break
            elif key==ord('d'):
                dst_dir = discard_dir
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir,filename)
                restore_list.append(dst)
                print(bayer_file)
                print(dst)
                shutil.move(bayer_file,dst)
            else:
                print('wtf, not a valid option, g = good, d = discard, b = back')
                break

        

    per_cam(cam0, "cam0", -1, discard_dir)
    per_cam(cam1, "cam1", 1, discard_dir)
    per_cam(cam2, "cam2", 1, discard_dir)