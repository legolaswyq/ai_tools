from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import os 
import glob 
import random
import multiprocessing as mp
import argparse
import cv2
import math


def rotate_image(array, angle):
    height, width = array.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def patchify_per_barcode(src_dir, barcode, shuffle_patchify_dir):
    files = glob.glob(f"{src_dir}/train*.jpg")
    os.makedirs(os.path.join(shuffle_patchify_dir, barcode), exist_ok=True)
    for file in files:
        img = Image.open(file)
        pachify = shuffle_patchify(img)
        basename = os.path.basename(file)
        dst = os.path.join(os.path.join(shuffle_patchify_dir, barcode), basename)
        pachify.save(dst)


def img_augmentation_sequence():
    """
    usage: 
    seq.augment_images(imgs)
    imgs is a list of image[np.array]
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(
            0.7,
            iaa.Crop(percent=(0.1,0.3)),
        ),
    ], random_order=True)

    return seq


def load_images(images):
    imgs_np = []
    for image in images:
        if os.path.exists(image):
            img = Image.open(image)
            img_np = np.array(img)
        else:  
            img_np = np.zeros((324, 324, 3), dtype=np.uint8)
        imgs_np.append(img_np)

    # if len(imgs_np) <= 2:
    #     imgs_np = random.choices(imgs_np, k=3)

    return imgs_np


def imgs_augmentation(imgs):
    augmented_imgs = []
    seq = img_augmentation_sequence()
    for img in imgs:
        if random.random() > 0.3:
            random_degree = random.choice([15, 30, 45, 60, 75])
            img = rotate_image(img, random_degree)
        img = seq.augment_image(img)
        augmented_imgs.append(img)
    
    return augmented_imgs


def resize(imgs, img_size=300):
    # expect to be rgb np array 
    resized_imgs = []
    for img in imgs:
        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(bgr_image, (int(img_size/2) , int(img_size/2)), interpolation=cv2.INTER_CUBIC)
        rgb_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img_np = np.array(rgb_image)
        resized_imgs.append(img_np)

    return resized_imgs


def form_fourview(imgs):
    # expect images to be np.array order in rgb channel
    assert len(imgs) == 3
    threeview_np = np.hstack([imgs[0], imgs[1], imgs[2]])
    return Image.fromarray(threeview_np)


def save_fourview(fourview, fourview_out, barcode, save_name):
    savedir = os.path.join(fourview_out, barcode)
    os.makedirs(savedir, exist_ok=True)
    dst = os.path.join(savedir, save_name)
    fourview.save(dst)


def _find_pair(find_pair, time_seq, cam_id_files):
    found_match = False
    for file in cam_id_files:
        infos = os.path.basename(file).split("_")
        seq = int(infos[-1].split(".")[0])
        if seq == time_seq:
            find_pair.append(file)
            found_match = True
            return find_pair, found_match
        
    find_pair.append("some_fake_path")
    return find_pair, found_match


def find_pairs_3_cameras(pairs, cam_0, cam_1, cam_2, max_pairs=2000):
    
    for i in range(150):
        counter = 0 
        
        find_pair = []
        find_pair, found_match = _find_pair(find_pair, i, cam_0)
        
        if found_match:
            counter += 1
        find_pair, found_match = _find_pair(find_pair, i, cam_1)
        
        if found_match:
            counter += 1
        find_pair, found_match = _find_pair(find_pair, i, cam_2)
        
        if found_match:
            counter += 1

        if counter >= 1:
            pairs.append(find_pair)
        
    return pairs


def find_pairs_2_cameras(pairs, cam_anchor, cam_target, max_pairs=200):
    for file in cam_anchor:
        find_pair = [file]
        infos = os.path.basename(file).split("_")
        time_seq = int(infos[-1].split(".")[0])

        find_pair = _find_pair(find_pair, time_seq, cam_target)
        
        if len(pairs) >= max_pairs:
            break

        pairs.append(find_pair)

    return pairs


def prepare_train_val_dataset(crop_dir, barcode, train_num, val_num):
    cameras = {str(i): [] for i in range(9)}
    train_pair = []
    val_pair = []

    for file in glob.glob(f"{crop_dir}/{barcode}/**/*.jpg"):
        infos = os.path.basename(file).split("_")
        cam_id = infos[3]
        cameras[cam_id].append(file)

    for i in range(9):
        cameras[str(i)] = sorted(cameras[str(i)])

    # get training pair 
    three_pair = []
    three_pair = find_pairs_3_cameras(three_pair, cameras["4"], cameras["8"], cameras["1"])
    three_pair = find_pairs_3_cameras(three_pair, cameras["2"], cameras["6"], cameras["7"])
    train_pair.extend(three_pair)
    
    
    # get val pair 
    three_pair = []
    three_pair = find_pairs_3_cameras(three_pair, cameras["0"], cameras["3"], cameras["5"])
    val_pair.extend(three_pair)
    return train_pair, val_pair


def shuffle_patchify(img, num_patches=8):
    img_np = np.array(img)

    # Get the dimensions of the image
    height, width, channels = img_np.shape

    # Calculate the size of each patch
    patch_height = height // num_patches
    patch_width = width // num_patches

    # Create an array to store the patches
    patches = []

    # Extract patches from the image
    for i in range(num_patches):
        for j in range(num_patches):
            patch = img_np[i * patch_height:(i + 1) * patch_height,
                            j * patch_width:(j + 1) * patch_width, :]
            patches.append(patch)

    # Shuffle the patches
    np.random.shuffle(patches)

    # Create a mosaic by concatenating the shuffled patches
    mosaic = np.concatenate([np.concatenate(patches[j:j + num_patches], axis=1) for j in range(0, len(patches), num_patches)], axis=0)

    # Convert the NumPy array back to a PIL image
    mosaic_pil = Image.fromarray(mosaic)
    return mosaic_pil


def per_barcode(crop_dir, fourview_out, barcode, train_num, val_num, is_shuffle_patchify, augmented_val):
    train_pair, val_pair = prepare_train_val_dataset(crop_dir, barcode, train_num, val_num)

    for i in range(len(train_pair)):
        # with open(f"/home/walter/git/ai_tools/log/{barcode}log.txt", 'a') as f:
        #     seq1 = os.path.basename(train_pair[i][0]).split("_")[3] + ' ' + os.path.basename(train_pair[i][0]).split("_")[-1]
        #     seq2 = os.path.basename(train_pair[i][1]).split("_")[3] + ' ' + os.path.basename(train_pair[i][1]).split("_")[-1]
        #     seq3 = os.path.basename(train_pair[i][2]).split("_")[3] + ' ' + os.path.basename(train_pair[i][2]).split("_")[-1]
        #     f.write(f"train_{i}.jpg {seq1} {seq2} {seq3}\n")
        imgs = load_images(train_pair[i])
        # imgs = imgs_augmentation(imgs)
        imgs = resize(imgs)
        fourview = form_fourview(imgs)
        save_fourview(fourview, fourview_out, barcode, f"train_{i}.jpg")

    for i in range(len(val_pair)):
        imgs = load_images(val_pair[i])
        if augmented_val:
            imgs = imgs_augmentation(imgs)
        imgs = resize(imgs)
        fourview = form_fourview(imgs)
        save_fourview(fourview, fourview_out, barcode, f"val_{i}.jpg")



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--crop_in', required=True, help='crop_dir')
    parser.add_argument('-f', '--fourview_out', required=True, help='fourview_saveDir')
    parser.add_argument('-t', '--train_num', default=400, help='num of train sample')
    parser.add_argument('-v', '--val_num', default=80, help='num of val sample')
    parser.add_argument('-s', '--is_shuffle_patchify', action='store_true', default=False, help='shuffle_patchify')
    parser.add_argument('-a', '--augmented_val', action='store_true', default=False, help='whether to augmented validation')
    args = parser.parse_args()

    crop_in = args.crop_in
    fourview_out = args.fourview_out
    os.makedirs(fourview_out, exist_ok=True)
    train_num = int(args.train_num)
    val_num = int(args.val_num)
    is_shuffle_patchify = args.is_shuffle_patchify
    augmented_val = args.augmented_val
    barcodes = os.listdir(crop_in)

    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    for barcode in barcodes:
        # per_barcode(crop_in, fourview_out, barcode, train_num, val_num, is_shuffle_patchify, augmented_val)
        print(f"processing barcode: {barcode}")
        pool.apply_async(per_barcode, args=(crop_in, fourview_out, barcode, train_num, val_num, is_shuffle_patchify, augmented_val))

    pool.close()
    pool.join()


if __name__=="__main__":
    main()
