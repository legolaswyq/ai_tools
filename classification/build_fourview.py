from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
import os 
import glob 
import random
import multiprocessing as mp
import argparse
import cv2


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
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-90, 90),
            shear=(-8, 8)
        ),
    ], random_order=True)

    return seq


def form_fourview(images, seq, image_size=300):
    imgs_np = []
    for image in images:
        img = Image.open(image)
        bgr_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(bgr_image, (int(image_size/2) , int(image_size/2)), interpolation=cv2.INTER_CUBIC)
        rgb_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # img = img.resize((int(image_size/2) , int(image_size/2)))
        img_np = np.array(rgb_image)
        imgs_np.append(img_np)
    
    if seq:
        imgs_np.extend(seq.augment_images(imgs_np))
    imgs = random.sample(imgs_np, k=4)

    fourview_np = np.vstack(
        [np.hstack([imgs[0], imgs[1]]),
        np.hstack([imgs[2], imgs[3]])]
    )
    
    return Image.fromarray(fourview_np)


def find_train_val_files(crop_dir, barcode):
    files = glob.glob(f"{crop_dir}/{barcode}/**/*.jpg")

    train = []
    val = []
    # num_crop = len(files)
    # random.shuffle(files)
    # split = int(num_crop * 0.8)
    # train = files[:split]
    # val = files[split:]
    for file in files:
        name = os.path.basename(file)
        cam_id = name.split("_")[3]
        if cam_id == "0" or cam_id == "3" or cam_id == "5":
            val.append(file)
        else:
            train.append(file)

    
    return train, val


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



def per_barcode(crop_dir, fourview_out, barcode, seq, train_num, val_num, is_shuffle_patchify, augmented_val):

    train, val = find_train_val_files(crop_dir, barcode)
    
    for i in range(train_num):
        imgs = random.sample(train, 4)
        fourview_pil = form_fourview(imgs, seq)
        if is_shuffle_patchify:
            fourview_pil = shuffle_patchify(fourview_pil)
        savedir = os.path.join(fourview_out, barcode)
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, f"train_{i}.jpg")
        fourview_pil.save(dst)


    for i in range(val_num):
        imgs = random.sample(val, 4)
        if augmented_val:
            fourview_pil = form_fourview(imgs, seq)
        else:
            fourview_pil = form_fourview(imgs, None)

        if is_shuffle_patchify:
            fourview_pil = shuffle_patchify(fourview_pil)
        savedir = os.path.join(fourview_out, barcode)
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, f"val_{i}.jpg")
        fourview_pil.save(dst)


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
    seq = img_augmentation_sequence()
    barcodes = os.listdir(crop_in)

    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)

    for barcode in barcodes:
        print(f"processing barcode: {barcode}")
        pool.apply_async(per_barcode, args=(crop_in, fourview_out, barcode, seq, train_num, val_num, is_shuffle_patchify, augmented_val))

    pool.close()
    pool.join()


if __name__=="__main__":
    main()
