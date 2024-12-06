from PIL import Image, ImageDraw
import glob 
import os
import re
import multiprocessing as mp
import argparse




def blackout_image(img_path, cam_pos, img_root, save_root):
    save_path = re.sub(img_root, save_root, img_path)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.basename(img_path)
    img = Image.open(img_path)
    cam_id = basename.split(".")[0].split("_")[int(cam_pos)]
    print(save_path)
    print(cam_id)

    if cam_id == "0":
        x1, y1 = 206, 250
        x2, y2 = 125,25
        x3, y3 = 15,324  
        x4, y4 = 324, 324
    elif cam_id == "1":
        x1, y1 = 324, 180
        x2, y2 = 0, 214
        x3, y3 = 0, 324  
        x4, y4 = 324, 324
    elif cam_id == "2":
        x1, y1 = 324, 215
        x2, y2 = 0, 183
        x3, y3 = 0, 324  
        x4, y4 = 324, 324

    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=255)

    blacked_out_img = Image.composite(Image.new('RGB', img.size, (0, 0, 0)), img, mask)
    blacked_out_img.save(save_path)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images_root', required=True, help='root path to the images')
    parser.add_argument('-s', '--save_root', required=True, help='root path to save the blackout images')
    parser.add_argument('-c', '--cam_pos', required=True, help='camera position index')
    parser.add_argument('-p', '--pattern', required=True, help='glob pattern')
    args = parser.parse_args()

    images_root = args.images_root
    save_root = args.save_root
    cam_pos = args.cam_pos
    pattern = args.pattern
    
    imgs = glob.glob(f"{images_root}{pattern}")

    print(f"Total number of images to be processed are: {len(imgs)}")

    CPU_CORES = mp.cpu_count()
    pool = mp.Pool(CPU_CORES-1 or 1)

    for img in imgs:
        pool.apply_async(blackout_image, args=(img, cam_pos, images_root, save_root))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()