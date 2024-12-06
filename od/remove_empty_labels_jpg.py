import os 
import glob 
import re
import argparse
import tempfile
import shutil 

def get_img_path_by(label_path, labels_root, images_root):
    img_path = re.sub(labels_root, images_root, label_path)
    img_path = re.sub(".txt", ".jpg", img_path)
    if not os.path.exists(img_path):
        print("image path not exist")
    return img_path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images_root', required=True, help='root path to the images')
    parser.add_argument('-l', '--labels_root', required=True, help='root path to the labels')
    parser.add_argument('-p', '--pattern', required=True, help='glob pattern to glob all the labels file')
    args = parser.parse_args()


    labels_path = args.labels_root
    labels = glob.glob(f"{labels_path}{args.pattern}")
    print(f"total labels files : {len(labels)}")

    temp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created: {temp_dir}")
    tmp_labels = os.path.join(temp_dir, "labels")
    tmp_images = os.path.join(temp_dir, "images")
    os.makedirs(tmp_labels, exist_ok=True)
    os.makedirs(tmp_images, exist_ok=True)

    for label in labels:
        if os.path.getsize(label) == 0:
            jpg_path = get_img_path_by(label, args.labels_root, args.images_root)
            shutil.move(label, tmp_labels)
            shutil.move(jpg_path, tmp_images)
            
    
    labels = glob.glob(f"{labels_path}{args.pattern}")
    print(f"total labels files after remove: {len(labels)}")

if __name__ == '__main__':  
    main()
