import os 
import zipfile
import shutil
import argparse
import glob

def unzip_to_one_folder(src_zipfile, dst):
    dst = os.path.join(dst, "out")
    zip_file = zipfile.ZipFile(src_zipfile, "r")
    zip_file.extractall(dst)

    obj_folder = os.path.join(dst, "obj_train_data")
    if os.path.exists(obj_folder):
        for root, dirs, files in os.walk(obj_folder):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    shutil.move(file_path, dst)
                    print(file_path)


    shutil.rmtree(obj_folder)
    os.remove(os.path.join(dst, "obj.data"))
    os.remove(os.path.join(dst, "obj.names"))
    os.remove(os.path.join(dst, "train.txt"))

    




def move_and_merge(src, dst):
    if not os.path.isdir(src):
        return 
    
    subfolders = os.listdir(src)
    for subfolder in subfolders:
        subfolder_path = os.path.join(src, subfolder)
        subfolder_dst = os.path.join(dst, subfolder)
        if os.path.exists(subfolder_dst):
            move_and_merge(subfolder_path, subfolder_dst)
        else:
            shutil.move(subfolder_path, subfolder_dst)


def unzip_file_keep_folder_structure(src_zipfile, dst):
    zip_file = zipfile.ZipFile(src_zipfile, "r")
    zip_file.extractall(dst)

    obj_folder = os.path.join(dst, "obj_train_data")
    move_and_merge(obj_folder, dst)

    shutil.rmtree(obj_folder)
    os.remove(os.path.join(dst, "obj.data"))
    os.remove(os.path.join(dst, "obj.names"))
    os.remove(os.path.join(dst, "train.txt"))

    label_out = os.path.join(dst, "labels")
    os.makedirs(label_out, exist_ok=True)
    image_out = os.path.join(dst, "images")
    os.makedirs(image_out, exist_ok=True)

    labels = glob.glob(f"{dst}/**/*.txt")
    images = glob.glob(f"{dst}/**/*.jpg")
    
    for label in labels:
        label_name = os.path.basename(label)
        label_dst = os.path.join(label_out, label_name)
        shutil.move(label, label_dst)
        
    for image in images:
        image_name = os.path.basename(image)
        image_dst = os.path.join(image_out, image_name)
        shutil.move(image, image_dst)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root', required=True, help='root path to the source bayers')
    parser.add_argument('-o', '--output', required=True, help='input dir')
    parser.add_argument('-f', '--folder', action="store_true", help='put all txt file in one folder')
    parser.add_argument('-s', '--split', action="store_true", help='split txt and jpg')
    args = parser.parse_args()

    root = args.root
    out = args.output
    os.makedirs(out, exist_ok=True)

    zipfiles = glob.glob(f"{root}/*.zip")
    for zipfile in zipfiles:
        if args.folder:
            unzip_to_one_folder(zipfile, out)
        else:
            unzip_file_keep_folder_structure(zipfile, out)



if __name__ == '__main__':  
    main()