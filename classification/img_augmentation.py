import os 
from PIL import Image
import imgaug.augmenters as iaa
import imageio

src_dir = "/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/stack_green/train/cam0/008362.jpg"
# img = Image.open(src_dir)
# img.show()
img = imageio.imread(src_dir)
rotate = iaa.Affine(rotate=(90))
image_aug = rotate(image=img)

img = Image.fromarray(image_aug)
img.show()

image_aug_2 = rotate(image_aug)
img = Image.fromarray(image_aug_2)
img.show()

# print(type(new_img))
# new_img_pil = Image.fromarray(new_img)
# new_img_pil.show()