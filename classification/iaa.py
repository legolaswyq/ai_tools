import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_images_in_grid(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(6, 4))
    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].xaxis.set_major_locator(MaxNLocator(prune='both'))
        axes[i].yaxis.set_major_locator(MaxNLocator(prune='both'))

    plt.tight_layout()
    plt.show()


rows = 2
cols = 5



seq = iaa.Sequential([
    iaa.BlendAlphaSimplexNoise(
    foreground=iaa.EdgeDetect(1),
    per_channel=False
)
])

img_path = "/home/walter/nas_cv/walter_stuff/raw_data/crops/029470000105/0/imagr-051023_dylan_029470000105_0_1696479070902_445932.jpg"
img = Image.open(img_path)
imgs = []
for i in range(5):
    imgs.append(np.array(img))
augmented_imgs = seq.augment_images(imgs)


for aug in augmented_imgs:
    imgs.append(Image.fromarray(aug))

plot_images_in_grid(imgs, rows, cols)
# plot_images_in_grid(aug_imgs, rows, cols)