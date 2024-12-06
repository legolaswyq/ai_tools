from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
import numpy as np
import time


def resize_crop_pad(img, is_label=False):
    """
    img -> PIL.Image
    return -> PIL.Image
    """
    target_width, target_height = 128, 128
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    if original_aspect_ratio > target_aspect_ratio:
        scale_factor = target_width / original_width
    else:
        scale_factor = target_height / original_height

    resized_image = img.resize((int(original_width * scale_factor), int(original_height * scale_factor)), Image.BILINEAR)
    if is_label:
        padded_image = Image.new("L", (target_width, target_height))
    else:
        padded_image = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    paste_x = (target_width - resized_image.width) // 2
    paste_y = (target_height - resized_image.height) // 2
    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def apply_single_channel_mask(image, mask):
    mask_3_channels = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.where(mask_3_channels, image, 0)
    return masked_image


def crop_masked_region(masked_image, mask):
    """crop out the mask region from original image 

    Args:
        masked_image (np.array): original image
        mask (np.array): single channel mask 

    Returns:
        np.array: crop
    """
    # Find bounding box of the mask
    non_zero_indices = np.argwhere(mask)
    min_y, min_x = np.min(non_zero_indices, axis=0)
    max_y, max_x = np.max(non_zero_indices, axis=0)
    # Crop out the region containing the mask
    cropped_masked_image = masked_image[min_y:max_y+1, min_x:max_x+1]
    return cropped_masked_image


def binarize_img_np(img_np):
    """
    img -> np.array
    return -> np.array
    """
    binary_img_array = np.where(img_np > 0, 1, 0)
    return binary_img_array.astype(np.uint8)


model = "/home/walter/git/barcode_pipeline/barcode_detector/tflite/train_with_maskrcnn_data.tflite"

interpreter = edgetpu.make_interpreter(model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




img_path = "/home/walter/nas_cv/walter_stuff/unet-dataset/jpgs/9300633320775/9300633320775_1708733394893_2_1708733410695_9604598.jpg"
img = Image.open(img_path)
resize_pad = resize_crop_pad(img)
resize_pad_np = np.array(resize_pad) / 255.
resize_pad_np = np.expand_dims(resize_pad_np, axis=0)



common.set_input(interpreter, resize_pad_np)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
mask = np.squeeze(output)
mask = binarize_img_np(mask)
masked_image = apply_single_channel_mask(resize_pad, mask)
crop = crop_masked_region(np.array(masked_image), mask)
crop_pil = Image.fromarray(crop)
# calculate time
# for _ in range(10):
#     start = time.perf_counter()
#     interpreter.invoke()
#     inference_time = time.perf_counter() - start
#     print('%.2f ms' % (inference_time * 1000))


crop_pil.save("crop.png")

# output = np.squeeze(output) * 255
# output = output.astype(np.uint8)
# output = Image.fromarray(output, "L")
# output.save("output.png")


img_path = "/home/walter/big_daddy/imagr_nnr_classifier_events_rgb_crop/021200001833/imagr-nnr-070324_mvp_021200001833_ce260627-9c39-3ce6-edac-6920243a59d3_0_1709777898619_6177779_116_146_47x42_crop.jpg"
img = Image.open(img_path)
resize_pad = resize_crop_pad(img)
resize_pad_np = np.array(resize_pad) / 255.
resize_pad_np = np.expand_dims(resize_pad_np, axis=0)



common.set_input(interpreter, resize_pad_np)
interpreter.invoke()

# output = common.output_tensor(interpreter, 0)
output = interpreter.get_tensor(output_details[0]['index'])
mask = np.squeeze(output)
mask = binarize_img_np(mask)
masked_image = apply_single_channel_mask(resize_pad, mask)
crop = crop_masked_region(np.array(masked_image), mask)
crop_pil = Image.fromarray(crop)
# calculate time
# for _ in range(10):
#     start = time.perf_counter()
#     interpreter.invoke()
#     inference_time = time.perf_counter() - start
#     print('%.2f ms' % (inference_time * 1000))


crop_pil.save("crop2.png")