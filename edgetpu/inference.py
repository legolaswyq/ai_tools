from pycoral.utils import edgetpu
from pycoral.adapters import common
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


model = ""
interpreter = edgetpu.make_interpreter(model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_path = ""
img = Image.open(img_path)
# if you need resize could use common.set_resized_input(interpreter, size, resize)
# _, scale = common.set_resized_input(
#     interpreter, img.size, lambda size: img.resize(size, Image.LANCZOS))

# if input_details['dtype'] == np.uint8:
#     input_scale, input_zero_point = input_details["quantization"]
#     test_image = test_image / input_scale + input_zero_point

# if output_details['dtype'] == np.uint8:
#     output_scale, output_zero_point = output_details["quantization"]
#     output = interpreter.get_tensor(output_details[0]['index'])
#     output = (output - output_zero_point ) * output_scale

common.set_input(interpreter, img)

for _ in range(10):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    print('%.2f ms' % (inference_time * 1000))

output = common.output_tensor(interpreter, 0)
