import tensorflow as tf
from PIL import Image
import numpy as np
import glob 


def representative_dataset(dataset_dir):
    files = glob.glob(dataset_dir + "/*.jpg")
    for file in files:
        image = Image.open(file)
        image = image.resize((224, 224))
        image = np.array(image) / 255.
        yield [image]


checkpoint_path = ""
model = tf.keras.models.load_model(checkpoint_path)
model.input.set_shape((1,) + model.input.shape[1:])
model.summary()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

tflite_both_quant_file = "/barcode_pipeline/barcode_detector/tflite/model.tflite"

with open(tflite_both_quant_file, 'wb') as f:
    f.write(tflite_model)