import tensorflow as tf
import glob
import os

tf.config.experimental_run_functions_eagerly(True)


class TFrecordDataLoader:
    def __init__(self, datadir, batch_size=64, buffer_size=1000):
        self.datadir = datadir
        self.NUM_CLASSES = len(os.listdir(datadir))
        self.tfrecords = glob.glob(f"{datadir}/*/*.tfrecord")
        self.label_map = self.create_label_map()
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def create_label_map(self):
        barcodes = os.listdir(self.datadir)
        label_map = {barcode: index for index,
                     barcode in enumerate(barcodes, 1)}
        return label_map

    def parse_tfrecord_fn(self, example):
        feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.io.FixedLenFeature([], tf.string),
            'image/class/text': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            # Add more features as needed, specify their types and shapes
        }
        example = tf.io.parse_single_example(example, feature_description)
        return example

    def parse_label(self, barcode):
        label = self.label_map[barcode.numpy().decode('utf-8')]
        return tf.one_hot(label, self.NUM_CLASSES)


    def parse_example(self, example):
        image = example['image/encoded']
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (224, 224))  
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.py_function(
            self.parse_label, [example['image/class/text']], tf.float32)
        return image, label

    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(self.parse_tfrecord_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.parse_example,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


src_dir = "/home/walter/git/ai_tools/classification/tfrecord"
dataloader = TFrecordDataLoader(src_dir)
dataset = dataloader.load_dataset()

for images, labels in dataset:
    print(images.shape)
    print(labels.shape)
    print(type(images))
    print(type(labels))
    break


IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
print(NUM_CLASSES)

base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
#     base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(dataset, epochs=10)
