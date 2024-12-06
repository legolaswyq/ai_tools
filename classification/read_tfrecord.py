import tensorflow as tf
import os 
import glob

# Define a function to parse a single example from the TFRecord file

class TFrecordDataLoader:
    def __init__(self, datadir, batch_size=64, buffer_size=20000):
        self.datadir = datadir
        self.tfrecords = glob.glob(f"{datadir}/*/*.tfrecord")
        self.label_map = self.create_label_map()
        self.batch_size = batch_size
        self.buffer_size = buffer_size


    def create_label_map(self):
        barcodes = os.listdir(self.datadir)
        label_map = {barcode: index for index, barcode in enumerate(barcodes, 1)}
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
        return label 
    

    def parse_example(self, example):
        image = example['image/encoded']
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.py_function(self.parse_label, [example['image/class/text']], tf.int32)
        return image, label
    

    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords)
        dataset = dataset.map(self.parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    


datadir = "/home/walter/git/ai_tools/classification/tfrecord"
batch_size = 32
data_loader = TFrecordDataLoader(datadir, batch_size=batch_size)
dataset = data_loader.load_dataset()


# print(len(dataset))
# for images, labels in dataset:
    # print(images.shape)
    # print(labels.numpy())





# i = 0
# for example in dataset:
#     height = example['image/height'].numpy()
#     width = example['image/width'].numpy()
#     channels = example['image/channels'].numpy()
#     colorspace = example['image/colorspace'].numpy().decode('utf-8')
#     text = example['image/class/text'].numpy().decode('utf-8')
#     image_format = example['image/format'].numpy().decode('utf-8')
#     filename = example['image/filename'].numpy().decode('utf-8')
#     encoded_image = example['image/encoded'].numpy()

#     # Process the features as needed
#     # For example, you can decode the encoded image using tf.io.decode_jpeg
#     decoded_image = tf.io.decode_jpeg(encoded_image)
#     print(type(decoded_image))
#     print(decoded_image.shape)
#     print(height)
#     print(width)
#     print(channels)
#     print(colorspace)
#     print(text)
#     print(image_format)
#     print(filename)

# Create a TFRecordDataset
# file_paths_1 = "/home/walter/git/ai_tools/classification/tfrecord/038900012441/train-038900012441-00001-of-00001.tfrecord"
# file_paths_2 = "/home/walter/git/ai_tools/classification/tfrecord/038900012441/val-038900012441-00001-of-00001.tfrecord"
# file_paths = [file_paths_1, file_paths_2]
# dataset = tf.data.TFRecordDataset(file_paths)

# # Map the parsing function to the dataset
# parsed_dataset = dataset.map(parse_tfrecord_fn)

# Now, parsed_dataset is a dataset containing your parsed features.

# Optionally, you can batch and shuffle the dataset as needed
# batch_size = 32
# parsed_dataset = parsed_dataset.batch(batch_size)
# parsed_dataset = parsed_dataset.shuffle(buffer_size=10000)

# Iterate through the dataset (example)
# i = 0
# for example in parsed_dataset:
#     height = example['image/height'].numpy()
#     width = example['image/width'].numpy()
#     channels = example['image/channels'].numpy()
#     colorspace = example['image/colorspace'].numpy().decode('utf-8')
#     text = example['image/class/text'].numpy().decode('utf-8')
#     image_format = example['image/format'].numpy().decode('utf-8')
#     filename = example['image/filename'].numpy().decode('utf-8')
#     encoded_image = example['image/encoded'].numpy()

#     # Process the features as needed
#     # For example, you can decode the encoded image using tf.io.decode_jpeg
#     decoded_image = tf.io.decode_jpeg(encoded_image)
#     print(type(decoded_image))
#     print(decoded_image.shape)
#     print(height)
#     print(width)
#     print(channels)
#     print(colorspace)
#     print(text)
#     print(image_format)
#     print(filename)
#     print(i)
#     i = i + 1

# barcodes_txt = "/home/walter/git/ai_tools/classification/barcodes.txt"

# with open(barcodes_txt, 'r') as file:
#     barcodes = file.readlines()
#     barcodes = [barcode.strip() for barcode in barcodes]

# # Create a label map
# label_map = {barcode: index for index, barcode in enumerate(barcodes, 1)}

# # Print the label map
# print(label_map)