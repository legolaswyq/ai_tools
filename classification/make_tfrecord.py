import tensorflow as tf 
import glob 
import os 
import tensorflow as tf
from PIL import Image
import multiprocessing as mp
import argparse

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_to_example(barcode, image_path):

    width, height = Image.open(image_path).size
    img = open(image_path, 'rb').read()
    filename = os.path.basename(image_path).encode('utf-8')
    text = barcode.encode('utf-8')
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    example =   tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image/height':     _int64_feature(height),
                            'image/width':      _int64_feature(width),
                            'image/channels':   _int64_feature(channels),
                            'image/colorspace': _bytes_feature(colorspace),
                            'image/class/text': _bytes_feature(text),
                            'image/format':     _bytes_feature(image_format),
                            'image/filename':   _bytes_feature(filename),
                            'image/encoded':    _bytes_feature(img)
                        }
                    )
                )
    
    return example


def process_per_barcode(datadir, barcode, outdir):
    out = os.path.join(outdir, barcode)
    os.makedirs(out, exist_ok=True)

    train_images_path = glob.glob(f"{datadir}/{barcode}/train*")
    with tf.io.TFRecordWriter(f"{out}/train-{barcode}-00001-of-00001.tfrecord") as writer:
        for image_path in train_images_path:
            tf_example = convert_to_example(barcode, image_path)
            writer.write(tf_example.SerializeToString())


    val_images_path = glob.glob(f"{datadir}/{barcode}/val*")
    with tf.io.TFRecordWriter(f"{out}/val-{barcode}-00001-of-00001.tfrecord") as writer:
        for image_path in val_images_path:
            tf_example = convert_to_example(barcode, image_path)
            writer.write(tf_example.SerializeToString())



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datadir', required=True, help='datadir')
    parser.add_argument('-o', '--outdir', required=True, help='outdir')
    args = parser.parse_args()

    
    datadir = args.datadir
    outdir = args.outdir
    barcodes = os.listdir(datadir)


    MAX_NUM_CORES = mp.cpu_count()
    pool = mp.Pool(MAX_NUM_CORES-1 or 1)
    print(pool)

    for barcode in barcodes:
        print(f"processing barcode {barcode}")
        pool.apply_async(process_per_barcode, args=(datadir, barcode, outdir))

    pool.close()
    pool.join()


if __name__=="__main__":
    main()