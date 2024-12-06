import os 
import glob 
import argparse
import re

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='fourview dir')
    parser.add_argument('-o', '--output', required=True, help='train val save dir')
    parser.add_argument('-b', '--bucket', required=True, help='google cloud bucket path')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    bucket_path = args.bucket

    trains  = glob.glob(f"{input_dir}/*/train*")
    vals  = glob.glob(f"{input_dir}/*/val*")

    train_output_path = os.path.join(output_dir, "train.txt")
    val_output_path = os.path.join(output_dir, "val.txt")

    with open(train_output_path, 'w') as f:
        for train in trains:
            if bucket_path:
                train = re.sub(input_dir, bucket_path, train)
            f.write(f"{train}\n")

    with open(val_output_path, 'w') as f:
        for val in vals:
            if bucket_path:
                val = re.sub(input_dir, bucket_path, val)
            f.write(f"{val}\n")


if __name__=="__main__":
    main()