import random 
import glob 
import os 
import pprint

def _find_pair(find_pair, time_seq, cam_id_files):
    for file in cam_id_files:
        infos = os.path.basename(file).split("_")
        seq = int(infos[-1].split(".")[0])
        if seq >= time_seq:
            find_pair.append(file)
            return find_pair

    return find_pair


def find_pairs_3_cameras(pairs, cam_anchor, cam_target_1, cam_target_2, max_pairs=200):
    for file in cam_anchor:
        find_pair = [file]
        infos = os.path.basename(file).split("_")
        time_seq = int(infos[-1].split(".")[0])

        find_pair = _find_pair(find_pair, time_seq, cam_target_1)
        find_pair = _find_pair(find_pair, time_seq, cam_target_2)
        
        if len(pairs) >= max_pairs:
            break

        pairs.append(find_pair)

    return pairs


def find_pairs_2_cameras(pairs, cam_anchor, cam_target, max_pairs=200):
    for file in cam_anchor:
        find_pair = [file]
        infos = os.path.basename(file).split("_")
        time_seq = int(infos[-1].split(".")[0])

        find_pair = _find_pair(find_pair, time_seq, cam_target)
        
        if len(pairs) >= max_pairs:
            break

        pairs.append(find_pair)

    return pairs



def prepare_train_val_dataset(crop_dir, barcode):
    cameras = {str(i): [] for i in range(9)}
    train_pair = []
    val_pair = []

    for file in glob.glob(f"{crop_dir}/{barcode}/**/*.jpg"):
        infos = os.path.basename(file).split("_")
        cam_id = infos[3]
        cameras[cam_id].append(file)

    for i in range(9):
        cameras[str(i)] = sorted(cameras[str(i)])

    # get training pair 
    three_pair = []
    three_pair = find_pairs_3_cameras(three_pair, cameras["1"], cameras["4"], cameras["8"], max_pairs=200)
    three_pair = find_pairs_3_cameras(three_pair, cameras["4"], cameras["8"], cameras["1"], max_pairs=200)
    three_pair = find_pairs_3_cameras(three_pair, cameras["8"], cameras["1"], cameras["4"], max_pairs=200)
    three_pair = find_pairs_3_cameras(three_pair, cameras["2"], cameras["6"], cameras["7"], max_pairs=400)
    three_pair = find_pairs_3_cameras(three_pair, cameras["6"], cameras["7"], cameras["2"], max_pairs=400)
    three_pair = find_pairs_3_cameras(three_pair, cameras["7"], cameras["2"], cameras["6"], max_pairs=400)
    three_pair = random.choices(three_pair, k=400)
    

    two_pair = []
    two_pair = find_pairs_2_cameras(two_pair, cameras["4"], cameras["8"], max_pairs=600)
    two_pair = find_pairs_2_cameras(two_pair, cameras["4"], cameras["1"], max_pairs=600)
    two_pair = find_pairs_2_cameras(two_pair, cameras["8"], cameras["1"], max_pairs=600)
    two_pair = find_pairs_2_cameras(two_pair, cameras["6"], cameras["7"], max_pairs=800)
    two_pair = find_pairs_2_cameras(two_pair, cameras["7"], cameras["2"], max_pairs=800)
    two_pair = find_pairs_2_cameras(two_pair, cameras["6"], cameras["2"], max_pairs=600)
    two_pair = random.choices(two_pair, k=400)
    
    train_pair.extend(three_pair)
    train_pair.extend(two_pair)
    
    
    # get val pair 
    three_pair = []
    three_pair = find_pairs_3_cameras(three_pair, cameras["0"], cameras["3"], cameras["5"], max_pairs=80)
    three_pair = find_pairs_3_cameras(three_pair, cameras["3"], cameras["5"], cameras["0"], max_pairs=80)
    three_pair = find_pairs_3_cameras(three_pair, cameras["5"], cameras["3"], cameras["0"], max_pairs=80)
    three_pair = random.choices(three_pair, k=80)

    two_pair = []
    two_pair = find_pairs_2_cameras(two_pair, cameras["3"], cameras["5"], max_pairs=160)
    two_pair = find_pairs_2_cameras(two_pair, cameras["3"], cameras["0"], max_pairs=160)
    two_pair = find_pairs_2_cameras(two_pair, cameras["5"], cameras["0"], max_pairs=160)
    two_pair = random.choices(two_pair, k=80)
    
    val_pair.extend(three_pair)
    val_pair.extend(two_pair)

    assert len(train_pair) == 800
    assert len(val_pair) == 160 

    return train_pair, val_pair


crop_dir = "/home/walter/big_daddy/walter_stuff/onboard/crops"
barcode = "9420039513288"

prepare_train_val_dataset(crop_dir, barcode)