import os
import argparse
import numpy as np
import cv2

MAP = {(255, 255, 255): 1}
OTHERS = 0

# Credit: @heaversm at GitHub
# https://github.com/heaversm/deeplab-training/blob/master/models/research/deeplab/datasets/convert_rgb_to_index.py
def norm_mask(mask):
    normed = np.full((mask.shape[0], mask.shape[1]), OTHERS, dtype=np.uint8)
    for c, i in MAP.items():
        m = np.all(mask == np.array(c).reshape(1, 1, 3), axis=2)
        normed[m] = i
    return normed

def get_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.png')]
    files.sort()
    return files

def read_mask(mask_file):
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--masks" , help="Path to masks", required=True)
    parser.add_argument("-n", "--norm-masks", help="Path to normalized masks")
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    masks_path = args.masks
    norm_masks_path = args.norm_masks
    if not os.path.exists(norm_masks_path):
        os.mkdir(norm_masks_path)
    masks = get_files(masks_path)
    for mask_file in masks:
        print(f'[INFO] Processing {mask_file}...')
        mask = read_mask(os.path.join(masks_path, mask_file))
        normed = norm_mask(mask)
        cv2.imwrite(os.path.join(norm_masks_path, mask_file), normed)

if __name__ == '__main__':
    main()
