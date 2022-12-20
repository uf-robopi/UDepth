"""
# > Script for evaluating domain projection on USOD10K dataset 
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import os 
import numpy as np
import argparse
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
# local libs
from utils.utils import gf


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_img", type=str, default="./data/imgs/00002.png")
    parser.add_argument("--loc_mask", type=str, default="./data/imgs/00002.png")
    parser.add_argument("--loc_img_folder", type=str, default="./data/imgs")
    parser.add_argument("--loc_mask_folder", type=str, default="./data/masks")
    parser.add_argument("--data_type", type=str, default="folder", choices = ["image", "folder"])
    args = parser.parse_args()

    # Create output folder if not exist 
    output_folder = './data/output/usod10k/'
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    # Calculate the amount of input data
    input_count = len(os.listdir(args.loc_img_folder)) if args.data_type == "folder" else 1

    # Testing loop
    for idx in range(input_count):
        # Get image path
        img_fn = os.listdir(args.loc_img_folder)[idx].replace(".","_proj.") if args.data_type == "folder" else args.loc_img.split("/")[-1].replace(".","_proj.")
        img_path = os.path.join(args.loc_img_folder, os.listdir(args.loc_img_folder)[idx]) if args.data_type == "folder" else args.loc_img
        
        # Load image
        image = (Image.open(img_path))

        # Compute R, M, I channel 
        r, g, b = image.split()
        r = np.array(r) / 255.0
        gb_max = np.maximum.reduce([np.array(g), np.array(b)]) / 255.0
        gray_c = np.array(ImageOps.grayscale(image))
        # Prepare SOD mask
        mask_fn = os.path.join(args.loc_mask_folder, os.listdir(args.loc_img_folder)[idx]) if args.data_type == "folder" else args.loc_mask
        mask = (Image.open(mask_fn)).resize((r.shape[1],r.shape[0]))
        mask_norm = np.float32(np.array(mask) / 255.0)

        # Define coefficient of constant, R, M terms
        u0 = 0.46353632 # Constant
        u1 = 0.49598983 # R
        u2 = -0.38875134 # M

        # Generate domain projection and apply the guidedfilter
        np_1 = np.ones(r.shape)
        depth = np.float32( u0 * np_1 + u1 * r + u2 * gb_max )
        result = gf(mask_norm, depth) * 255

        # Save result
        plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')

    print("Total images: {0}\n".format(input_count))
