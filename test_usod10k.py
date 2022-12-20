"""
# > Script for evaluating UDepth on USOD10K dataset 
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
# local libs
from model.udepth import *
from utils.data import *
from utils.utils import *


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_img", type=str, default="./data/imgs/00002.png")
    parser.add_argument("--loc_mask", type=str, default="./data/masks/00002.png")
    parser.add_argument("--loc_img_folder", type=str, default="./data/imgs")
    parser.add_argument("--loc_mask_folder", type=str, default="./data/masks")
    parser.add_argument("--model_RMI_path", type=str, default="./saved_model/model_RMI.pth")
    parser.add_argument("--model_RGB_path", type=str, default="./saved_model/model_RGB.pth")
    parser.add_argument("--use_RMI", action='store_true')
    parser.add_argument("--data_type", type=str, default="folder", choices = ["image", "folder"])
    args = parser.parse_args()

    # Define input space
    image_space = "RMI" if args.use_RMI else "RGB"

    # Create output folder if not exist 
    output_folder = './data/output/usod10k/%s/' % (image_space)
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)

    # Use cuda 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data for image data type
    if args.data_type == "image":
        test_dataset = TestDataset(root_dir=args.loc_img, 
                                    mask_dir = args.loc_mask, 
                                    data_type = args.data_type, 
                                    transform=transforms.Compose([ToTestTensor()]), 
                                    use_RMI=args.use_RMI)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    # Load data for folder data type
    elif args.data_type == "folder":
        test_dataset = TestDataset(root_dir=args.loc_img_folder,
                                    mask_dir = args.loc_mask_folder, 
                                    data_type = args.data_type, 
                                    transform=transforms.Compose([ToTestTensor()]), 
                                    use_RMI=args.use_RMI)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    # Load specific model
    model_path = args.model_RMI_path if args.use_RMI else args.model_RGB_path
    net = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
    net.load_state_dict(torch.load(model_path))

    print("Model loaded: UDepth")

    net = net.to(device=device)
    net.eval()

    # Testing loop
    for i,sample_batched1 in enumerate (test_loader):
        # Prepare data
        img_fn = sample_batched1['file_name'][0]
        input_img = torch.autograd.Variable(sample_batched1['image'].to(device=device))
        mask = np.array(sample_batched1['mask']) / 255.0
        # Generate depth map
        _,out=net(input_img)
        # Apply guidedfilter to depth map
        result = output_result(out, mask[0])
        # Save result
        plt.imsave(os.path.join(output_folder, img_fn), result, cmap='inferno')

    print("Total images: {0}\n".format(len(test_loader)))
