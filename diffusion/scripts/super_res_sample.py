"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import sys
sys.path.append('D:/JH/DisC-Diff/guided_diffusion')

import os

import random
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import argparse
import yaml
import torch.distributed as dist
import statistics
from torch.utils.data import DataLoader
from image_datasets import load_data
import dist_util, logger
from script_util import (
    sr_create_model_and_diffusion,
    get_psnr,
    get_ssim,
    add_dict_to_argparser
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    print(args)
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(args)
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    
    # data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    data = load_superres_data(
        args.hr_data_dir,
        args.lr_data_dir,
        args.other_data_dir,
        args.batch_size,
    )

    logger.log("creating samples...")

    psnr_list, ssim_list = [], []
    sample_storage = []
     
    for i in range(args.num_samples // args.batch_size): 
        hr, model_kwargs = next(data)
        hr = hr.permute(0, 2, 3, 1)
        hr = hr.contiguous()
        hr = hr.cpu().numpy()

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # Choose the appropriate sample function based on the sampling method
        sample_fn = diffusion.ddim_sample_loop if args.sampling_method == 'ddim' else \
            (diffusion.dpm_solver_sample_loop if args.sampling_method == 'dpm++' else diffusion.p_sample_loop)

        sample = sample_fn(
            model,
            (args.batch_size, args.in_channel, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        for j in range(hr.shape[0]):
            psnr_list.append(get_psnr(hr[j, ...], sample[j, ...]))
            ssim_list.append(get_ssim(hr[j, ...], sample[j, ...]))
            if len(sample_storage) < 5:
                sample_storage.append((hr[j, ...], sample[j, ...]))

        print(f'Number of evaluated slices: {len(psnr_list)}')
        print(f'Mean PSNR: {statistics.mean(psnr_list)}')
        print(f'Mean SSIM: {statistics.mean(ssim_list)}')

        # Save the images
        save_images(sample_storage, args.save_dir)

        dist.barrier()
        logger.log("sampling complete")


    dist.barrier()
    logger.log("sampling complete")

def load_superres_data(hr_data_dir, lr_data_dir, other_data_dir, batch_size):
    # Load the super-resolution dataset using the provided directories
    dataset = load_data(
        hr_data_dir=hr_data_dir,
        lr_data_dir=lr_data_dir,
        other_data_dir=other_data_dir
    )
    
    # Create a data loader to load the dataset in batches
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    
    # Iterate over the data loader and yield high-resolution MRIs and model keyword arguments
    for hr_data, lr_data, other_data in loader:
        model_kwargs = {"low_res": lr_data, "other": other_data}
        yield hr_data, model_kwargs

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        
    add_dict_to_argparser(parser, config)
    return parser

def save_images(samples, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (hr_image, sr_image) in enumerate(samples):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(hr_image, cmap='gray')
        axes[0].set_title('High Resolution')
        axes[0].axis('off')

        axes[1].imshow(sr_image, cmap='gray')
        axes[1].set_title('Super Resolution')
        axes[1].axis('off')

        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
        plt.close()


if __name__ == "__main__":
    main()
