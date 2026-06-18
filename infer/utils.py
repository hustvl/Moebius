from functools import partial
import os
from typing import List
from pathlib import Path
import math

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

def get_batch_infer_args(parser=None):
    
    if parser is None:
        import argparse
        parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    

    # model argument
    parser.add_argument("--model-config", type=str, required=False, default=None)
    parser.add_argument("--model-weight", type=str, required=False, default=None)

    # sampling argument
    parser.add_argument("--num-step", type=int, required=False, default=20)
    parser.add_argument("--cfg", type=float, required=False, default=2.5)
    parser.add_argument("--pst", type=str2bool, required=False, default=True)
    parser.add_argument("--cps", type=str2bool, required=False, default=False)
    parser.add_argument("--noise-offset", type=float, required=False, default=0.0357)
    parser.add_argument("--seed", type=int, default=0, required=False)


    # data argument
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=False)
    parser.add_argument("--resolution", type=int, default=512, required=False)

    # runtime argument
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--batch-size", type=int, required=False, default=32)
    parser.add_argument("--num-workers", type=int, required=False, default=64)

    # save argument
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--visualize-latent", action="store_true", default=False)

    return parser.parse_args()

def build_pipeline(args):
    from diffusers import DDIMScheduler
    from removal.v1_2.pipeline import RemovalSDXLPipeline_BatchMode as Removal_Pipeline
    from removal.v1_2 import build_removal_model, load_cfg, load_removal_model
    from utils_train import build_vae


    model_cfg = load_cfg(args.model_config)

    removal_model = build_removal_model(model_cfg, 20).to(args.device)
    print(load_removal_model(removal_model, args.model_weight,args.device))

    vae = build_vae(model_cfg).to(args.device)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        num_train_timesteps=1000, clip_sample=False)

    pipe = Removal_Pipeline(
        removal_model=removal_model, 
        vae=vae,
        scheduler=scheduler, 
        device=args.device, 
        dtype=torch.float)

    return pipe

class SAVER:
    @staticmethod
    def save_image(img, name, path):
        img.save(path / name)
        return name

    @staticmethod
    def save_images(images:List[Image.Image], names:List[str], save_root:str):
        assert len(images) == len(names), \
            f"images and names are not equal: {len(images)}!={len(names)}"
        
        pbar_save = tqdm(zip(images, names), total=len(names))

        cache_names = os.listdir(save_root)
        for image, name in pbar_save:
            if name not in cache_names:
                SAVER.save_image(image, name, save_root)

    @staticmethod
    def save_images_mt(images:List[Image.Image], names:List[str], save_root:str, num_workers=8):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(SAVER.save_image, image, name, save_root) for image, name in zip(images, names)]
            
            for future in tqdm(futures):
                future.result()
    
    @staticmethod
    def save_images_mp(images:List[Image.Image], names:List[str], save_root:str, num_workers=8):
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(SAVER.save_image, image, name, save_root) for image, name in zip(images, names)]
            
            for future in tqdm(futures):
                future.result()



