from functools import partial
import os
from typing import List


from tqdm import tqdm
import numpy as np
import torch

from PIL import Image
from pathlib import Path

from .utils import get_batch_infer_args, build_pipeline, SAVER
from .utils_dataset import SimpleInferDataset, build_dataloader



def main():
    args = get_batch_infer_args()
    
    dataloader = build_dataloader(args, SimpleInferDataset)

    pipe = build_pipeline(args)
    pipe = partial(pipe, 
            guidance_scale=args.cfg,
            paste=args.pst, 
            compensate=args.cps, 
            num_steps=args.num_step,
            noise_offset=args.noise_offset
            )

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    pbar_loader = tqdm(enumerate(dataloader), 
        total=dataloader.dataset.__len__()//args.batch_size+1)

    for idx, (images, masks, inames) in pbar_loader:
        image_inpaint_list = pipe(images, masks)
        names = [iname+'.png' for iname in inames]
        SAVER.save_images_mp(image_inpaint_list, names, save_root)    


if __name__ == '__main__':
    main()