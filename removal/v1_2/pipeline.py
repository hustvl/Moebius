import os
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image as I
from PIL import ImageFilter as IFilter
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torchvision.transforms as T
import random
from collections import UserDict
from typing import List, Union
from types import SimpleNamespace
from functools import partial

from diffusers.models import AutoencoderKL
from diffusers import DDIMScheduler

from .removal_model import RemovalModel
from .compensation_utils import paste_compensate
from utils_infer import encode_clean_latents, predict_noise



def mask_dilate(mask, kernel_size):
    if type(mask) != np.ndarray:
        mask = np.array(mask)
    if kernel_size != 0:
        mask = cv2.dilate(mask,
                          np.ones((kernel_size, kernel_size), np.uint8),
                          iterations=1)
    return mask


def mask_morphologyEx(mask, kernel_size):
    if type(mask) != np.ndarray:
        mask = np.array(mask)
    if kernel_size != 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((kernel_size, kernel_size), np.uint8),
                                iterations=1)
    return mask


def get_timesteps(scheduler, num_inference_steps, strength, device):
    init_timestep = min(
        int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    return timesteps, num_inference_steps - t_start


def resize_image_to_multiple_of_64(imgs: Union[I.Image, List[I.Image]], image_size: int) -> List[I.Image]:
    """
    Resize input images so the short side equals image_size,
    then round width/height to multiples of 64.
    """
    if isinstance(imgs, I.Image):
        imgs = [imgs]
    w, h = imgs[0].size

    if w < h:
        scale = image_size / w
        w_t, h_t = image_size, int(h * scale)
    else:
        scale = image_size / h
        w_t, h_t = int(w * scale), image_size

    w_t, h_t = w_t // 64 * 64, h_t // 64 * 64

    out = []
    for img in imgs:
        out.append(img.resize((w_t, h_t), I.Resampling.LANCZOS))
    return out


def visualize_latent_steps(
    latent_list,
    save_path=".",
    mode="horizontal",
    normalize_each=True,
    dpi=100,
    colormap='plasma'
):
    """Visualize latent denoising steps as a grid image."""
    import matplotlib.pyplot as plt
    print("do latents visualization")
    assert mode in ["horizontal", "vertical"]
    num_steps = len(latent_list)
    num_channels = latent_list[0].shape[1]

    fig_w, fig_h = (num_steps * 3, num_channels * 3) if mode == "horizontal" else (num_channels * 3, num_steps * 3)
    fig, axes = plt.subplots(
        num_channels, num_steps if mode == "horizontal" else num_steps,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    if num_steps == 1:
        axes = [axes]

    for step_idx, latent in enumerate(latent_list):
        latent = latent[0]
        for ch in range(num_channels):
            ax = axes[ch, step_idx] if mode == "horizontal" else axes[step_idx, ch]
            channel_data = latent[ch]

            if normalize_each:
                vmin, vmax = channel_data.min(), channel_data.max()
                channel_data = (channel_data - vmin) / (vmax - vmin + 1e-5)
            else:
                channel_data = torch.clamp(channel_data, 0, 1)
            channel_data = channel_data.cpu().numpy()

            ax.imshow(channel_data, cmap=colormap)
            ax.text(0.5, -0.05, f"Step {num_steps - step_idx} - C{ch}",
                    transform=ax.transAxes, ha='center', va='top', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "latents.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved latent visualization to {save_path}")


class RemovalSDXLPipeline_BatchMode:
    def __init__(self,
                 removal_model: RemovalModel,
                 vae: AutoencoderKL,
                 scheduler: DDIMScheduler,
                 device='cuda',
                 dtype=torch.float16):
        # VAE
        self.vae = vae
        self.vae.to(device=device, dtype=dtype)
        self.vae.eval()

        # UNet removal model
        self.removal_model = removal_model
        self.removal_model.to(device=device, dtype=dtype)
        self.removal_model.eval()

        # Scheduler
        self.noise_scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_ds_ratio = self.vae_scale_factor
        self.device = device
        self.dtype = dtype

        # Build embedding input_ids for CFG (unconditional + conditional)
        half_id_num = self.removal_model.num_embeddings // 2
        id_num = self.removal_model.num_embeddings
        print(f'[Info]: removal_model.num_embeddings: {id_num}, half: {half_id_num}')

        input_ids = torch.tensor([list(range(half_id_num))], dtype=torch.int64, device=self.device, requires_grad=False)
        un_input_ids = torch.tensor([list(range(half_id_num, id_num))], dtype=torch.int64, device=self.device, requires_grad=False)
        self.input_ids = torch.cat([un_input_ids, input_ids]).to(device=self.device)

        print('[Info]: Load pipeline succeeded.')

    @staticmethod
    def mask_preprocess(mask_image, mask_dilate_kernel_size, kind='dilate'):
        assert kind in ('dilate', 'morphologyEx')
        if kind == 'dilate':
            mask = mask_dilate(mask_image, mask_dilate_kernel_size)
        elif kind == 'morphologyEx':
            mask = mask_morphologyEx(mask_image, mask_dilate_kernel_size)
        return I.fromarray(mask)

    def prepare_mask_and_masked_image(self, image, mask):
        image = image.float()
        mask = torch.where(mask >= 0.5, 1, 0).unsqueeze(0).to(device=self.device)
        masked_image = image * (1 - mask)
        masked_image = masked_image.to(dtype=self.dtype, device=self.device)
        return mask.to(dtype=torch.uint8), masked_image

    def _preprocess(self,
                    input_image: I.Image,
                    input_mask: I.Image,
                    image_size: int,
                    mask_dilate_kernel_size=0,
                    mask_preprocess_type='dilate',
                    enable_migan=False,
                    migan_model=None) -> List[Union[Tensor, SimpleNamespace]]:
        input_mask = input_mask.point(lambda x: 0 if x < 255 / 2 else 255, 'L')

        input_image, input_mask = resize_image_to_multiple_of_64([input_image, input_mask], image_size)

        input_image, input_mask = self._migan_process(
            input_image, input_mask,
            mask_dilate_kernel_size=mask_dilate_kernel_size,
            mask_preprocess_type=mask_preprocess_type,
            enable_migan=enable_migan,
            migan_model=migan_model)

        input_image_copy = deepcopy(input_image)
        mask_image_copy = deepcopy(input_mask)

        image, mask, masked_image = self._denoise_preprocess(
            input_image, input_mask)

        _info = SimpleNamespace(
            input_image_copy=input_image_copy,
            mask_image_copy=mask_image_copy,
            input_size=input_image.size)

        return image, mask, masked_image, _info

    @staticmethod
    def _migan_process(input_image: I.Image, input_mask: I.Image,
                       mask_dilate_kernel_size=0,
                       mask_preprocess_type='dilate',
                       enable_migan=False,
                       migan_model=None) -> List[I.Image]:
        input_mask = RemovalSDXLPipeline_BatchMode.mask_preprocess(
            input_mask, mask_dilate_kernel_size, kind=mask_preprocess_type)
        image_migan = input_image
        if enable_migan and migan_model is not None:
            from torchvision.transforms import ToPILImage
            result = migan_model(input_image.copy(), input_mask.copy())
            image_migan = ToPILImage()(result.kwargs['inpaint'])
        return image_migan, input_mask

    def _denoise_preprocess(self, image_migan: I.Image, input_mask: I.Image) -> List[Tensor]:
        image = np.asarray(image_migan) / 255. * 2 - 1
        image = torch.tensor(image).permute(2, 0, 1)
        image = image.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        mask = np.asarray(input_mask) / 255.
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        mask, masked_image = self.prepare_mask_and_masked_image(image, mask)
        return image, mask, masked_image

    @staticmethod
    def _post_process(images: Tensor, _info: SimpleNamespace, paste=False, compensate=False) -> I.Image:
        input_image_copy, mask_image_copy = _info.input_image_copy, _info.mask_image_copy

        image = I.fromarray((images[0].permute(1, 2, 0).float().clamp(0, 1) * 255).cpu().numpy().astype(np.uint8))

        result_resize = image.resize(_info.input_size, resample=I.Resampling.LANCZOS)
        if paste and not compensate:
            m_img = mask_image_copy.convert('RGB').filter(
                IFilter.GaussianBlur(radius=3))
            m_img = np.asarray(m_img) / 255.0
            img_np = np.asarray(input_image_copy.convert('RGB')) / 255.0
            ours_np = np.asarray(result_resize) / 255.0
            ours_np = ours_np * m_img + (1 - m_img) * img_np
            out_arr = np.uint8(ours_np * 255)
            out_sample = I.fromarray(out_arr)
        elif paste and compensate:
            m_img = mask_image_copy.resize(_info.input_size, I.Resampling.NEAREST)
            out_sample = paste_compensate(mask_image_copy, input_image_copy, result_resize, fac=1.1)
        else:
            out_sample = result_resize
        return out_sample

    def _denoise_steps(self,
                       image: Tensor, mask: Tensor, masked_image: Tensor,
                       num_steps=20,
                       strength=0.99,
                       noise_offset=None,
                       guidance_scale=4.5,
                       mute=True,
                       visualize=False):
        self.noise_scheduler.set_timesteps(
            num_inference_steps=num_steps, device=self.device)
        timesteps, num_inference_steps = get_timesteps(
            self.noise_scheduler,
            num_inference_steps=num_steps,
            strength=strength,
            device=self.device)
        latent_timestep = timesteps[:1]

        with torch.no_grad():
            latents, masked_latents = encode_clean_latents(
                dict(images=image, masked_images=masked_image),
                vae=self.vae
            )

            # Resize mask: image size -> latent size
            h, w = mask.shape[-2:]
            size = (h // self.vae_ds_ratio, w // self.vae_ds_ratio)
            resized_masks = torch.nn.functional.interpolate(mask, size=size).to(device=self.device, dtype=self.dtype)

            # Add noise
            noise = torch.randn_like(latents)
            if noise_offset:
                noise += noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, latent_timestep) if strength < 1 else noise

            # Build input_ids for CFG
            uncond, cond = self.input_ids.chunk(2)
            input_ids = torch.cat([
                uncond.repeat(latents.shape[0], 1),
                cond.repeat(latents.shape[0], 1)
            ])

            noisy_latents_list = [noisy_latents]
            for t in tqdm(timesteps, disable=mute):
                t = t.to(device=self.device).unsqueeze(0)
                noisy_latents = self.noise_scheduler.scale_model_input(noisy_latents, t)

                noise_pred = predict_noise(
                    self.removal_model,
                    noisy_latents,
                    resized_masks=resized_masks,
                    masked_latents=masked_latents,
                    timesteps=t,
                    input_ids=input_ids,
                    guidance_scale=guidance_scale
                )

                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]
                noisy_latents_list.append(noisy_latents)

            if visualize:
                visualize_latent_steps(noisy_latents_list)

            images = self.vae.decode((noisy_latents / self.vae.config.scaling_factor).to(self.dtype)).sample
            images = (images + 1) / 2
        return images

    def __call__(
            self,
            input_image_list: List[I.Image],
            input_mask_list: List[I.Image],
            image_size=512,
            mask_dilate_kernel_size=0,
            mask_preprocess_type='dilate',
            strength=0.99,
            num_steps=20,
            guidance_scale=4.5,
            retry=0,
            enable_migan=False,
            migan_model=None,
            paste=False,
            compensate=False,
            noise_offset=None,
            no_cfg_after_step=1000,
            mute=True,
            visualize=False):
        seed = 0 if retry == 0 else retry
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if not isinstance(input_image_list, list):
            input_image_list = [input_image_list]
        if not isinstance(input_mask_list, list):
            input_mask_list = [input_mask_list]

        batch_image, batch_mask, batch_masked_image = [], [], []
        _info_list = []
        for input_image, input_mask in zip(input_image_list, input_mask_list):
            image, mask, masked_image, _info = self._preprocess(
                input_image, input_mask, image_size,
                mask_dilate_kernel_size=mask_dilate_kernel_size,
                mask_preprocess_type=mask_preprocess_type,
                enable_migan=enable_migan,
                migan_model=migan_model)
            batch_image.append(image)
            batch_mask.append(mask)
            batch_masked_image.append(masked_image)
            _info_list.append(_info)

        batch_image = torch.cat(batch_image)
        batch_mask = torch.cat(batch_mask)
        batch_masked_image = torch.cat(batch_masked_image)

        images = self._denoise_steps(
            batch_image, batch_mask, batch_masked_image,
            num_steps=num_steps,
            strength=strength,
            noise_offset=noise_offset,
            guidance_scale=guidance_scale,
            mute=mute, visualize=visualize
        )

        out_sample_list = []
        for image, _info in zip(images.split(1), _info_list):
            out_sample = self._post_process(
                image, _info, paste=paste, compensate=compensate
            )
            out_sample_list.append(out_sample)
        return out_sample_list
