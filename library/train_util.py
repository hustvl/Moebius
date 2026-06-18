# common functions for training

import argparse
import ast
import asyncio
import importlib
import json
import orjson
import pathlib
import re
import shutil
import time
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from accelerate import Accelerator
import safetensors
import gc
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import transformers
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import AutoencoderKL
from library import custom_train_functions
import numpy as np
from PIL import Image
import cv2
from accelerate import DistributedDataParallelKwargs


# region dataset
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: torch.Tensor = None
        self.latents_flipped: torch.Tensor = None
        self.latents_npz: str = None
        self.latents_original_size: Tuple[int, int] = None  # original image size, not latents size
        self.latents_crop_ltrb: Tuple[int, int] = None  # crop left top right bottom in original pixel size, not latents size
        self.cond_img_path: str = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        # SDXL, optional
        self.text_encoder_outputs_npz: Optional[str] = None
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None
        # Inpainting Task
        self.rle_mask = Optional[str] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # print(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # print("use predef", image_width, image_height, reso, resized_size)
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # print(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # print(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # print(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # print("use arbitrary", image_width, image_height, resized_size, bucket_width, bucket_height)

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        num_repeats: int,
        shuffle_caption: bool,
        keep_tokens: int,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
        caption_key: str,
        load_jsonl_withopen: bool
    ) -> None:
        self.image_dir = image_dir
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる
        self.caption_key = caption_key
        self.load_jsonl_withopen = load_jsonl_withopen
        
        self.img_count = 0


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        num_repeats,
        shuffle_caption,
        keep_tokens,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            keep_tokens,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            token_warmup_min,
            token_warmup_step,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        num_repeats,
        shuffle_caption,
        keep_tokens,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        token_warmup_min,
        token_warmup_step,
        caption_key,
        load_jsonl_withopen
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            keep_tokens,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            token_warmup_min,
            token_warmup_step,
            caption_key,
            load_jsonl_withopen
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        conditioning_data_dir: str,
        caption_extension: str,
        num_repeats,
        shuffle_caption,
        keep_tokens,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            keep_tokens,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            token_warmup_min,
            token_warmup_step,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]],
        max_token_length: int,
        resolution: Optional[Tuple[int, int]],
        debug_dataset: bool,
    ) -> None:
        super().__init__()

        self.tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        self.max_token_length = max_token_length
        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.enable_dynamic_batch_size = False
        self.qwen_caption_prob = 0.5
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.tokenizer_max_length = self.tokenizers[0].model_max_length if max_token_length is None else max_token_length + 2

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            self.shuffle_buckets()
        self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                tokens = [t.strip() for t in caption.strip().split(",")]
                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor((self.current_step) * ((len(tokens) - subset.token_warmup_min) / (subset.token_warmup_step)))
                        + subset.token_warmup_min
                    )
                    tokens = tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                fixed_tokens = []
                flex_tokens = tokens[:]
                if subset.keep_tokens > 0:
                    fixed_tokens = flex_tokens[: subset.keep_tokens]
                    flex_tokens = tokens[subset.keep_tokens :]

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        token_res = tokenizer(
        caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt")
        input_ids = token_res.input_ids
        attention_mask = token_res.attention_mask

        # if self.tokenizer_max_length > tokenizer.model_max_length:
        #     input_ids = input_ids.squeeze(0)
        #     iids_list = []
        #     if tokenizer.pad_token_id == tokenizer.eos_token_id:
        #         # v1
        #         # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
        #         # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
        #         for i in range(
        #             1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
        #         ):  # (1, 152, 75)
        #             ids_chunk = (
        #                 input_ids[0].unsqueeze(0),
        #                 input_ids[i : i + tokenizer.model_max_length - 2],
        #                 input_ids[-1].unsqueeze(0),
        #             )
        #             ids_chunk = torch.cat(ids_chunk)
        #             iids_list.append(ids_chunk)
        #     else:
        #         # v2 or SDXL
        #         # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
        #         for i in range(1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
        #             ids_chunk = (
        #                 input_ids[0].unsqueeze(0),  # BOS
        #                 input_ids[i : i + tokenizer.model_max_length - 2],
        #                 input_ids[-1].unsqueeze(0),
        #             )  # PAD or EOS
        #             ids_chunk = torch.cat(ids_chunk)

        #             # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
        #             # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
        #             if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
        #                 ids_chunk[-1] = tokenizer.eos_token_id
        #             # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
        #             if ids_chunk[1] == tokenizer.pad_token_id:
        #                 ids_chunk[1] = tokenizer.eos_token_id

        #             iids_list.append(ids_chunk)

        #     input_ids = torch.stack(iids_list)  # 3,77
        return (input_ids, attention_mask)

    def register_image(self,info: ImageInfo, subset: BaseSubset,  idx=None):
        # self.image_data[info.image_key] = info
        if idx==None:
            idx = len(self.image_data)
        else:
            self.image_data[idx] = info
        self.image_to_subset[idx] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        print("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            print("make buckets")
        else:
            print("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    print(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for idx_key, image_info in self.image_data.items():
                if isinstance(image_info.image_size[0], list):
                    for item in image_info.image_size:
                        image_width, image_height = item
                        image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                            image_width, image_height
                        )
                        self.bucket_manager.add_image(image_info.bucket_reso, idx_key)
                        img_ar_errors.append(abs(ar_error))
                else:
                    image_width, image_height = image_info.image_size
                    image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                        image_width, image_height
                    )
                    self.bucket_manager.add_image(image_info.bucket_reso, idx_key)
                    img_ar_errors.append(abs(ar_error))
            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                if isinstance(image_info.image_size[0], list):
                    for item in image_info.image_size:
                        image_width, image_height = item
                        image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)
                else:
                    image_width, image_height = image_info.image_size
                    image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for idx_key in self.image_data.keys():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, idx_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            print("number of images (including repeats)")
            self.num_train_images = 0
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    print(f"bucket {i}: resolution {reso}, count: {len(bucket)}, aspect_ratio:{round(reso[0]/reso[1], 2)}")
                    self.num_train_images += len(bucket)
                    
            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            print(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices = [] #: List(BucketBatchIndex)
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            # print(f'train_utils.resos:{self.bucket_manager.resos[bucket_index]};train_utils.bucket:{len(bucket)}')
            if self.enable_dynamic_batch_size:
                reso = self.bucket_manager.resos[bucket_index]
                reso_value = reso[0] * reso[1]
                max_bucket_reso = self.max_bucket_reso
                bs_multiple = max(1, math.floor(max_bucket_reso * max_bucket_reso / reso_value))
                bs_multiple = min(8, bs_multiple) #  Limit the increase in multiples
                real_batch = bs_multiple * self.batch_size
                # real_batch = max(1, math.floor(max_bucket_reso * max_bucket_reso * self.batch_size / reso_value))
            else:
                real_batch = self.batch_size
            batch_count = int(math.ceil(len(bucket) / real_batch))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, real_batch, batch_index))

            # ↓以下はbucketごとのbatch件数があまりにも増えて混乱を招くので元に戻す
            # 　学習時はステップ数がランダムなので、同一画像が同一batch内にあってもそれほど悪影響はないであろう、と考えられる
            #
            # # bucketが細分化されることにより、ひとつのbucketに一種類の画像のみというケースが増え、つまりそれは
            # # ひとつのbatchが同じ画像で占められることになるので、さすがに良くないであろう
            # # そのためバッチサイズを画像種類までに制限する
            # # ただそれでも同一画像が同一バッチに含まれる可能性はあるので、繰り返し回数が少ないほうがshuffleの品質は良くなることは間違いない？
            # # TO DO 正則化画像をepochまたがりで利用する仕組み
            # num_of_image_types = len(set(bucket))
            # bucket_batch_size = min(self.batch_size, num_of_image_types)
            # batch_count = int(math.ceil(len(bucket) / bucket_batch_size))
            # # print(bucket_index, num_of_image_types, bucket_batch_size, batch_count)
            # for batch_index in range(batch_count):
            #   self.buckets_indices.append(BucketBatchIndex(bucket_index, bucket_batch_size, batch_index))
            # ↑ここまで

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)
    
    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]


        # torch.cuda.device_count()
        print(f'GLOBAL_NUM_PROCESSES: {torch.distributed.get_world_size()}')
        bucket_chunks = list(chunks(self.buckets_indices, torch.distributed.get_world_size()))

        random.shuffle(bucket_chunks)

        self.buckets_indices = [idx for chunk in bucket_chunks for idx in chunk]

        self.bucket_manager.shuffle()


    # def shuffle_buckets(self):
    #     # set random seed for this epoch
    #     random.seed(self.seed + self.current_epoch)

    #     random.shuffle(self.buckets_indices)
    #     self.bucket_manager.shuffle()

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        # マルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        print("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution
        batches = []
        batch = []
        print("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(info.bucket_reso, info.latents_npz, subset.flip_aug)

                if cache_available:  # do not add to batch
                    continue

            # if last member of batch has different resolution, flush the batch
            if len(batch) > 0 and batch[-1].bucket_reso != info.bucket_reso:
                batches.append(batch)
                batch = []

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        print("caching latents...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            cache_batch_latents(vae, cache_to_disk, batch, subset.flip_aug, subset.random_crop)

    # weight_dtypeを指定するとText Encoderそのもの、およひ出力がweight_dtypeになる
    # SDXLでのみ有効だが、datasetのメソッドとする必要があるので、sdxl_train_util.pyではなくこちらに実装する
    # SD1/2に対応するにはv2のフラグを持つ必要があるので後回し
    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        assert len(tokenizers) == 2, "only support SDXL"

        # latentsのキャッシュと同様に、ディスクへのキャッシュに対応する
        # またマルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        print("caching text encoder outputs.")
        image_infos = list(self.image_data.values())

        print("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            # subset = self.image_to_subset[info.image_key]
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:  # store to info only
                    continue

                if os.path.exists(te_out_npz):
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # prepare tokenizers and text encoders
        for text_encoder in text_encoders:
            text_encoder.to(device)
            if weight_dtype is not None:
                text_encoder.to(dtype=weight_dtype)

        # create batch
        batch = []
        batches = []
        for info in image_infos_to_cache:
            input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
            input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
            batch.append((info, input_ids1, input_ids2))

            if len(batch) >= self.batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        print("caching text encoder outputs...")
        for batch in tqdm(batches):
            infos, input_ids1, input_ids2 = zip(*batch)
            input_ids1 = torch.stack(input_ids1, dim=0)
            input_ids2 = torch.stack(input_ids2, dim=0)
            cache_batch_text_encoder_outputs(
                infos, tokenizers, text_encoders, self.max_token_length, cache_to_disk, input_ids1, input_ids2, weight_dtype
            )

    def get_image_size(self, image_path):
        image = Image.open(image_path)
        return image.size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str):
        img = load_image(image_path)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size
        bucket_resos = self.bucket_manager.resos[self.buckets_indices[index].bucket_index]

        if self.caching_mode is not None:  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        input_ids2_list = []
        input_att_mask1_list = []
        input_att_mask2_list = []
        latents_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs1_list = []
        text_encoder_outputs2_list = []
        text_encoder_pool2_list = []
        img_path_list = []
        caption_list = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                else:
                    latents = image_info.latents_flipped

                image = None
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents, original_size, crop_ltrb, flipped_latents = load_latents_from_disk(image_info.latents_npz)
                if flipped:
                    latents = flipped_latents
                    del flipped_latents
                latents = torch.FloatTensor(latents)

                image = None
            else:
                # 画像を読み込み、必要ならcropする
                try:
                    img_path_list.append(image_info.absolute_path)
                    img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(subset, image_info.absolute_path)
                    im_h, im_w = img.shape[0:2]
                except Exception as e:
                    print(e)
                    continue
                
                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        # subset.random_crop, img, image_info.bucket_reso, image_info.resized_size
                        subset.random_crop, img, bucket_resos, bucket_resos
                    )
                    # print(image_info.bucket_reso, bucket_resos, (im_h, im_w))
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    img = aug(image=img)["image"]

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる

            images.append(image)
            latents_list.append(latents)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default
            caption_list.append(caption)
            if isinstance(caption, list):
                if len(caption) > 1:
                    caption = self.select_caption_prob(caption, self.qwen_caption_prob)
                else:
                    caption = caption[0] if len(caption) > 0 else ''
                
            if image_info.text_encoder_outputs1 is not None:
                text_encoder_outputs1_list.append(image_info.text_encoder_outputs1)
                text_encoder_outputs2_list.append(image_info.text_encoder_outputs2)
                text_encoder_pool2_list.append(image_info.text_encoder_pool2)
                captions.append(caption)
            elif image_info.text_encoder_outputs_npz is not None:
                text_encoder_outputs1, text_encoder_outputs2, text_encoder_pool2 = load_text_encoder_outputs_from_disk(
                    image_info.text_encoder_outputs_npz
                )
                text_encoder_outputs1_list.append(text_encoder_outputs1)
                text_encoder_outputs2_list.append(text_encoder_outputs2)
                text_encoder_pool2_list.append(text_encoder_pool2)
                captions.append(caption)
            else:
                caption = image_info.caption
                if isinstance(caption, list):
                    if len(caption) > 1:
                        caption = self.select_caption_prob(caption, self.qwen_caption_prob)
                    else:
                        caption = caption[0] if len(caption) > 0 else ''
                    
                caption = self.process_caption(subset, caption)
                if self.XTI_layers:
                    caption_layer = []
                    for layer in self.XTI_layers:
                        token_strings_from = " ".join(self.token_strings)
                        token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                        caption_ = caption.replace(token_strings_from, token_strings_to)
                        caption_layer.append(caption_)
                    captions.append(caption_layer)
                else:
                    captions.append(caption)

                if not self.token_padding_disabled:  # this option might be omitted in future
                    if self.XTI_layers:
                        token_caption, attention_mask = self.get_input_ids(caption_layer, self.tokenizers[0])
                    else:
                        token_caption, attention_mask = self.get_input_ids(caption, self.tokenizers[0])
                    input_ids_list.append(token_caption)
                    input_att_mask1_list.append(attention_mask)

                    if len(self.tokenizers) > 1:
                        if self.XTI_layers:
                            token_caption2, attention_mask2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                        else:
                            token_caption2, attention_mask2 = self.get_input_ids(caption, self.tokenizers[1])
                        input_ids2_list.append(token_caption2)
                        input_att_mask2_list.append(attention_mask2)

        example = {}
        example["loss_weights"] = torch.FloatTensor(loss_weights)

        if len(text_encoder_outputs1_list) == 0:
            if self.token_padding_disabled:
                # padding=True means pad in the batch
                example["input_ids"] = self.tokenizer[0](captions, padding=True, truncation=True, return_tensors="pt").input_ids
                if len(self.tokenizers) > 1:
                    example["input_ids2"] = self.tokenizer[1](
                        captions, padding=True, truncation=True, return_tensors="pt"
                    ).input_ids
                else:
                    example["input_ids2"] = None
            else:
                example["input_ids"] = torch.stack(input_ids_list)
                example["attention_mask"] = torch.stack(input_att_mask1_list)
                example["input_ids2"] = torch.stack(input_ids2_list) if len(self.tokenizers) > 1 else None
                example["attention_mask2"] = torch.stack(input_att_mask2_list) if len(self.tokenizers) > 1 else None
            
            example['img_path'] = img_path_list
            example['caption'] = caption_list
            
            example["text_encoder_outputs1_list"] = None
            example["text_encoder_outputs2_list"] = None
            example["text_encoder_pool2_list"] = None
        else:
            example["input_ids"] = None
            example["input_ids2"] = None
            example["attention_mask"] = None
            example["attention_mask2"] = None
            # # for assertion
            # example["input_ids"] = torch.stack([self.get_input_ids(cap, self.tokenizers[0]) for cap in captions])
            # example["input_ids2"] = torch.stack([self.get_input_ids(cap, self.tokenizers[1]) for cap in captions])
            example["text_encoder_outputs1_list"] = torch.stack(text_encoder_outputs1_list)
            example["text_encoder_outputs2_list"] = torch.stack(text_encoder_outputs2_list)
            example["text_encoder_pool2_list"] = torch.stack(text_encoder_pool2_list)

        if len(images) and images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        while images is None:
            random_idx = random.sample(range(self.__len__()), 1)[0]
            print(f"images is None, random idx is:{random_idx}")
            example = self.__getitem__(random_idx)
        
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.
            if isinstance(caption, list):
                if len(caption) > 1:
                    caption = self.select_caption_prob(caption, self.qwen_caption_prob)
                else:
                    caption = caption[0] if len(caption) > 0 else ''
            
            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example

    def select_caption_prob(self, caption, last_item_prob=0.9):
        weights = [1-last_item_prob] * (len(caption) - 1)  
        weights.append(last_item_prob)  
        chosen = random.choices(caption, weights, k=1)
        return chosen[0]

class DreamBoothDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        assert resolution is not None, f"resolution is required / resolution（解像度）指定は必須です"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            assert (
                min(resolution) >= min_bucket_reso
            ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
            assert (
                max(resolution) <= max_bucket_reso
            ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            print(f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
                        caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                print(f"not directory: {subset.image_dir}")
                return [], []

            img_paths = glob_images(subset.image_dir, "*")
            print(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
            captions = []
            missing_captions = []
            for img_path in img_paths:
                cap_for_img = read_caption(img_path, subset.caption_extension)
                if cap_for_img is None and subset.class_tokens is None:
                    print(
                        f"neither caption file nor class tokens are found. use empty caption for {img_path} / キャプションファイルもclass tokenも見つかりませんでした。空のキャプションを使用します: {img_path}"
                    )
                    captions.append("")
                    missing_captions.append(img_path)
                else:
                    if cap_for_img is None:
                        captions.append(subset.class_tokens)
                        missing_captions.append(img_path)
                    else:
                        captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # タグ頻度を記録

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                print(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}枚の画像にキャプションファイルが見つかりませんでした。これらの画像についてはキャプションなしで学習を続行します。class tokenが存在する場合はそれを使います。"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        print(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    print(missing_caption)
            return img_paths, captions

        print("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[ImageInfo] = []
        for subset in subsets:
            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one"
                )
                continue

            img_paths, captions = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                print(f"ignore subset with image_dir='{subset.image_dir}': no images found")
                continue

            if subset.is_reg:
                num_reg_images += subset.num_repeats * len(img_paths)
            else:
                num_train_images += subset.num_repeats * len(img_paths)

            for img_path, caption in zip(img_paths, captions):
                info = ImageInfo(img_path, subset.num_repeats, caption, subset.is_reg, img_path)
                if subset.is_reg:
                    reg_infos.append(info)
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        print(f"{num_train_images} train images with repeating.")
        self.num_train_images = num_train_images

        print(f"{num_reg_images} reg images.")
        if num_train_images < num_reg_images:
            print("some of reg images are not used")

        if num_reg_images == 0:
            print("no regularization images")
        else:
            n = 0
            first_loop = True
            while n < num_train_images:
                for info in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images

class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        enable_dynamic_batch_size: bool,
        qwen_caption_prob: float,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.enable_dynamic_batch_size = enable_dynamic_batch_size
        self.qwen_caption_prob = qwen_caption_prob
        for subset in subsets:
            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one"
                )
                continue
            
            # if subset.load_jsonl_withopen:
            if os.path.exists(subset.metadata_file):
                print(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file) as f:
                    lines = f.readlines()
                chunk = [
                    {
                        'img_path': json_data['img_path'],
                        subset.caption_key: json_data[subset.caption_key],
                        'img_size': json_data['img_size'],
                        'train_resolution': json_data['train_resolution']
                    }
                    for line in tqdm(lines, desc="Loading metadata", ncols=120, unit=" lines")
                    for json_data in (orjson.loads(line),)
                ]

            else:
                raise ValueError(f"no metadata: {subset.metadata_file}")

            tags_list = []
            data_count = 0
            for idx, img_md in enumerate(chunk):
                abs_path = img_md['img_path']
                image_key = abs_path
                
                caption_key = subset.caption_key if subset.caption_key else "caption"
                caption = img_md.get(caption_key, "")
                if data_count < 10:
                    print(caption)

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                img_size = img_md.get("img_size", [0,0])
                train_resolution = img_md.get("train_resolution", [0,0])
                # train_resolution = self.limit_areas(img_size, max_pixel_nums=1024*1024, step=64)
                if (0 in img_size) or (-1 in img_size) or (0 in train_resolution) or (-1 in train_resolution):
                    continue
                
                image_info.image_size = train_resolution
                self.register_image(image_info, subset, idx)
                data_count = data_count + 1

            self.num_train_images += data_count * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = data_count
            self.subsets.append(subset)

        # check existence of all npz files
        use_npz_latents = all([not (subset.color_aug or subset.random_crop) for subset in self.subsets])
        if use_npz_latents:
            flip_aug_in_subset = False
            npz_any = False
            npz_all = True

            for idx_key,image_info in self.image_data.items():
                subset = self.image_to_subset[idx_key]

                has_npz = image_info.latents_npz is not None
                npz_any = npz_any or has_npz

                if subset.flip_aug:
                    has_npz = has_npz and image_info.latents_npz_flipped is not None
                    flip_aug_in_subset = True
                npz_all = npz_all and has_npz

                if npz_any and not npz_all:
                    break

            if not npz_any:
                use_npz_latents = False
                print(f"npz file does not exist. ignore npz files")
            elif not npz_all:
                use_npz_latents = False
                print(f"some of npz file does not exist. ignore npz files")
                if flip_aug_in_subset:
                    print("maybe no flipped files")
        # else:
        #   print("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            if isinstance(image_info.image_size[0], list):
                for item in image_info.image_size:
                    sizes.add(item[0])
                    sizes.add(item[1])
                    resos.add(tuple(item))
            else:
                sizes.add(image_info.image_size[0])
                sizes.add(image_info.image_size[1])
                resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                print(f"npz files exist, but no bucket info in metadata. ignore npz files")

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                print("metadata has bucket info, enable bucketing")
            print("using bucket info in metadata")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used"

            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, subset: FineTuningSubset, image_key):
        base_name = os.path.splitext(image_key)[0]
        npz_file_norm = base_name + ".npz"

        if os.path.exists(npz_file_norm):
            # image_key is full path
            npz_file_flip = base_name + "_flip.npz"
            if not os.path.exists(npz_file_flip):
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # if not full path, check image_dir. if image_dir is None, return None
        if subset.image_dir is None:
            return None, None

        # image_key is relative path
        npz_file_norm = os.path.join(subset.image_dir, image_key + ".npz")
        npz_file_flip = os.path.join(subset.image_dir, image_key + "_flip.npz")

        if not os.path.exists(npz_file_norm):
            npz_file_norm = None
            npz_file_flip = None
        elif not os.path.exists(npz_file_flip):
            npz_file_flip = None

        return npz_file_norm, npz_file_flip

    def limit_areas(self, img_size, max_pixel_nums=1152*1152, step=64):
        width_src = img_size[0]
        height_src = img_size[1]
        src_pixel_nums = width_src * height_src
        
        if 0 in img_size or -1 in img_size:
            return [0, 0]
        
        if max(width_src, height_src) / min(width_src, height_src) > 5:
            return [0, 0]

        if src_pixel_nums > max_pixel_nums:
            scaling_factor = math.sqrt(max_pixel_nums / src_pixel_nums)
            width = width_src * scaling_factor
            height = height_src * scaling_factor
        else:
            width = width_src
            height = height_src

        width_resize = int(math.floor(width / step) * step)
        height_resize = int(math.floor(height / step) * step)

        return [width_resize, height_resize]


class InpaintingDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        enable_dynamic_batch_size: bool,
        qwen_caption_prob: float,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.enable_dynamic_batch_size = enable_dynamic_batch_size
        self.qwen_caption_prob = qwen_caption_prob
        for subset in subsets:
            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one"
                )
                continue
            
            chunk = []
            if os.path.exists(subset.metadata_file):
                print(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file) as f:
                    for line in tqdm(f.readlines(), desc="Loading metadata", ncols=120, unit=" lines"):
                        item = json.loads(line.strip())
                        chunk.append(item)

            else:
                raise ValueError(f"no metadata: {subset.metadata_file}")

            tags_list = []
            data_count = 0
            for idx, img_md in enumerate(chunk):
                abs_path = img_md['ImagePath']
                image_key = abs_path
                
                caption_key = "Caption"
                caption = img_md.get(caption_key, "")
                if data_count < 10:
                    print(caption)

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                img_size = img_md.get("ImageSize", [0,0])
                train_resolution = img_md.get("TrainResolution", [0,0])
                # train_resolution = self.limit_areas(img_size, max_pixel_nums=1024*1024, step=64)
                if (0 in img_size) or (-1 in img_size) or (0 in train_resolution) or (-1 in train_resolution):
                    continue
                
                image_info.image_size = train_resolution
                image_info.rle_mask = img_md['RLE']
                self.register_image(image_info, subset, idx)
                data_count = data_count + 1

            self.num_train_images += data_count * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = data_count
            self.subsets.append(subset)

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            if isinstance(image_info.image_size[0], list):
                for item in image_info.image_size:
                    sizes.add(item[0])
                    sizes.add(item[1])
                    resos.add(tuple(item))
            else:
                sizes.add(image_info.image_size[0])
                sizes.add(image_info.image_size[1])
                resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                print(f"npz files exist, but no bucket info in metadata. ignore npz files")

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                print("metadata has bucket info, enable bucketing")
            print("using bucket info in metadata")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used"

            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def __getitem__(self, index):
        return super().__getitem__(index)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        db_subsets = []
        for subset in subsets:
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.keep_tokens,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.token_warmup_min,
                subset.token_warmup_step,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            batch_size,
            tokenizer,
            max_token_length,
            resolution,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_img = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                print(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.basename(info.absolute_path)
            ctrl_img_path = os.path.join(subset.conditioning_data_dir, img_basename)
            if not os.path.exists(ctrl_img_path):
                missing_imgs.append(img_basename)

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_img.add(ctrl_img_path)

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            extra_imgs.extend(
                [cond_img_path for cond_img_path in conditioning_img_paths if cond_img_path not in cond_imgs_with_img]
            )

        assert len(missing_imgs) == 0, f"missing conditioning data for {len(missing_imgs)} images: {missing_imgs}"
        assert len(extra_imgs) == 0, f"extra conditioning data for {len(extra_imgs)} images: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                cond_img = cv2.resize(cond_img, image_info.resized_size, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"
                ct, cl = crop_top_left
                h, w = target_size_hw
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                assert (
                    cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        for i, dataset in enumerate(self.datasets):
            print(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            print(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    npz = np.load(npz_path)
    if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
        return False
    if npz["latents"].shape[1:3] != expected_latents_size:
        return False

    if flip_aug:
        if "latents_flipped" not in npz:
            return False
        if npz["latents_flipped"].shape[1:3] != expected_latents_size:
            return False

    return True


# 戻り値は、latents_tensor, (original_size width, original_size height), (crop left, crop top)
def load_latents_from_disk(
    npz_path,
) -> Tuple[Optional[torch.Tensor], Optional[List[int]], Optional[List[int]], Optional[torch.Tensor]]:
    npz = np.load(npz_path)
    if "latents" not in npz:
        raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

    latents = npz["latents"]
    original_size = npz["original_size"].tolist()
    crop_ltrb = npz["crop_ltrb"].tolist()
    flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
    return latents, original_size, crop_ltrb, flipped_latents


def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
    np.savez(
        npz_path,
        latents=latents_tensor.float().cpu().numpy(),
        original_size=np.array(original_size),
        crop_ltrb=np.array(crop_ltrb),
        **kwargs,
    )


def debug_dataset(train_dataset, show_input_ids=False):
    print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    print("`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します")

    epoch = 1
    while True:
        print(f"\nepoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            print(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                print(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, iid, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                print(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )

                if show_input_ids:
                    print(f"input ids: {iid}")
                    if "input_ids2" in example:
                        print(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    print(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        print(f"conditioning image size: {cond_img.size()}")
                        cond_img = (cond_img.numpy() * 255.0).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths


class MinimalDataset(BaseDataset):
    def __init__(self, tokenizer, max_token_length, resolution, debug_dataset=False):
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError


def load_arbitrary_dataset(args, tokenizer) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path):
    image = Image.open(image_path)
    if image.mode == 'P':
        image = image.convert("RGBA").convert("RGB")
    if not image.mode == "RGB":
        image = image.convert('RGB')
        
    img = np.array(image, np.uint8)
    return img


def trim_and_resize_if_required(
    random_crop: bool, image: Image.Image, reso, resized_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        
        padding = 0.02
        scaling_factor = resized_size[0] / image_width
        while True:
            resized_width = int(image_width * scaling_factor)
            resized_height = int(image_height * scaling_factor)
            if resized_height >= resized_size[1] and resized_width >= resized_size[0]:
                break
            else:
                scaling_factor += padding

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    image_height, image_width = image.shape[0:2]

    left_p = 0
    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # print("w", trim_size, p)
        image = image[:, p : p + reso[0]]
        left_p = p
    top_p = 0
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # print("h", trim_size, p)
        image = image[p : p + reso[1]]
        top_p = p
    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    # crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)
    crop_ltrb = [left_p, top_p, left_p+reso[0], top_p+reso[1]]  # crop_left, crop_top, crop_right, crop_bottom
    original_size = (image_width, image_height)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}, {resized_width}, {resized_height}"
    return image, original_size, crop_ltrb


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    for info in image_infos:
        image = load_image(info.absolute_path) if info.image is None else np.array(info.image, np.uint8)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(random_crop, image, info.bucket_reso, info.resized_size)
        image = IMAGE_TRANSFORMS(image)
        images.append(image)

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent in zip(image_infos, latents, flipped_latents):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            save_latents_to_disk(info.latents_npz, latent, info.latents_original_size, info.latents_crop_ltrb, flipped_latent)
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent




# def cache_batch_text_encoder_outputs(
#     image_infos, tokenizers, text_encoders, max_token_length, cache_to_disk, input_ids1, input_ids2, dtype
# ):
#     input_ids1 = input_ids1.to(text_encoders[0].device)
#     input_ids2 = input_ids2.to(text_encoders[1].device)

#     with torch.no_grad():
#         b_hidden_state1, b_hidden_state2, b_pool2 = get_hidden_states_sdxl(
#             max_token_length,
#             input_ids1,
#             input_ids2,
#             tokenizers[0],
#             tokenizers[1],
#             text_encoders[0],
#             text_encoders[1],
#             dtype,
#         )

#         # ここでcpuに移動しておかないと、上書きされてしまう
#         b_hidden_state1 = b_hidden_state1.detach().to("cpu")  # b,n*75+2,768
#         b_hidden_state2 = b_hidden_state2.detach().to("cpu")  # b,n*75+2,1280
#         b_pool2 = b_pool2.detach().to("cpu")  # b,1280

#     for info, hidden_state1, hidden_state2, pool2 in zip(image_infos, b_hidden_state1, b_hidden_state2, b_pool2):
#         if cache_to_disk:
#             save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, hidden_state1, hidden_state2, pool2)
#         else:
#             info.text_encoder_outputs1 = hidden_state1
#             info.text_encoder_outputs2 = hidden_state2
#             info.text_encoder_pool2 = pool2


def save_text_encoder_outputs_to_disk(npz_path, hidden_state1, hidden_state2, pool2):
    np.savez(
        npz_path,
        hidden_state1=hidden_state1.cpu().float().numpy(),
        hidden_state2=hidden_state2.cpu().float().numpy(),
        pool2=pool2.cpu().float().numpy(),
    )


def load_text_encoder_outputs_from_disk(npz_path):
    with np.load(npz_path) as f:
        hidden_state1 = torch.from_numpy(f["hidden_state1"])
        hidden_state2 = torch.from_numpy(f["hidden_state2"]) if "hidden_state2" in f else None
        pool2 = torch.from_numpy(f["pool2"]) if "pool2" in f else None
    return hidden_state1, hidden_state2, pool2


# endregion

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except:
        return "(unknown)"



def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model")
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use: AdamW (default), AdamW8bit, PagedAdamW8bit, Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, AdaFactor",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) ",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch)",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping"
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...")',
    )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100")',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler (default is 0)",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument("--output_dir", type=str, default=None, help="directory to output trained model")
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file")
    parser.add_argument(
        "--huggingface_repo_id", type=str, default=None, help="huggingface repo name to upload"
    )
    parser.add_argument(
        "--huggingface_repo_type", type=str, default=None, help="huggingface repo type to upload"
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files"
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private)",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving",
    )
    parser.add_argument(
        "--save_every_n_epochs", type=int, default=None, help="save checkpoint every N epochs"
    )
    parser.add_argument(
        "--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps"
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total)",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints)",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed)",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps)",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.)",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225)",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention",
    )
    parser.add_argument("--xformers", action="store_true", help="use xformers for CrossAttention")
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=100,
        help="training epochs (overrides max_train_steps)",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading)",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="use mixed precision"
    )
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients")
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients"
    )  # TODO move to SDXL training, because it is not supported by SD1/2
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1)",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used)",
    )
    parser.add_argument("--log_prefix", type=str, default=None, help="add prefix for each log directory")
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional).",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended)",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) ",
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations)",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default)",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0)",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000)",
    )

    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle)",
    )

    parser.add_argument(
        "--sample_every_n_steps", type=int, default=None, help="generate sample images every N steps"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps)",
    )
    parser.add_argument(
        "--sample_prompts", type=str, default=None, help="file for prompts to generate sample images"
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type for sample images",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file"
    )
    parser.add_argument("--script_args", type=str, default='', help="train.sh info")

    if support_dreambooth:
        # DreamBooth training
        parser.add_argument(
            "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images"
        )


def verify_training_args(args: argparse.Namespace):
    if args.v_parameterization and not args.v2:
        print("v_parameterization should be with v2 not v1 or sdxl")
    if args.v2 and args.clip_skip is not None:
        print("v2 with clip_skip will be unexpected")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        print(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled"
        )

    # noise_offset, perlin_noise, multires_noise_iterations cannot be enabled at the same time
    # Listを使って数えてもいいけど並べてしまえ
    if args.noise_offset is not None and args.multires_noise_iterations is not None:
        raise ValueError(
            "noise_offset and multires_noise_iterations cannot be enabled at the same time"
        )
    # if args.noise_offset is not None and args.perlin_noise is not None:
    #     raise ValueError("noise_offset and perlin_noise cannot be enabled at the same time / noise_offsetとperlin_noiseは同時に有効にできません")
    # if args.perlin_noise is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "perlin_noise and multires_noise_iterations cannot be enabled at the same time / perlin_noiseとmultires_noise_iterationsを同時に有効にできません"
    #     )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError(
            "v_pred_like_loss cannot be enabled with v_parameterization"
        )

    if args.zero_terminal_snr and not args.v_parameterization:
        print(
            f"zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
        )


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    # dataset common
    parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images")
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle comma-separated caption"
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="extension of caption files"
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility)",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings)",
    )
    parser.add_argument("--color_aug", action="store_true", help="enable weak color augmentation")
    parser.add_argument("--flip_aug", action="store_true", help="enable horizontal flip augmentation")
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0)",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation)",
    )
    parser.add_argument(
        "--debug_dataset", action="store_true", help="show images for debugging (do not train)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height')",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled)",
    )
    parser.add_argument("--vae_batch_size", type=int, default=1, help="batch size for caching latents")
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled)",
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="enable buckets for multi aspect ratio training"
    )
    parser.add_argument(
        "--enable_dynamic_batch_size", action="store_true", default=False, help="enable dynamic batch size for training"
    )
    parser.add_argument(
        "--qwen_caption_prob", type=float, default=0.5, help="qwen_caption_prob"
    )
    parser.add_argument("--min_bucket_reso", type=int, default=64, help="minimum resolution for buckets")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended",
    )
    parser.add_argument(
        "--bucket_no_upscale", action="store_true", help="make bucket for each image without upscaling"
    )

    parser.add_argument(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs)",
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1)",
    )

    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class)",
    )

    if support_caption_dropout:
        parser.add_argument(
            "--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0)"
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0)",
        )

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images")

    if support_caption:
        # caption dataset
        parser.add_argument("--in_json", type=str, default=None, help="json metadata for dataset")
        parser.add_argument(
            "--dataset_repeats", type=int, default=0, help="repeat dataset when training with captions"
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original)",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified)",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            print(f"Config file already exists. Aborting...: {config_path}")
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        print(f"Saved config file: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        print(f"{config_path} not found.")
        exit(1)

    print(f"Loading settings from {config_path}...")
    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    print(args.config_file)

    return args




def get_optimizer(args, trainable_params, named_trainable_params=None):
    # "Optimizer to use: AdamW, AdamW8bit, Muon"
    # use named_trainable_params for Muon to split parameter groups

    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified"
        optimizer_type = "AdamW8bit"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    # 引数を分解する
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)

            optimizer_kwargs[key] = value
    # print("optkwargs:", optimizer_kwargs)

    lr = args.learning_rate
    optimizer = None

    if optimizer_type.endswith("8bit".lower()):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes")

        if optimizer_type == "AdamW8bit".lower():
            print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = bnb.optim.AdamW8bit
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(f"SGD with Nesterov must be with momentum, set momentum to 0.9")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Muon".lower():
        print(f"use Muon optimizer | {optimizer_kwargs}")
        from moonlight.muon import Muon
        optimizer_class = Muon
        named_model_params = named_trainable_params[0]["params"] # 在目前的蒸馏训练代码下，trainable_params里只会有一个模型的参数，所以直接取下标0了
        # maybe move 'conv_out' from Muon to AdamW as it's considered diffusion head?
        # maybe move 'embedding_layer' from Muon to AdamW as it's considered embedding?
        muon_params  = [
            p 
            for name, p in named_model_params 
            if     p.ndim >= 2 and "embedding_layer" not in name and "conv_out" not in name and "conv_in" not in name
        ]
        adamw_params = [
            p 
            for name, p in named_model_params 
            if not (
                p.ndim >= 2 and "embedding_layer" not in name and "conv_out" not in name and "conv_in" not in name
            )
        ]
        print(f"params optimized by Muon :",
              f"{sum([p.numel() for p in muon_params])} | {len(muon_params)}")
        print(f"params optimized by AdamW:",
              f"{sum([p.numel() for p in adamw_params])} | {len(adamw_params)}")
        optimizer = optimizer_class(
            lr=lr,
            muon_params=muon_params,
            adamw_params=adamw_params,
            **optimizer_kwargs)

    if optimizer is None:
        optimizer_type = args.optimizer_type
        print(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    return optimizer_name, optimizer_args, optimizer

# Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# Add some checking and features to the original function.


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    name = args.lr_scheduler
    num_warmup_steps: Optional[int] = args.lr_warmup_steps
    num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        print(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer"
        initial_lr = float(name.split(":")[1])
        # print("adafactor scheduler init lr", initial_lr)
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

    if name == SchedulerType.PIECEWISE_CONSTANT:
        return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **lr_scheduler_kwargs,
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power, **lr_scheduler_kwargs
        )

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **lr_scheduler_kwargs)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    # assert args.resolution is not None, f"resolution is required"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert (
            len(args.resolution) == 2
        ), f"resolution must be 'size' or 'width,height': {args.resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            print(
                f"latents in npz is ignored when color_aug or random_crop is True"
            )


def prepare_accelerator(args: argparse.Namespace, fsdp_plugin: None, dynamo_backend=None):
    # prepare logger
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())
    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)


    # prepare accelerator
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        fsdp_plugin=fsdp_plugin,
        dynamo_backend=dynamo_backend
        # kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        src_path_list, dst_path_list = [], []
        
        if args.script_args:
            sh_basename = os.path.basename(args.script_args)
            src_path_list.append(args.script_args)
            dst_path_list.append(os.path.join(logging_dir, sh_basename))
        
        if hasattr(args,"dataset_config"):
            data_basename = os.path.basename(args.dataset_config)
            src_path_list.extend(args.dataset_config)
            dst_path_list.extend(os.path.join(logging_dir, data_basename))
            
        # 备份重要代码
        cur_file_dir = os.path.dirname(os.path.realpath(__file__))
        src_path_list.extend([
            os.path.join(cur_file_dir, 'train_util.py'),
            os.path.join(cur_file_dir, 'chinese_sdxl_train_util.py')
        ])
        dst_path_list.extend([
            os.path.join(logging_dir, 'train_util.py'),
            os.path.join(logging_dir, 'chinese_sdxl_train_util.py')
        ])

        for src, dst in zip(src_path_list, dst_path_list):
            try:
                shutil.copyfile(src, dst)
            except Exception as e:
                print("===>", e)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def default_if_none(value, default):
    return default if value is None else value


def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset: # 默认None
        noise = custom_train_functions.apply_noise_offset(latents, noise, args.noise_offset, args.adaptive_noise_scale)
    elif args.multires_noise_iterations: # 默认none
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    if args.lognorm_t:
        timesteps = torch.randn((b_size,), device=latents.device)
        timesteps = torch.sigmoid(timesteps)
        timesteps *= 1000
        timesteps = timesteps.long()
    else:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device=latents.device)
        timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            print(f"Could not load image path: {img_path}, error: {e}")
            return None

        return (tensor_pil, img_path)


# endregion


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))

        size += divisible

    resos = list(resos)
    resos.sort()
    return resos

# collate_fn用 epoch,stepはmultiprocessing.Value
class collater_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


# import copy
# class EMAModel:
#     """
#     Exponential Moving Average of models weights
#     """

#     def __init__(
#         self,
#         model,
#         update_after_step=0,
#         inv_gamma=1.0,
#         power=2 / 3,
#         min_value=0.0,
#         max_value=0.9999,
#         device=None,
#     ):
#         """
#         @crowsonkb's notes on EMA Warmup:
#             If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
#             to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
#             gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
#             at 215.4k steps).
#         Args:
#             inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
#             power (float): Exponential factor of EMA warmup. Default: 2/3.
#             min_value (float): The minimum EMA decay rate. Default: 0.
#         """

#         self.averaged_model = copy.deepcopy(model).eval()
#         self.averaged_model.requires_grad_(False)

#         self.update_after_step = update_after_step
#         self.inv_gamma = inv_gamma
#         self.power = power
#         self.min_value = min_value
#         self.max_value = max_value

#         if device is not None:
#             self.averaged_model = self.averaged_model.to(device=device)

#         self.decay = 0.0
#         self.optimization_step = 0

#     def get_decay(self, optimization_step):
#         """
#         Compute the decay factor for the exponential moving average.
#         """
#         step = max(0, optimization_step - self.update_after_step - 1)
#         value = 1 - (1 + step / self.inv_gamma) ** -self.power

#         if step <= 0:
#             return 0.0

#         return max(self.min_value, min(value, self.max_value))

#     @torch.no_grad()
#     def step(self, new_model):
#         ema_state_dict = {}
#         ema_params = self.averaged_model.state_dict()

#         self.decay = self.get_decay(self.optimization_step)

#         for key, param in new_model.named_parameters():
#             if isinstance(param, dict):
#                 continue
#             try:
#                 ema_param = ema_params[key]
#             except KeyError:
#                 ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
#                 ema_params[key] = ema_param

#             if not param.requires_grad:
#                 ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
#                 ema_param = ema_params[key]
#             else:
#                 ema_param.mul_(self.decay)
#                 ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

#             ema_state_dict[key] = ema_param

#         for key, param in new_model.named_buffers():
#             ema_state_dict[key] = param

#         self.averaged_model.load_state_dict(ema_state_dict, strict=False)
#         self.optimization_step += 1
