import json
import numpy as np
import cv2
from PIL import Image
import random

from einops import rearrange
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import RandomMaskCrop, LaMaMaskGenerator


class RemovalDataset(Dataset):
    def __init__(
            self,
            ann_files,
            image_size,
            mask_config="config/rand_mask_cfg/random_medium_512.yaml",
            is_pure_background_train=False,
            extra_ann_files_4_PureBackTrain_2_RandMask=None,
            num_embeddings=20,
            use_rand_mask=True,
            use_extra_fg_mask=True,
            quiet=False
            ) -> None:
        super().__init__()
        """Dataset for object removal.

        Args:
            ann_files: a list of annotation file paths
        """
        self.data_source = []
        self.data_source_bg = []
        self.data_source_fg = []
        self.extra_data_source_fg = []

        self.num_beddings = num_embeddings

        self.use_rand_mask = use_rand_mask
        self.use_extra_fg_mask = use_extra_fg_mask

        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                if not quiet:
                    print(f'[Info]: Loading inpainting_jsonl: {ann_file}')
                for line in f.readlines():
                    # item = json.loads(line.strip())
                    if line is None:
                        continue

                    try:
                        item = json.loads(line.strip())
                    except json.decoder.JSONDecodeError:
                        item = eval(line.strip())

                    if isinstance(item, list):
                        item = item[1]
    
                    if not isinstance(item, dict):
                        continue

                    if item['prompt'] == 'background':
                        self.data_source_bg.append(item)
                    elif item['prompt'] == 'foreground':
                        self.data_source_fg.append(item)
                    else:
                        raise ValueError('Only `background` and `foreground`')
        self.data_source = self.data_source_bg + self.data_source_fg

        if not quiet:
            print(f'[Info]: has {len(self.data_source_bg)} background task samples.')
            print(f'[Info]: has {len(self.data_source_fg)} foreground task samples.')
            print(f'[Info]: has {len(self.data_source)} total samples.')
    

        if len(self.data_source_fg) == 0: 
            if extra_ann_files_4_PureBackTrain_2_RandMask is None:
                pass
            else:
                with open(extra_ann_files_4_PureBackTrain_2_RandMask, 'r') as ff:
                    print(f'[info]: Loading extra inpainting_jsonl to use rand obj mask as _tmp_mask, preventing the shape from following when generated: {extra_ann_files_4_PureBackTrain_2_RandMask}')
                    for line in ff.readlines():
                        item = json.loads(line.strip())
                        if item is None:
                            continue
                        if isinstance(item, list):
                            item = item[1]
                        if not isinstance(item, dict):
                            continue
                        if item['prompt'] == 'foreground':
                            self.extra_data_source_fg.append(item)
                        elif item['prompt'] != 'background':
                            raise ValueError('Only `background` and `foreground`')
                assert len(self.extra_data_source_fg) != 0
                print(f'[Info]: has {len(self.extra_data_source_fg)} extra foreground samples for fg_mask augmentation.')
                self.data_source_fg.extend(self.extra_data_source_fg)
        else: 
            assert not is_pure_background_train
  

        # Preprocessing the datasets.
        self.resize_transform = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        
        self.random_transforms = transforms.Compose(
            [
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip()
            ]
        )
        self.random_transforms_for_segmask = transforms.Compose(
            [
                RandomMaskCrop(image_size),
                transforms.RandomHorizontalFlip()
            ]
        )

        self.generator = LaMaMaskGenerator(mask_config)

    def __len__(self):
        return len(self.data_source)
    
    def _get_input_ids(self, task_type):
        if task_type == 'background':
            return torch.tensor(list(range(self.num_beddings//2)), dtype=torch.int64)
        if task_type == 'foreground':
            return torch.tensor(list(range(self.num_beddings//2, self.num_beddings)), dtype=torch.int64)
        raise ValueError('Task type error.')

    def _get_pool_ids(self, task_type):
        if task_type == 'background':
            return torch.tensor([0], dtype=torch.int64)
        if task_type == 'foreground':
            return torch.tensor([1], dtype=torch.int64)
        raise ValueError('Task type error.')

    def _prepare_item(self, index):
        example = self.data_source[index]
            
        source_image = Image.open(example["image"]).convert('RGB')
        h_ori, w_ori, _ = np.asarray(source_image).shape
        source_image = self.resize_transform(source_image)

        h, w, _ = np.asarray(source_image).shape

        task_type = example['prompt']
        input_ids = self._get_input_ids(task_type)
        pool_input_ids =self._get_pool_ids(task_type)
        if 'mask' not in example.keys(): # loading item from mask-free dataset
            random_mask = self.generator(source_image)
            mask = np.asarray(random_mask) / 255. # 0-1
        else:
            if task_type == 'background':
                mask = Image.open(example['mask']).convert('L')
                mask = mask.resize(size=(w, h))
                

                random_mask, fg_mask = None, None
                p = random.random()
                if p < 0.5: 
                    if self.use_rand_mask:
                        random_mask = self.generator(source_image) 
                        
                p = random.random()
                if p < 0.5:
                    if self.use_extra_fg_mask:
                        _tmp_example = self.data_source_fg[np.random.randint(0, len(self.data_source_fg))]
                        _tmp_mask = Image.open(_tmp_example["mask"]).convert('L')
                        fg_mask = _tmp_mask.resize((w, h))

                if random_mask is not None: 
                    mask = cv2.bitwise_and(np.asarray(mask), np.asarray(random_mask))

                if fg_mask is not None: 
                    mask = cv2.bitwise_and(np.asarray(mask), np.asarray(fg_mask))

                mask = np.asarray(mask) / 255. # 0-1
            elif task_type == 'foreground':
                mask = Image.open(example['mask']).convert('L')
                mask = np.asarray(mask.resize(size=(w, h))).astype(np.float32)
                mask = mask / 255.
            else:
                raise ValueError('Task type error.')

        assert mask.max() <= 1.
        random_mask = mask
        source_image = rearrange(2 * torch.tensor(np.array(source_image)).float() / 255 - 1, "h w c -> c h w") 
        random_mask = rearrange(torch.tensor(np.array(random_mask)).float(), "h w -> 1 h w")

        image_and_mask = self.random_transforms_for_segmask(torch.cat((source_image, random_mask)))
        source_image, random_mask = image_and_mask[:3], image_and_mask[3:]

        bg_flag = task_type == 'background'


        _, h, w = source_image.shape
        return dict(
            source_image=source_image, 
            mask=random_mask, 
            instruction=task_type,
            input_ids=input_ids, 
            pool_input_ids=pool_input_ids,
            original_sizes_hw=np.asarray([h, w]), 
            target_sizes_hw=np.asarray([h, w]), 
            crop_top_lefts=np.asarray([0, 0]),
            bg_flag=bg_flag
            )

    def __getitem__(self, index):
        return self._prepare_item(index)

    @staticmethod
    def prepare_mask_and_masked_image(image, mask):
        image = image.float() 
        mask = torch.where(mask >= 0.5, 1, 0)

        masked_image = image * (1 - mask) 

        return mask.to(dtype=torch.uint8), masked_image.to(dtype=torch.float32) 

    @staticmethod
    def collate_fn(examples):
        input_ids_collector = []
        pool_input_ids_collector = []

        image_collector = []
        mask_collector = []
        masked_image_collector = []
        instruction_collector = []

        original_sizes_hw_collector = []
        target_sizes_hw_collector = []
        crop_top_lefts_collector = []

        bg_flag_collector = []

        for example in examples:
            input_ids = example["input_ids"]
            pool_input_ids = example["pool_input_ids"]

            original_sizes_hw = torch.tensor(example["original_sizes_hw"])
            target_sizes_hw = torch.tensor(example["target_sizes_hw"])
            crop_top_lefts = torch.tensor(example["crop_top_lefts"])

            image = example["source_image"]
            mask = example["mask"]
            instruction = example["instruction"]

            # prepare mask and masked image
            mask, masked_image = RemovalDataset.prepare_mask_and_masked_image(image, mask)

            input_ids_collector.append(input_ids)
            pool_input_ids_collector.append(pool_input_ids)

            image_collector.append(image)
            mask_collector.append(mask)
            masked_image_collector.append(masked_image)
            instruction_collector.append(instruction)

            original_sizes_hw_collector.append(original_sizes_hw)
            target_sizes_hw_collector.append(target_sizes_hw)
            crop_top_lefts_collector.append(crop_top_lefts)

            bg_flag_collector.append(example["bg_flag"])

        input_ids_collector = torch.stack(input_ids_collector)
        pool_input_ids_collector = torch.stack(pool_input_ids_collector)

        image_collector = torch.stack(image_collector)
        image_collector = image_collector.to(memory_format=torch.contiguous_format).float()

        mask_collector = torch.stack(mask_collector)
        masked_image_collector = torch.stack(masked_image_collector)

        original_sizes_hw_collector = torch.stack([torch.LongTensor(x) for x in original_sizes_hw_collector])
        target_sizes_hw_collector = torch.stack([torch.LongTensor(x) for x in target_sizes_hw_collector])
        crop_top_lefts_collector = torch.stack([torch.LongTensor(x) for x in crop_top_lefts_collector])

        bg_flag_collector = torch.tensor(bg_flag_collector)
        
        return {
            "images": image_collector, 
            "text_instruction": instruction_collector,
            "input_ids": input_ids_collector, 
            "pool_input_ids": pool_input_ids_collector,
            "masks": mask_collector, 
            "masked_images": masked_image_collector,
            "original_sizes_hw": original_sizes_hw_collector, "target_sizes_hw": target_sizes_hw_collector, 
            "crop_top_lefts": crop_top_lefts_collector,
            "bg_flags": bg_flag_collector
            }

class RemovalDataset_v1_2(RemovalDataset):
    def __init__(
            self,
            ann_files,
            image_size,
            mask_config="config/rand_mask_cfg/random_medium_512.yaml",
            is_pure_background_train=False,
            extra_ann_files_4_PureBackTrain_2_RandMask=None,
            num_embeddings=20,
            use_rand_mask=True,
            use_extra_fg_mask=True,
            quiet=False
            ) -> None:
        super(RemovalDataset_v1_2, self).__init__(
            ann_files=ann_files,
            image_size=image_size,
            mask_config=mask_config,
            is_pure_background_train=is_pure_background_train,
            extra_ann_files_4_PureBackTrain_2_RandMask=extra_ann_files_4_PureBackTrain_2_RandMask,
            num_embeddings=num_embeddings,
            use_rand_mask=use_rand_mask,
            use_extra_fg_mask=use_extra_fg_mask,
            quiet=quiet)
 
        self.resize_transform = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.LANCZOS)

    def _prepare_item(self, index):
        example = self.data_source[index]
       
        try:
            source_image = Image.open(example["image"]).convert('RGB')
        except Exception as e:
            index = index+1
            example = self.data_source[index]
            source_image = Image.open(example["image"]).convert('RGB')
            

        h_ori, w_ori, _ = np.asarray(source_image).shape
        source_image = self.resize_transform(source_image)

        h, w, _ = np.asarray(source_image).shape

        task_type = example['prompt']
        input_ids = self._get_input_ids(task_type)
        pool_input_ids =self._get_pool_ids(task_type)

        if task_type == 'background':
            p = random.random()
            if p < 0.5:
                assert self.use_rand_mask, "must set use_rand_mask in dataset.yaml."
                mask = self.generator(source_image) 
            else: # scene semantic Mask
                if 'mask' not in example.keys(): # loading item from mask-free dataset
                    mask = self.generator(source_image)
                else:
                    mask = Image.open(example['mask']).convert('L')
                    mask = mask.resize(size=(w, h))

            fg_mask = None
            p = random.random()
            if p < 0.5: 
                if self.use_extra_fg_mask:
                    _tmp_example = self.data_source_fg[np.random.randint(0, len(self.data_source_fg))]
                    _tmp_mask = Image.open(_tmp_example["mask"]).convert('L')
                    fg_mask = _tmp_mask.resize((w, h))

            if fg_mask is not None: 
                mask = cv2.bitwise_and(np.asarray(mask), np.asarray(fg_mask))

            mask = np.asarray(mask) / 255. # 0-1
        elif task_type == 'foreground':
            mask = Image.open(example['mask']).convert('L')
            mask = np.asarray(mask.resize(size=(w, h))).astype(np.float32)
            mask = mask / 255.
        else:
            raise ValueError('Task type error.')

        if random.random() < 0.99: 
            kernel_size = 2 * random.randint(1, 3) + 1 # rand [1,2,3] -> kernel size [3,5,7]
            iters = random.randint(1, 2)
            mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iters)

        assert mask.max() <= 1.
        random_mask = mask
        source_image = rearrange(2 * torch.tensor(np.array(source_image)).float() / 255 - 1, "h w c -> c h w")  
        random_mask = rearrange(torch.tensor(np.array(random_mask)).float(), "h w -> 1 h w")

        image_and_mask = self.random_transforms_for_segmask(torch.cat((source_image, random_mask)))
        source_image, random_mask = image_and_mask[:3], image_and_mask[3:]

        bg_flag = task_type == 'background'


        _, h, w = source_image.shape
        return dict(
            source_image=source_image, 
            mask=random_mask, 
            instruction=task_type,
            input_ids=input_ids, 
            pool_input_ids=pool_input_ids,
            original_sizes_hw=np.asarray([h, w]), 
            target_sizes_hw=np.asarray([h, w]), 
            crop_top_lefts=np.asarray([0, 0]),
            bg_flag=bg_flag
            )