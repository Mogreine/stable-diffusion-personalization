import os
from pathlib import Path
from typing import Optional, List

import PIL.Image
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL
from loguru import logger
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer

from src.utils import extract_filename


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer: CLIPTokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
        vae=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.image_captions_filename = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.instance_images, self.instance_prompts = self._load_images(instance_data_root, instance_prompt)
        self.is_latents_precalculated = False
        if vae is not None:
            logger.info("Precalculating VAE latents...")
            self.instance_images = self._precalculate_vae_latents(self.instance_images, vae)
            # logger.info(f"Images shape after precalculation: {self.instance_images.shape}")
            self.is_latents_precalculated = True
            logger.info("Done!")

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_images, self.class_prompts = self._load_images(class_data_root, class_prompt)
            if vae is not None:
                logger.info("Precalculating VAE latents...")
                self.class_images = self._precalculate_vae_latents(self.class_images, vae)
                # logger.info(f"Images shape after precalculation: {self.instance_images.shape}")
                logger.info("Done!")

            self.num_class_images = len(self.class_images)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def _load_images(self, dir: str, prompt: Optional[str] = None):
        images, prompts = [], []
        for path in list(Path(dir).iterdir()):
            image = Image.open(path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            img_prompt = prompt if prompt else extract_filename(path)
            image = self.image_transforms(image)

            images.append(image)
            prompts.append(img_prompt)

        return images, prompts

    def __len__(self):
        return self._length

    def _prepare_sample(self, index, images, prompts):
        dataset_length = len(images)
        image, prompt = images[index % dataset_length], prompts[index % dataset_length]
        prompt = self.tokenizer(
            prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return image, prompt

    @torch.no_grad()
    def _precalculate_vae_latents(
        self, images: List[torch.Tensor], vae: AutoencoderKL, batch_size: int = 32
    ) -> torch.Tensor:
        res_latents = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch = torch.stack(batch).to(vae.device)

            latents = vae.encode(batch).latent_dist.sample()
            latents = latents * 0.18215

            res_latents.append(latents.to("cpu"))

        return torch.cat(res_latents, dim=0)

    def __getitem__(self, index):
        example = {}
        image, prompt = self._prepare_sample(index, self.instance_images, self.instance_prompts)
        example["instance_images"] = image
        example["instance_prompt_ids"] = prompt

        if self.class_images is not None:
            image, prompt = self._prepare_sample(index, self.class_images, self.class_prompts)
            example["class_images"] = image
            example["class_prompt_ids"] = prompt

        return example


def collate_fn(examples, with_prior_preservation, tokenizer):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch
