import argparse
import functools
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import subprocess
import sys
import bitsandbytes as bnb
import pyrallis

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from loguru import logger

from src.dreambooth.datasets import PromptDataset, DreamBoothDataset, collate_fn
from src.dreambooth.configs import TrainConfig
from src.utils import convert2ckpt


def train(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    train_dataset: DreamBoothDataset,
    cfg: TrainConfig,
    is_train_text_encoder: bool,
):
    if not is_train_text_encoder:
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if is_train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        else:
            text_encoder.gradient_checkpointing_disable()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if is_train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.w_decay,
        eps=cfg.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    collate_fn_ = functools.partial(collate_fn, with_prior_preservation=is_train_text_encoder, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn_
    )

    total_batch_size = cfg.batch_size * accelerator.num_processes
    n_steps_per_epoch = len(train_dataloader)
    total_steps = cfg.n_steps_clip if is_train_text_encoder else cfg.n_steps
    n_epochs = math.ceil(total_steps / n_steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.n_warmup_steps,
        num_training_steps=total_steps,
    )

    if is_train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if cfg.precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    # if not cfg.train_text_encoder:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(cfg))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {n_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(n_epochs):
        unet.train()
        if is_train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if is_train_text_encoder:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + cfg.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if is_train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log(logs, step=global_step)

            if global_step >= total_steps:
                break

        accelerator.wait_for_everyone()


@pyrallis.wrap()
def main(cfg: TrainConfig):
    logging_dir = Path(cfg.output_dir, "logs/")
    set_seed(cfg.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=cfg.precision,
        logging_dir=logging_dir,
    )

    text_encoder = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder", use_auth_token=True)
    vae = AutoencoderKL.from_pretrained(cfg.model_path, subfolder="vae", use_auth_token=True)
    unet = UNet2DConditionModel.from_pretrained(cfg.model_path, subfolder="unet", use_auth_token=True)
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.model_path,
        subfolder="tokenizer",
    )

    # Auto-encoder is always frozen
    vae.requires_grad_(False)
    vae.eval()

    # First stage -- training clip text encoder + unet
    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.instance_data_folder,
        instance_prompt=cfg.instance_prompt,
        class_data_root=cfg.class_data_folder,
        class_prompt=cfg.class_prompt,
        tokenizer=tokenizer,
        size=cfg.resolution,
        center_crop=False,
    )
    train(accelerator, vae, unet, text_encoder, tokenizer, train_dataset, cfg, is_train_text_encoder=True)

    # Second stage -- training only unet
    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.instance_data_folder,
        instance_prompt=cfg.instance_prompt,
        class_data_root=None,
        tokenizer=tokenizer,
        size=cfg.resolution,
        center_crop=False,
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    train(accelerator, vae, unet, text_encoder, tokenizer, train_dataset, cfg, is_train_text_encoder=False)

    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        vae.to("cpu")
        unet = accelerator.unwrap_model(unet)
        unet.to("cpu")
        text_encoder = accelerator.unwrap_model(text_encoder)
        text_encoder.to("cpu")

        pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.model_path,
            unet=unet,
            text_encoder=text_encoder,
        )
        pipeline.save_pretrained(cfg.output_dir)
        convert2ckpt(cfg.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
