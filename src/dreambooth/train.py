import functools
import itertools
import math
from pathlib import Path
import bitsandbytes as bnb
import pyrallis

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from loguru import logger

from src.dreambooth.datasets import PromptDataset, DreamBoothDataset, collate_fn
from src.dreambooth.configs import TrainConfig
from src.utils import convert2ckpt


class DreamBoothPipeline:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # Loading and freezing VAE
        self.vae = AutoencoderKL.from_pretrained(cfg.model_path, subfolder="vae", use_auth_token=True)
        self._freeze_model(self.vae)

        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model_path, subfolder="text_encoder", use_auth_token=True)
        self.unet = UNet2DConditionModel.from_pretrained(cfg.model_path, subfolder="unet", use_auth_token=True)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model_path,
            subfolder="tokenizer",
        )

        self.noise_scheduler = DDPMScheduler.from_config(cfg.model_path, subfolder="scheduler")
        self.precision = torch.float32

    def _freeze_model(self, model):
        model.requires_grad_(False)
        model.eval()

    def _loss(self, noise, noise_pred, noise_prior=None, noise_prior_pred=None) -> torch.Tensor:
        # Compute instance loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

        # Compute prior loss
        if noise_prior_pred and noise_prior:
            prior_loss = F.mse_loss(noise_prior_pred.float(), noise_prior.float(), reduction="mean")
        else:
            prior_loss = 0

        # Add the prior loss to the instance loss.
        loss = loss + self.cfg.prior_loss_weight * prior_loss

        return loss

    def _train_step(self, batch):
        # Convert images to latent space
        latents = self.vae.encode(batch["pixel_values"].to(dtype=self.precision)).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        return noise, noise_pred

    def train(self, n_steps: int, train_text_encoder: bool, train_unet: bool):
        # Enabling gradient checkpoint
        if self.cfg.gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.gradient_checkpointing_enable()

        # Freezing models
        if not train_text_encoder:
            self.text_encoder.gradient_checkpointing_disable()
            self._freeze_model(self.text_encoder)
        if not train_unet:
            self.unet.gradient_checkpointing_disable()
            self._freeze_model(self.unet)

        train_dataset = DreamBoothDataset(
            instance_data_root=self.cfg.instance_data_folder,
            instance_prompt=self.cfg.instance_prompt,
            class_data_root=self.cfg.class_data_folder if train_text_encoder else None,
            class_prompt=self.cfg.class_prompt,
            tokenizer=self.tokenizer,
            size=self.cfg.resolution,
            center_crop=False,
        )

        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
            if train_text_encoder
            else self.unet.parameters()
        )
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.w_decay,
            eps=self.cfg.adam_epsilon,
        )

        collate_fn_ = functools.partial(
            collate_fn, with_prior_preservation=train_text_encoder, tokenizer=self.tokenizer
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate_fn_
        )

        n_steps_per_epoch = len(train_dataloader)
        n_epochs = math.ceil(n_steps / n_steps_per_epoch)

        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.n_warmup_steps,
            num_training_steps=n_steps,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {n_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.cfg.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.cfg.batch_size}")
        logger.info(f"  Total optimization steps = {n_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(n_steps))
        global_step = 0

        for epoch in range(n_epochs):
            if train_unet:
                self.unet.train()
            if train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(train_dataloader):
                noise, noise_pred = self._train_step(batch)
                noise_prior, noise_prior_pred = None, None

                if train_text_encoder:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_prior_pred = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                loss = self._loss(noise, noise_pred, noise_prior, noise_prior_pred)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": loss.detach().item()})

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

                if global_step >= n_steps:
                    break

    def save_sd(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_path,
            unet=self.unet,
            text_encoder=self.text_encoder,
        )
        pipeline.save_pretrained(self.cfg.output_dir)
        convert2ckpt(self.cfg.output_dir)


@pyrallis.wrap()
def main(cfg: TrainConfig):
    logging_dir = Path(cfg.output_dir, "logs/")
    set_seed(cfg.seed)

    dreambooth_pipeline = DreamBoothPipeline(cfg)

    dreambooth_pipeline.train(350, train_text_encoder=True, train_unet=True)
    dreambooth_pipeline.train(1000, train_text_encoder=False, train_unet=True)

    dreambooth_pipeline.save_sd()


if __name__ == "__main__":
    main()
