import functools
import itertools
import math
from pathlib import Path
import bitsandbytes as bnb
import lpips
import numpy as np
import pyrallis
import wandb

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
from src.utils import convert2ckpt, sample_images, read_photos_from_folder, pil2tensor


class DreamBoothPipeline:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.precision = torch.float16
        self.device = "cuda"

        # Loading models
        self.load_weights("vae")
        self.load_weights("unet")
        self.load_weights("text_encoder")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model_path,
            subfolder="tokenizer",
        )
        self.lpips = lpips.LPIPS(net="alex").to(self.device, dtype=self.precision)

        self.noise_scheduler = DDPMScheduler.from_config(cfg.model_path, subfolder="scheduler")

    def load_weights(self, model_name: str):
        if model_name == "vae":
            self.vae = AutoencoderKL.from_pretrained(self.cfg.model_path, subfolder="vae", use_auth_token=True)
            # self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            self._freeze_model(self.vae)
            self.vae.to(self.device)
        elif model_name == "unet":
            self.unet = UNet2DConditionModel.from_pretrained(self.cfg.model_path, subfolder="unet", use_auth_token=True)
            self.unet.to(self.device)
        elif model_name == "text_encoder":
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.cfg.model_path, subfolder="text_encoder", use_auth_token=True
            )
            self.text_encoder.to(self.device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _freeze_model(self, model):
        model.requires_grad_(False)
        model.eval()

    def _loss(self, noise, noise_pred, noise_prior=None, noise_prior_pred=None) -> torch.Tensor:
        # Compute instance loss
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

        # Compute prior loss
        if noise_prior_pred is not None and noise_prior is not None:
            prior_loss = F.mse_loss(noise_prior_pred.float(), noise_prior.float(), reduction="mean")
        else:
            prior_loss = 0

        # Add the prior loss to the instance loss.
        loss = loss + self.cfg.prior_loss_weight * prior_loss

        return loss

    def _train_step(self, batch):
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Convert images to latent space
        latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
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

    def _set_wise_lil_peep(self, set1, set2):
        res = np.mean([self.lpips(im, set2).mean().item() for im in set1])
        return res

    @torch.no_grad()
    def _log_images(self):
        # Generating images
        images_generated = sample_images(
            "a sks man", self.vae, self.unet, self.text_encoder, self.tokenizer
        )
        images_gt = read_photos_from_folder(self.cfg.instance_data_folder)

        # Converting PIL images to tensors
        images_generated_t = torch.stack([pil2tensor(im) for im in images_generated]).to(self.device)
        images_gt_t = torch.stack([pil2tensor(im) for im in images_gt]).to(self.device)

        # Calculating LPIPS
        lil_peep = self._set_wise_lil_peep(images_generated_t, images_gt_t)

        wandb.log({"images": [wandb.Image(im) for im in images_generated], "lpips": lil_peep})

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
            class_data_root=self.cfg.class_data_folder if self.cfg.use_prior_preservation else None,
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
            collate_fn, with_prior_preservation=self.cfg.use_prior_preservation, tokenizer=self.tokenizer
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

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(n_epochs):
            if train_unet:
                self.unet.train()
            if train_text_encoder:
                self.text_encoder.train()

            for step, batch in enumerate(train_dataloader):
                with torch.amp.autocast(device_type=self.device, dtype=self.precision):
                    noise, noise_pred = self._train_step(batch)
                    noise_prior, noise_prior_pred = None, None

                    if self.cfg.use_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_prior_pred = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    loss = self._loss(noise, noise_pred, noise_prior, noise_prior_pred)

                # Backward pass with loss scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": loss.detach().item()})

                # Logging
                wandb.log({"loss": loss.detach().item()})
                if global_step % self.cfg.log_images_every_n_steps == 0:
                    with torch.amp.autocast(device_type=self.device, dtype=self.precision):
                        self._log_images()

                if global_step >= n_steps:
                    break

    def save_sd(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_path,
            unet=self.unet,
            text_encoder=self.text_encoder,
            feature_extractor=None,
            safety_checker=None,
        )
        pipeline.save_pretrained(self.cfg.output_dir)
        convert2ckpt(self.cfg.output_dir)


@pyrallis.wrap()
def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    logging_dir = Path(cfg.output_dir, "logs/")
    wandb.init(project="dreambooth", config=cfg, dir=logging_dir)

    dreambooth_pipeline = DreamBoothPipeline(cfg)

    dreambooth_pipeline.train(300, train_text_encoder=True, train_unet=True)
    dreambooth_pipeline.load_weights("unet")
    dreambooth_pipeline.train(2000, train_text_encoder=False, train_unet=True)

    dreambooth_pipeline.save_sd()


if __name__ == "__main__":
    main()
