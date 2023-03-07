import argparse
import functools
import os
from pathlib import Path

from random import randint

import wandb
from PIL import Image
import requests
import PIL.Image
import pyrallis
from diffusers import StableDiffusionPipeline
from loguru import logger
from typing import List, Tuple

import vk_api
from vk_api.longpoll import VkEventType, VkLongPoll

from definitions import TOKEN
from src.dreambooth.configs import TrainConfig
from src.dreambooth.train import DreamBoothPipeline
from src.utils import sample_images, set_seed, clean_directory


def upload_photo(api, uid: int, files: List[str]):
    upload = vk_api.VkUpload(api)
    photos = upload.photo_messages(files)

    for i in range(len(photos)):
        attachment = "photo" + str(photos[i]["owner_id"]) + "_" + str(photos[i]["id"])

        res = api.messages.send(user_id=uid, attachment=attachment, random_id=randint(100, 200000))

        logger.info(res)

        if i != len(photos) - 1:
            attachment += ","


def generate_and_save_images(prompt: str, pipe: StableDiffusionPipeline, args: argparse.ArgumentParser) -> List[str]:
    base_count = len(os.listdir(args.outdir))
    images = pipe(
        prompt, height=args.H, width=args.W, num_images_per_prompt=args.n_samples, num_inference_steps=args.num_steps
    ).images
    image_paths = []

    for image in images:
        image_path = os.path.join(args.outdir, f"grid-{base_count:04}.png")
        image.save(image_path)
        base_count += 1
        image_paths.append(image_path)

    return image_paths


def train_dreambooth(cfg: TrainConfig):
    set_seed(cfg.seed)
    dreambooth_pipeline = DreamBoothPipeline(cfg)

    dreambooth_pipeline.train(300, train_text_encoder=True, train_unet=True)
    dreambooth_pipeline.load_weights("unet")
    dreambooth_pipeline.train(train_text_encoder=False, train_unet=True)

    return functools.partial(
        sample_images,
        vae=dreambooth_pipeline.vae,
        unet=dreambooth_pipeline.unet,
        text_encoder=dreambooth_pipeline.text_encoder,
        tokenizer=dreambooth_pipeline.tokenizer,
    )


def load_prompts() -> Tuple[List[str], List[str]]:
    def read_lines_file(path):
        with open(path, "r") as f:
            return f.readlines()

    return read_lines_file("data/prompts_man.txt"), read_lines_file("data/prompts_girl.txt")


@pyrallis.wrap()
def main(cfg: TrainConfig):
    # def main():
    # Connecting to the vk api
    wandb.init(project="dreambooth", mode="offline")
    vk_session = vk_api.VkApi(token=TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)

    man_prompts, girl_prompts = load_prompts()
    cfg.instance_data_folder = "data/input_bot/"
    cfg.output_dir = "data/output_bot/"
    cfg.precalculate_latents = True
    Path(cfg.instance_data_folder).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded model: {cfg.model_path}")
    logger.info("Start listening server...")
    while True:
        try:
            for event in longpoll.listen():
                if (
                    event.type == VkEventType.MESSAGE_NEW
                    and event.to_me
                    and event.text
                    and len(event.attachments) // 2 > 0
                ):
                    if event.text == "male":
                        cfg.gender = "male"
                        prompts = man_prompts
                    elif event.text == "female":
                        cfg.gender = "female"
                        prompts = girl_prompts
                    else:
                        logger.info(f"Wrong gender retrieved!")
                        continue
                    cfg.__post_init__()

                    logger.info(f"Got message from: {event.user_id}")
                    logger.info(f"Prompt text: {event.text}")

                    # Cleaning-up directories
                    clean_directory(cfg.instance_data_folder)
                    clean_directory(cfg.output_dir)

                    # Downloading attached images
                    num_ims = len(event.attachments) // 2
                    urls = vk.messages.getHistoryAttachments(
                        peer_id=event.peer_id, media_type="doc", preserve_order=True
                    )
                    urls = [item["attachment"]["doc"]["url"] for item in urls["items"][:num_ims]]
                    ims = [Image.open(requests.get(url, stream=True).raw) for url in urls]
                    for i, im in enumerate(ims):
                        im.save(f"{cfg.instance_data_folder}/im_{i}.png")

                    logger.info("Training dreambooth...")

                    image_sampler = train_dreambooth(cfg)

                    logger.info("Generating images...")
                    for prompt in prompts:
                        images = image_sampler(prompt)

                        ims_paths = []
                        for i, im in enumerate(images):
                            im_path = f"{cfg.output_dir}im_{i}.png"
                            im.save(im_path)
                            ims_paths.append(im_path)

                        # Need to send images somehow
                        upload_photo(vk, event.user_id, ims_paths)

                    logger.info("Image has been sent!")
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
