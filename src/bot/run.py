import functools
import io
import os

import numpy as np

import wandb
import requests
import pyrallis

from PIL import Image
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Callable

from definitions import ROOT_DIR, GREETINGS_MESSAGE
from src.dreambooth.configs import TrainConfig
from src.dreambooth.train import DreamBoothPipeline
from src.utils import sample_images, set_seed, clean_directory


from vkbottle.bot import Bot, Message
from vkbottle.tools import DocMessagesUploader


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

    return read_lines_file(os.path.join(ROOT_DIR, "data/prompts_man.txt")), read_lines_file(
        os.path.join(ROOT_DIR, "data/prompts_girl.txt")
    )


bot = Bot(token=os.environ.get("VK_TOKEN"))
doc_uploader = DocMessagesUploader(bot.api)


def download_attached_images(message: Message) -> List[str]:
    urls = [attach.doc.url for attach in message.attachments]
    logger.info(f"{message.attachments}")
    logger.info(f"Found {len(urls)} urls of images")
    ims = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    paths = []

    for i, im in enumerate(ims):
        path = f"{cfg.instance_data_folder}/im_{i}.png"
        im.save(path)
        paths.append(path)

    return paths


async def generate_images(message: Message, image_sampler: Callable, prompts: List[str]):
    for prompt in prompts:
        images = image_sampler(prompt)

        for i, im in enumerate(images):
            buf = io.BytesIO()
            im.save(buf, format='png')
            byte_im = buf.getvalue()

            doc = await doc_uploader.upload(
                f"im_{np.random.randint(100, 100000)}.png",
                # file_source=im_path,
                file_source=byte_im,
                peer_id=message.peer_id,
            )
            await message.answer(attachment=doc)


def crop_and_save_image(paths: List[str]):
    for path in paths:
        im_pil = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
        im_pil.save(path)


@bot.on.message(func=lambda message: message.text.lower() == "правила")
async def rules_handler(message: Message):
    await message.answer(GREETINGS_MESSAGE)


IS_GENERATING = False


@bot.on.message()
async def generation_handler(message: Message):
    global IS_GENERATING
    try:
        if message.text and len(message.attachments) > 0:
            if IS_GENERATING:
                return "The bot is busy -- please try later"

            IS_GENERATING = True

            if message.text == "male":
                cfg.gender = "male"
                prompts = man_prompts
            elif message.text == "female":
                cfg.gender = "female"
                prompts = girl_prompts
            else:
                logger.info(f"Wrong gender retrieved!")
                return f"Wrong gender: {message.text}. Must be 'male' or 'female'."
            cfg.__post_init__()

            logger.info(f"Got message from: {message.from_id}")

            await message.answer("Training personalized model...")

            # Cleaning-up directories
            clean_directory(cfg.instance_data_folder)
            clean_directory(cfg.output_dir)

            # Downloading attached images
            paths = download_attached_images(message)

            # Crop images to 512x512
            crop_and_save_image(paths)

            logger.info("Training dreambooth...")
            image_sampler = train_dreambooth(cfg)

            logger.info("Generating images...")
            await message.answer("Generating images...")
            await generate_images(message, image_sampler, prompts)

            logger.info("Images has been sent!")
            IS_GENERATING = False
    except Exception as e:
        logger.info(e)
        return "Something went wrong, please, send another request"


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=TrainConfig)

    wandb.init(project="dreambooth", mode="offline")

    man_prompts, girl_prompts = load_prompts()
    cfg.instance_data_folder = os.path.join(ROOT_DIR, "data/input_bot/")
    cfg.output_dir = os.path.join(ROOT_DIR, "data/output_bot/")
    cfg.precalculate_latents = True
    Path(cfg.instance_data_folder).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded model: {cfg.model_path}")
    logger.info(f"Running with config:\n {cfg}")

    bot.run_forever()
