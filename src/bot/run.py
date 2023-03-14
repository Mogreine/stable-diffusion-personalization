import functools
import os
import wandb
import requests
import pyrallis

from PIL import Image
from pathlib import Path
from loguru import logger
from typing import List, Tuple

from definitions import ROOT_DIR
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


@bot.on.message()
async def handle(message: Message):
    try:
        if message.text and len(message.attachments) > 0:
            if message.text == "male":
                cfg.gender = "male"
                prompts = man_prompts
            elif message.text == "female":
                cfg.gender = "female"
                prompts = girl_prompts
            else:
                logger.info(f"Wrong gender retrieved!")
                return "Wrong gender!"
            cfg.__post_init__()

            logger.info(f"Got message from: {message.from_id}")

            await message.answer("Training personalized model...")

            # Cleaning-up directories
            clean_directory(cfg.instance_data_folder)
            clean_directory(cfg.output_dir)

            # Downloading attached images
            urls = [attach.doc.url for attach in message.attachments]
            ims = [Image.open(requests.get(url, stream=True).raw) for url in urls]
            for i, im in enumerate(ims):
                path = f"{cfg.instance_data_folder}/im_{i}.png"
                im.save(path)

            logger.info("Training dreambooth...")
            image_sampler = train_dreambooth(cfg)

            logger.info("Generating images...")
            await message.answer("Generating images...")
            ims_paths = []
            for prompt in prompts:
                images = image_sampler(prompt)

                for i, im in enumerate(images):
                    im_path = f"{cfg.output_dir}im_{i}.png"
                    im.save(im_path)
                    ims_paths.append(im_path)

                    doc = await doc_uploader.upload(
                        f"im_{i}.png",
                        file_source=im_path,
                        peer_id=message.peer_id,
                    )
                    await message.answer(attachment=doc)

            logger.info("Images has been sent!")
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

    bot.run_forever()
