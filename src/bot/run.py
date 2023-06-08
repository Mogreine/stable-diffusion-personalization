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
from face_detection import RetinaFace
from skimage import io as skio
from pillow_heif import register_heif_opener

from definitions import ROOT_DIR, GREETINGS_MESSAGE
from src.dreambooth.configs import TrainConfig
from src.dreambooth.train import DreamBoothPipeline
from src.utils import sample_images, set_seed, clean_directory


from vkbottle.bot import Bot, Message
from vkbottle.tools import DocMessagesUploader


bot = Bot(token=os.environ.get("VK_TOKEN"))
doc_uploader = DocMessagesUploader(bot.api)
IS_GENERATING = False
register_heif_opener()


def train_dreambooth(cfg: TrainConfig):
    set_seed(cfg.seed)
    dreambooth_pipeline = DreamBoothPipeline(cfg)

    # dreambooth_pipeline.train(300, train_text_encoder=True, train_unet=True)
    # dreambooth_pipeline.load_weights("unet")
    # dreambooth_pipeline.train(train_text_encoder=False, train_unet=True)

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


# async def generate_images(message: Message, image_sampler: Callable, prompts: List[str]):
#     for prompt in prompts:
#         images = image_sampler(prompt)
#
#         for i, im in enumerate(images):
#             buf = io.BytesIO()
#             im.save(buf, format="png")
#             byte_im = buf.getvalue()
#
#             doc = await doc_uploader.upload(
#                 f"im_{np.random.randint(100, 100000)}.png",
#                 # file_source=im_path,
#                 file_source=byte_im,
#                 peer_id=message.peer_id,
#             )
#             await message.answer(attachment=doc)


import os, shutil

def delete_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

async def generate_images(message: Message, image_sampler: Callable, prompts: List[str]):
    delete_files("./data/output_bot")
    for i, prompt in enumerate(prompts):
        images = image_sampler(prompt)

        for j, im in enumerate(images):
            im.save(f"./data/output_bot/{i}_{j}.png")

    await upload_archive(message)


import zipfile


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

async def upload_archive(message: Message):
    archive_path = "./data/test_archive.zip"
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('./data/output_bot', zipf)

    doc = await doc_uploader.upload(
        f"im_{np.random.randint(100, 100000)}.zip",
        file_source=archive_path,
        peer_id=message.peer_id,
    )
    await message.answer(attachment=doc)


async def crop_and_save_image(message, paths: List[str]):
    for path in paths:
        im = skio.imread(path)
        im_pil = Image.open(path).convert("RGB")
        box = get_face_box(im)
        await message.answer(box)
        im_cropped, box = crop_face(im_pil, *box)
        await message.answer(box)
        im_cropped.save(path)


def get_face_box(img):
    detector = RetinaFace(gpu_id=0)
    faces = detector(img)
    box, landmarks, score = faces[0]

    return box


def crop_face(img: Image.Image, lt_x: float, lt_y: float, br_x: float, br_y: float, crop_dim: int = 512) -> Image.Image:
    center_x = (lt_x + br_x) / 2
    center_y = (lt_y + br_y) / 2

    width, height = img.size
    new_side_size = min(width, height)
    half_crop = new_side_size // 2

    lt_x_new, lt_y_new = center_x - half_crop, center_y - half_crop
    rb_x_new, rb_y_new = center_x + half_crop, center_y + half_crop

    lt_x_new += min(0, width - rb_x_new)
    rb_x_new += abs(min(0, lt_x_new))
    lt_y_new += min(0, height - rb_y_new)
    rb_y_new += abs(min(0, lt_y_new))

    lt_x_new, lt_y_new = max(0, lt_x_new), max(0, lt_y_new)
    rb_x_new, rb_y_new = min(width, rb_x_new), min(height, rb_y_new)

    return img.crop((lt_x_new, lt_y_new, rb_x_new, rb_y_new)).resize((crop_dim, crop_dim)), (
        lt_x_new,
        lt_y_new,
        rb_x_new,
        rb_y_new,
    )


@bot.on.message(func=lambda message: message.text.lower() == "правила")
async def rules_handler(message: Message):
    await message.answer(GREETINGS_MESSAGE)


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
                IS_GENERATING = False
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
            await crop_and_save_image(message, paths)

            ###############################
            for im_path in paths:
                doc = await doc_uploader.upload(
                    f"im_{np.random.randint(100, 100000)}.png",
                    file_source=im_path,
                    peer_id=message.peer_id,
                )
                await message.answer(attachment=doc)
            ###############################

            logger.info("Training dreambooth...")
            image_sampler = train_dreambooth(cfg)

            logger.info("Generating images...")
            await message.answer("Generating images...")
            await generate_images(message, image_sampler, prompts)

            logger.info("Images has been sent!")
            IS_GENERATING = False
    except Exception as e:
        logger.info(e)
        IS_GENERATING = False
        await message.answer("Something went wrong, please, send another request")


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
