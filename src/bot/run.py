import asyncio
import functools
import io
import multiprocessing
import os
from collections import defaultdict
from multiprocessing import Process
import multiprocessing as mp
from copy import deepcopy

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


bot = Bot(token=os.environ.get("VK_TOKEN2"))
doc_uploader = DocMessagesUploader(bot.api, generate_attachment_strings=True)
IS_GENERATING = False
register_heif_opener()


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


def download_attached_images(message: Message, dir_path) -> List[str]:
    urls = [attach.doc.url for attach in message.attachments]
    logger.info(f"{message.attachments}")
    logger.info(f"Found {len(urls)} urls of images")
    ims = [Image.open(requests.get(url, stream=True).raw) for url in urls]

    paths = []
    for i, im in enumerate(ims):
        path = os.path.join(dir_path, f"im_{i}.png")
        im.save(path)
        paths.append(path)

    return paths


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
            print("Failed to delete %s. Reason: %s" % (file_path, e))


async def generate_images(message: Message, image_sampler: Callable, prompts: List[str]):
    delete_files("./data/output_bot")
    for i, prompt in enumerate(prompts):
        images = image_sampler(prompt)

        for j, im in enumerate(images):
            im.save(f"./data/output_bot/{i}_{j}.png")

    await upload_archive(message)


def generate_images2(image_sampler: Callable, prompts: List[str]):
    delete_files("./data/output_bot")
    res = []
    for i, prompt in enumerate(prompts):
        images = image_sampler(prompt)

        for j, im in enumerate(images):
            res.append(im)
            im.save(f"./data/output_bot/{i}_{j}.png")

    return res


import zipfile


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, "..")))


async def upload_archive(message: Message):
    archive_path = "./data/test_archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir("./data/output_bot", zipf)

    doc = await doc_uploader.upload(
        f"im_{np.random.randint(100, 100000)}.zip",
        file_source=archive_path,
        peer_id=message.peer_id,
    )
    await message.answer(attachment=doc)


async def crop_and_save_image(message, paths: List[str]):
    res = []
    for path in paths:
        im = skio.imread(path)
        im_pil = Image.open(path).convert("RGB")
        box = get_face_box(im)
        # await message.answer(box)
        im_cropped, box = crop_face(im_pil, *box)
        # await message.answer(box)
        im_cropped.save(path)
        res.append(im_cropped)
    return res


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


def generation_handler(cfg):
    try:
        logger.info("Training dreambooth...")
        image_sampler = train_dreambooth(cfg)

        logger.info("Generating images...")
        man_prompts, girl_prompts = load_prompts()
        prompts = man_prompts if cfg.gender == "male" else girl_prompts
        imgs = generate_images2(image_sampler, prompts)
        logger.info("Images has been sent!")
        return imgs
    except Exception as e:
        logger.info(e)
        # await message.answer("Something went wrong, please, send another request")


async def download_crop_images(message: Message):
    logger.info(f"Got message from: {message.from_id}")

    dir_path = os.path.join(ROOT_DIR, f"data/input_bot/{message.peer_id}_{np.random.randint(100, 100000)}")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    clean_directory(dir_path)

    # Downloading attached images
    paths = download_attached_images(message, dir_path)

    # Crop images to 512x512
    await crop_and_save_image(message, paths)

    return dir_path


@bot.on.message()
async def task_creator(message: Message):
    dir_path = await download_crop_images(message)
    logger.info(f"From {message.peer_id} got gender: {message.text}")
    logger.info(f"Saved images to: {dir_path}")
    input_queue.put_nowait({"peer_id": message.peer_id, "gender": message.text, "ims_dir": dir_path})


def input_worker(queue_in: multiprocessing.Queue, queue_out: multiprocessing.Queue, cfg):
    logger.info(f"darova gpu: {cfg.device}")
    worker_id = cfg.device[-1]
    # cfg.instance_data_folder = os.path.join(ROOT_DIR, f"data/input_bot/{worker_id}")
    cfg.output_dir = os.path.join(ROOT_DIR, f"data/output_bot/{worker_id}")
    cfg.precalculate_latents = True
    # Path(cfg.instance_data_folder).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    while True:
        task = queue_in.get()
        logger.info(f"Worker{worker_id} got: {task}")

        cfg.instance_data_folder = task["ims_dir"]
        cfg.gender = task["gender"]
        cfg.__post_init__()
        ims = generation_handler(cfg)

        for im in ims:
            queue_out.put({"peer_id": task["peer_id"], "image": im})


async def upload_archive2(peer_id):
    archive_path = "./data/test_archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir("./data/output_bot", zipf)

    doc = await doc_uploader.upload(
        f"im_{np.random.randint(100, 100000)}.zip",
        file_source=archive_path,
        peer_id=peer_id,
    )
    return doc


async def upload_image(peer_id, im: Image):
    buf = io.BytesIO()
    im.save(buf, format="png")
    byte_im = buf.getvalue()

    doc = await doc_uploader.upload(
        f"im_{np.random.randint(100, 100000)}.png",
        file_source=byte_im,
        peer_id=peer_id,
    )
    return doc


@bot.loop_wrapper.interval(seconds=5)
async def output_worker():
    docs = defaultdict(list)
    while True:
        try:
            task = output_queue.get_nowait()
            peer_id = task["peer_id"]
            im = task["image"]
            logger.info(f"Output worker got result with id: {peer_id}")
            doc = await upload_image(peer_id, im)
            logger.debug(f"doc: {doc}")
            docs[peer_id].append(doc)
        except Exception as e:
            logger.error(f"error: {e}")
            break

    for peer_id, docs in docs.items():
        attachments = ",".join(docs)
        stream = io.StringIO("Generation is over!")
        msg = stream.read(4096)
        await bot.api.messages.send(
            message=msg, peer_ids=[peer_id], attachment=attachments, random_id=np.random.randint(100, 100000)
        )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    cfg = pyrallis.parse(config_class=TrainConfig)

    wandb.init(project="dreambooth", mode="offline")

    logger.info(f"Loaded model: {cfg.model_path}")
    logger.info(f"Running with config:\n {cfg}")

    cfg.device = "cuda:6"
    consumer_process1 = Process(target=input_worker, args=(input_queue, output_queue, deepcopy(cfg)))
    cfg.device = "cuda:7"
    consumer_process2 = Process(target=input_worker, args=(input_queue, output_queue, deepcopy(cfg)))
    consumer_process1.start()
    consumer_process2.start()

    bot.run_forever()

    consumer_process1.join()
    consumer_process2.join()
