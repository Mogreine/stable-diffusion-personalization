import os.path
from dataclasses import dataclass
from typing import Union, List, Optional
from pyrallis import field

from definitions import ROOT_DIR


@dataclass
class TrainConfig:
    """Config for training."""
    # Path to the model to fine-tune.
    model_path: str = field(default="runwayml/stable-diffusion-v1-5")
    # Path to the instance images.
    instance_data_folder: str = field(default="data/instance_images/kirill")
    # Prompt for the instance images.
    instance_prompt: str = field(default=None)
    # Path to the class images.
    class_data_folder: str = field(default=None)
    # Prompt for the class images.
    class_prompt: str = field(default=None)
    # Number of fine-tuning steps.
    n_steps: int = field(default=200)
    # Number of fine-tuning steps for CLIP.
    n_steps_clip: int = field(default=100)
    # Path to the output directory.
    output_dir: str = field(default=None)
    # Random seed.
    seed: int = field(default=123)
    # Use gradient checkpointing.
    gradient_checkpointing: bool = field(default=False)
    # Use 8-bit Adam.
    use_8bit_adam: bool = field(default=False)
    # Learning rate.
    lr: float = field(default=5e-6)
    # Weight decay.
    w_decay: float = field(default=0.01)
    # Adam epsilon.
    adam_epsilon: float = field(default=1e-8)
    # Adam beta1.
    adam_beta1: float = field(default=0.9)
    # Adam beta2.
    adam_beta2: float = field(default=0.999)
    # Precision type
    precision: str = field(default="fp16")
    # Batch size.
    batch_size: int = field(default=1)
    # Max gradient norm.
    max_grad_norm: float = field(default=1.0)
    # Prior loss weight.
    prior_loss_weight: float = field(default=1.0)
    # LR scheduler.
    lr_scheduler: str = field(default="constant")
    # LR scheduler warmup steps.
    n_warmup_steps: int = field(default=0)
    # Resolution.
    resolution: int = field(default=512)
    # Log images every n steps.
    log_images_every_n_steps: int = field(default=50)

    def __post_init__(self):
        assert self.instance_prompt is not None, "Instance prompt must be specified."
        assert self.class_prompt is not None, "Class prompt must be specified."
