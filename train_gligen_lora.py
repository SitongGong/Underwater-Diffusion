#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from functools import partial
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

import accelerate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor

from mmdet.datasets import build_dataset
from mmengine.config import Config
from torch.utils.data import ConcatDataset, RandomSampler
from einops import rearrange
from diffusers.training_utils import EMAModel

import diffusers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion_gligen import StableDiffusionGLIGENPipeline
# from diffusers.pipelines.stable
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler, 
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")
logger = get_logger(__name__)


def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# {'SDXL' if 'playground' not in base_model else 'Playground'} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/18515601223/Gligen",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help=
        "Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=
        ("Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
         " `args.validation_prompt` multiple times: `args.num_validation_images`."
         ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A real underwater image",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--do_edm_style_training",
        default=False,
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--cfg_name",
        type=str,
        default='/18515601223/UnderwaterDiffusion/dataset_config/mix_dataset.py',
        help="dataset cfg",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/18515601223/UnderwaterDiffusion/model_weight",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
        help="Flag to additionally generate final state dict in the Kohya format so that it becomes compatible with A111, Comfy, Kohya, etc.",
    )
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        default=False,
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=1, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        default=False,
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_max_obj", default=300, type=int, help="Max objects per image.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--use_ema",
                        action="store_true",
                        help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help=
        "Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        default=False,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--fg_reweight",
        type=float,
        default=1.0,
        help="Scale of foreground object loss",
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # 日志加载以及分布式训练设置
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,           # 进行多次正向传播后再反传一次
        mixed_precision=args.mixed_precision,            # 训练精度
        log_with=args.report_to,          # 保存日志的训练平台
        project_config=accelerator_project_config,       # 日志记录路径
        kwargs_handlers=[kwargs],
    )
    
    # 随机数生成器
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision)

    # Load noise scheduler and models
    noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        # use_dora=args.use_dora,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    
    feature_extractor = CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, 
                                                             subfolder="feature_extractor", 
                                                             revision=args.revision, 
                                                             variant=args.variant)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(args.pretrained_model_name_or_path, 
                                                                  subfolder="safety_checker", 
                                                                  revision=args.revision, 
                                                                  variant=args.variant)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:      # 是否训练text encoder
        text_lora_config = LoraConfig(
            r=args.rank,
            # use_dora=args.use_dora,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(),
                            model_cls=UNet2DConditionModel,
                            model_config=unet.config)

    def unwrap_model(model):     # 对模型进行解包操作，从而访问模型的原始版本
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:      # 这里我们仅对unet进行训练
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionGLIGENPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:           # 保存lora信息
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # 学习率调整策略
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder])

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # 需要更新的仅有unet中lora层中的参数
    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # # different learning rate for text encoder and unet
        # text_lora_parameters_one_with_lr = {
        #     "params": text_lora_parameters_one,
        #     "weight_decay": args.adam_weight_decay_text_encoder,
        #     "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        # }
        # text_lora_parameters_two_with_lr = {
        #     "params": text_lora_parameters_two,
        #     "weight_decay": args.adam_weight_decay_text_encoder,
        #     "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        # }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,        # 确定需要更新的参数仅有une的lora层
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    # processing the batch_data returned by the _get_item_ function of the Dataset
    def collate_fn(batch, num_max_obj):
        captions = []
        image_list = []
        bboxes_list = []
        class_text = []
        mask_list = []
        for bs in batch:
            prompt = ' An underwater image incorporating '
            data_class = bs['img_metas'].data['filename'].split('/')
            if 'AUDD' in data_class:
                    CLASSES = ["seacucumber", "seaurchin", "scallop"]
            elif 'DUO' in data_class:
                    CLASSES = ["holothurian", "echinus", "scallop", "starfish"]
            elif 'UIIS' in data_class:
                    CLASSES = ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins','human divers', 'robots', 'sea-floor'] 
            elif 'DUT-USEG' in data_class:
                    CLASSES = ["holothurian", "echinus", "scallop", "starfish"]
            else:
                    raise ValueError("You load the wrong dataset!")
            
            class_list = bs['gt_labels'].data.numpy().tolist()
            class_set = list(set(class_list))    # get the class text to construct the caption
            for class_idx in class_set:
                    if class_idx == class_set[-1]:
                            prompt = prompt + CLASSES[class_idx] + '. '
                    else:
                            prompt = prompt + CLASSES[class_idx] + ' and '
            captions.append(prompt)
            bboxes = torch.zeros((num_max_obj, 4))
            bboxes[:len(bs['gt_bboxes'].data)] = bs['gt_bboxes'].data       # n x 4
            box_class_text = [CLASSES[class_idx] for class_idx in class_list]     # n x category_name
            mask = [1 for _ in class_list]
            
            for _ in range(num_max_obj-len(class_list)):
                box_class_text.append("")
                mask.append(0)
            
            class_text.append(box_class_text)
            image_list.append(bs['img'].data)      # 3, h, w
            bboxes_list.append(bboxes)
            mask_list.append(torch.tensor(mask))
                
        return {
                'prompts': captions,
                'pixel_values': torch.stack(image_list, dim=0),      # bs, 3, h, w
                'bboxes': torch.stack(bboxes_list, dim=0),      # bs, n, 4
                'class_text': class_text, 
                'masks': torch.stack(mask_list, dim=0),       # bs, n
        }
    
    cfg = Config.fromfile(args.cfg_name)
    train_dataset = build_dataset(cfg.data.train)
    # print(train_dataset.data)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=args.per_gpu_batch_size,
                                num_workers=args.dataloader_num_workers,
                                collate_fn=partial(collate_fn, num_max_obj=args.num_max_obj), 
                                pin_memory=True)

    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    def compute_time_ids(original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer]
        text_encoders = [text_encoder]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)         # 每个epoch中梯度更新的次数
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch         # 重新计算训练step的次数
        overrode_max_train_steps = True

     # 学习率调整策略
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = ("gligen-lora-fine-tune")
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # 对于各部分数据的训练过程，使用gligen结构
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            # 对于每一步迭代过程
            with accelerator.accumulate(unet):
                # 对于图像的处理
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                pixel_values = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                # latents = tensor_to_vae_latent(pixel_values, vae)
                latents = vae.encode(pixel_values).latent_dist.sample()
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(     # 为batch中的每一张图像随机生成一个加噪时间步长
                    0,
                    noise_scheduler.config.num_train_timesteps, (bsz, ),
                    device=latents.device)
                timesteps = timesteps.long()   
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)       # 输入图像加噪

                # Get the text embedding for conditioning
                # tokenizer
                inputs = tokenizer(batch['prompts'],
                                   max_length=tokenizer.model_max_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")
                
                encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]
                
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz,
                                          device=latents.device,
                                          generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    inputs = tokenizer([""], 
                                      max_length=tokenizer.model_max_length,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
                    null_conditioning = text_encoder(inputs.input_ids.to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                bboxes_prompt = batch['bboxes']       # bs, n, dim
                fixed_num_objs = bboxes_prompt.shape[-2]
                text_embeddings = []
                for class_text in batch['class_text']:
                    # print(class_text)
                    token_ids = tokenizer(class_text,
                                          truncation=True,
                                          padding="max_length",
                                          max_length=tokenizer.model_max_length,
                                          return_tensors="pt",
                                        ).input_ids.to(accelerator.device)
                    _text_embeddings = text_encoder(token_ids).pooler_output       # 选择池化到相同长度的embedding
                    _text_embeddings = _text_embeddings.view(fixed_num_objs, -1)      # n, dim
                    text_embeddings.append(_text_embeddings)
                text_embeddings = torch.stack(text_embeddings, dim=0)      # bs, n, dim

                masks = batch['masks']
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz,
                                          device=latents.device,
                                          generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1)
                    # Final text conditioning.
                    null_masks = torch.zeros_like(masks)
                    masks = torch.where(prompt_mask, null_masks, masks)

                # print(masks.shape)

                cross_attention_kwargs = {}
                cross_attention_kwargs["gligen"] = {
                    "boxes": bboxes_prompt,
                    "positive_embeddings": text_embeddings,
                    "masks": masks
                }
                cross_attention_kwargs["fixed_objs"] = fixed_num_objs

                # print(noisy_latents.shape, timesteps.shape, encoder_hidden_states.shape)
                # print(bboxes_prompt.shape, text_embeddings.shape, masks.shape)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Convert normalized bboxes to actual coordinates
                bboxes_actual = batch['bboxes'].clone().float()
                bboxes_actual[..., [0, 2]] *= noisy_latents.size(3)  # x dimension
                bboxes_actual[..., [1, 3]] *= noisy_latents.size(2)  # y dimension
                bboxes_actual = bboxes_actual.long()  # Convert to long type for indexing

                # print(bboxes_actual.shape)

                # Step 1: Initialize the weights tensor with all ones
                re_weights = torch.ones_like(noisy_latents)
                # Step 2: Set higher weights for areas indicated by bboxes
                for i in range(bboxes_prompt.shape[0]):      # bs
                    for j in range(fixed_num_objs):  # ins index
                        x1, y1, x2, y2 = bboxes_actual[i, j]
                        re_weights[i, :, y1:y2, x1:x2] = args.fg_reweight
                # Normalize the weights matrix so that its mean is 1
                re_weights = re_weights / re_weights.mean()
                loss = F.mse_loss(model_pred.float(),
                                  target.float(),
                                  reduction="none")
                loss = loss * re_weights
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                    # sample images!
                    if ((args.validation_prompt is not None) and (global_step % args.validation_steps == 0)):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}.")
                        # create pipeline 
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            vae=accelerator.unwrap_model(vae),
                            safety_checker=safety_checker,
                            feature_extractor=feature_extractor,
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)
                            
                        edited_images = []
                        with torch.autocast(str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"):
                            boxes = batch['bboxes'][0]      # n, 4
                            class_text = batch['class_text'][0]
                            # video_masks = batch['mask']
                            args.validation_prompt = batch['prompts'][0]
                            for val_img_idx in range(args.num_validation_images):
                                image = pipeline(
                                    args.validation_prompt,
                                    gligen_phrases=class_text,
                                    width=args.width,
                                    height=args.height,
                                    gligen_boxes=boxes,
                                    num_inference_steps=50,
                                    guidance_scale=7,
                                    generator=generator,
                                    
                                )
                                # print(image.images[0])
                                out_file = os.path.join(val_save_dir, f"step_{global_step}_val_img_{val_img_idx}.jpg")
                                frame_size = (args.width, args.height)
                                img = image.images[0]
                                draw = ImageDraw.Draw(img)
                                for bbox in batch['bboxes'][0].cpu().numpy():
                                    x1, y1, x2, y2 = bbox
                                    top_left = (x1 * frame_size[0], y1 * frame_size[1])
                                    bottom_right = (x2 * frame_size[0], y2 * frame_size[1])
                                    draw.rectangle([top_left, bottom_right], outline="red", width=1)
                                    
                                img.save(out_file)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # if accelerator.is_main_process:     # 判断当前是否为模型主进程
        #     if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
        #         # create pipeline
        #         pipeline = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        #             args.pretrained_model_name_or_path,
        #             vae=vae,
        #             text_encoder=accelerator.unwrap_model(text_encoder),
        #             unet=accelerator.unwrap_model(unet),
        #             revision=args.revision,
        #             variant=args.variant,
        #             torch_dtype=weight_dtype,
        #         )
        #         pipeline_args = {"prompt": args.validation_prompt}

        #         images = log_validation(
        #             pipeline,
        #             args,
        #             accelerator,
        #             pipeline_args,
        #             epoch,
        #         )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
            
        # save lora weights
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        StableDiffusionGLIGENPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
        )
        if args.output_kohya_format:
            lora_state_dict = load_file(f"{args.output_dir}/pytorch_lora_weights.safetensors")
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            save_file(kohya_state_dict, f"{args.output_dir}/pytorch_lora_weights_kohya.safetensors")

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionGLIGENPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # load attention processors
        pipeline.save_pretrained(args.output_dir)

        # run inference
        # images = []
        # if args.validation_prompt and args.num_validation_images > 0:
        #     pipeline_args = {"prompt": args.validation_prompt, "num_inference_steps": 25}
        #     images = log_validation(
        #         pipeline,
        #         args,
        #         accelerator,
        #         pipeline_args,
        #         epoch,
        #         is_final_validation=True,
        #     )

        if args.push_to_hub:
            # save_model_card(
            #     repo_id,
            #     use_dora=args.use_dora,
            #     images=image,
            #     base_model=args.pretrained_model_name_or_path,
            #     train_text_encoder=args.train_text_encoder,
            #     instance_prompt=args.instance_prompt,
            #     validation_prompt=args.validation_prompt,
            #     repo_folder=args.output_dir,
            #     vae_path=args.pretrained_vae_model_name_or_path,
            # )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
