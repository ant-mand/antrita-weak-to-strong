import json
import os
import random
import subprocess
from typing import Dict, List, Optional

import fire
import numpy as np
import torch
from datasets import load_dataset, load_from_disk

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import VALID_DATASETS, load_dataset, tokenize_dataset
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-medium",
        default_lr=5e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-large",
        default_lr=1e-5,
        eval_batch_size=32,
    ),
    ModelConfig(
        name="gpt2-xl",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        # Should use model_parallel on V100s (note: ironically if you have a single V100 it should run,
        # but if you have multiple it won't run without model_parallel because of the overhead of data
        # parallel training).
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
    ),
    ModelConfig(
        name="Qwen/Qwen-1_8B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=(
            torch.cuda.get_device_properties(0).total_memory < 35e9
            and torch.cuda.device_count() > 1
        ),
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "5fde88dff770a7d036847211f5d9d9705f0caa69",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-7B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "d4efd21e866b9cb3466cb65b963933f5e98016d1",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-14B",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this bf16 support and without many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "8be2854218fea9054331e217fd26a06f3fd02004",
        },
    ),
    ModelConfig(
        name="Qwen/Qwen-72B",
        default_lr=1e-5,
        eval_batch_size=1,
        gradient_checkpointing=True,
        model_parallel=True,
        # note: you will probably not be able to run this without bf16 support and many gpus
        custom_kwargs={
            "trust_remote_code": True,
            "bf16": torch.cuda.is_bf16_supported(),
            "fp32": not torch.cuda.is_bf16_supported(),
            "revision": "fec78c0e3b3b10dd9f0ce775c34a686a3255a7d1",
        },
        # This model is really big, save space by using adafactor.
        # Note that even then it will take up ~60GB per GPU on an 8-GPU machine.
        default_optimizer="adafactor",
    ),
]

MODELS_DICT: Dict[str, ModelConfig] = {model_config.name: model_config for model_config in MODEL_CONFIGS}
loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}
VALID_LOSSES: List[str] = list(loss_dict.keys())

def main(
    num_steps: int,  # Number of fixed steps for training
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "boolq",
    loss: str = "xent",
    model_size: str = "gpt2",
    lr: Optional[float] = None,
    epochs: int = 1,  # Adjusted to manage via num_steps
    results_folder: str = "/tmp/results",
    # Other parameters...
):
    model_config = MODELS_DICT[model_size]
    if lr is None:
        lr = model_config.default_lr

    # Load and prepare data
    dataset = load_dataset(ds_name, split='train')
    train_dataset, test_ds = dataset.train_test_split(test_size=0.5).values()

    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_ctx)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)

    # Create data loader with proper batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)

    # Train model
    model = model_config.load_model()
    for step in range(num_steps):
        batch = next(iter(train_loader))  # Get a batch from the dataset
        loss = train_and_save_model(model, batch, lr)  # Placeholder function

        if step % 100 == 0:  # Example condition for logging
            print(f"Step {step}/{num_steps}: Loss = {loss}")

    # Save results
    # Additional code for saving model, logging results, etc.

if __name__ == "__main__":
    fire.Fire(main)
