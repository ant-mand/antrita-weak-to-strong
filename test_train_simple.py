import json
import os
import argparse
import torch
from datasets import load_dataset, load_from_disk
import numpy as np
import fire

# Custom imports for model configuration and training, replace these with your actual modules
from weak_to_strong.train import ModelConfig, train_and_save_model
from weak_to_strong.datasets import VALID_DATASETS, tokenize_dataset
from weak_to_strong.common import get_tokenizer
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
import weak_to_strong.logger as logger

# Model configurations - You should replace this with your actual configuration
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

# Construct a dictionary from model configurations for easy access
MODELS_DICT = {config.name: config for config in MODEL_CONFIGS}

# Define available losses
loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

def main():
    parser = argparse.ArgumentParser(description="Train models on various datasets")
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of fixed training steps')
    parser.add_argument('--ds_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_size', type=str, required=True, help='Model size to use for training')
    args = parser.parse_args()

    assert args.ds_name in VALID_DATASETS, f"Dataset {args.ds_name} is not recognized. Valid datasets are: {VALID_DATASETS}"
    model_config = MODELS_DICT.get(args.model_size, None)
    assert model_config is not None, f"Model size {args.model_size} is not recognized."

    # Load dataset
    dataset = load_dataset(args.ds_name)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    # Tokenize datasets
    tokenizer = get_tokenizer(model_config.name)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_ctx=1024)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, max_ctx=1024)

    # Setup DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Configure logger, replace with your logging setup
    logger.configure_logging()

    # Start training loop
    for step in range(args.num_steps):
        try:
            # Assume each step processes one batch of data
            batch = next(iter(train_loader))
            loss = train_and_save_model(
                model_config=model_config, 
                batch=batch, 
                loss_fn=loss_dict['xent']  # Example using cross-entropy loss
            )
            if step % 100 == 0:
                print(f"Step {step}: Loss {loss}")
        except StopIteration:
            # If the dataset runs out of data, restart the DataLoader
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Training complete.")

if __name__ == "__main__":
    fire.Fire(main)
