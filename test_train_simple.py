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
    ModelConfig(name="gpt2", default_lr=5e-5, eval_batch_size=32),
    # Add more configurations as needed
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
