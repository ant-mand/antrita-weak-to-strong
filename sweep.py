import os
import subprocess
import sys
from typing import List, Union

import fire


def main(model_sizes: Union[List[str], str], **kwargs):
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(",")
    assert (
        "weak_model_size" not in kwargs
        and "model_size" not in kwargs
        and "weak_labels_path" not in kwargs
    ), "Need to use model_sizes when using sweep.py"

    # hyperparameter_grid = {
    #         'learning_rate': [1e-5, 1e-7],
    #         'batch_size': [8, 16],
    #         'epochs': [4, 5]
    #     }

    basic_args = [sys.executable, os.path.join(os.path.dirname(__file__), "train_simple.py")]
    for key, value in kwargs.items():
        basic_args.extend([f"--{key}", str(value)])

    print("Running ground truth models")
    for model_size in model_sizes:
        subprocess.run(basic_args + ["--model_size", model_size], check=True)

    # # Grid search for ground truth models
    # print("Running ground truth models")
    # for model_size in model_sizes:
    #     for lr in hyperparameter_grid['learning_rate']:
    #         for bs in hyperparameter_grid['batch_size']:
    #             for ep in hyperparameter_grid['epochs']:
    #                 args = basic_args + 
    #                 ["--model_size", model_size,
    #                 "--lr", str(lr), 
    #                 "--batch_size", str(bs), 
    #                 "--epochs", str(ep)]
                    
    #                 for key, value in kwargs.items():
    #                     args.extend([f"--{key}", str(value)])
    #                 print(f"Training ground truth model size {model_size} with LR={lr}, BS={bs}, Epochs={ep}")
    #                 subprocess.run(args, check=True)

    print("Running transfer models")
    for i in range(len(model_sizes)):
        for j in range(i, len(model_sizes)):
            weak_model_size = model_sizes[i]
            strong_model_size = model_sizes[j]
            print(f"Running weak {weak_model_size} to strong {strong_model_size}")
            subprocess.run(
                basic_args
                + ["--weak_model_size", weak_model_size, "--model_size", strong_model_size],
                check=True,
            )

    # # Grid search for transfer models
    # print("Running transfer models")
    # for i in range(len(model_sizes)):
    #     for j in range(i, len(model_sizes)):
    #         weak_model_size = model_sizes[i]
    #         strong_model_size = model_sizes[j]
    #         for lr in hyperparameter_grid['learning_rate']:
    #             for bs in hyperparameter_grid['batch_size']:
    #                 for ep in hyperparameter_grid['epochs']:
    #                     args = basic_args + 
    #                     ["--weak_model_size", weak_model_size, 
    #                     "--model_size", strong_model_size, 
    #                     "--lr", str(lr), 
    #                     "--batch_size", str(bs), 
    #                     "--epochs", str(ep)]

    #                     for key, value in kwargs.items():
    #                         args.extend([f"--{key}", str(value)])
    #                     print(f"Running transfer from weak {weak_model_size} to strong {strong_model_size} with LR={lr}, BS={bs}, Epochs={ep}")
    #                     subprocess.run(args, check=True)

if __name__ == "__main__":
    fire.Fire(main)
