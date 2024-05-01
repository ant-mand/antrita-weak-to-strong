import subprocess
import sys

def main(dataset_name, model_sizes, num_steps=600):  # Set default steps to 600
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(',')

    # Base command that includes the script to run and the fixed number of steps
    basic_args = [sys.executable, "train_simple.py", "--num_steps", str(num_steps), "--ds_name", dataset_name]
    
    # Loop over all specified model sizes and run the training script for each one
    for model_size in model_sizes:
        model_specific_args = basic_args + ["--model_size", model_size]
        subprocess.run(model_specific_args, check=True)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
