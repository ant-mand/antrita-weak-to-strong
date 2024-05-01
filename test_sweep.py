import subprocess
import sys

def main(model_sizes, num_steps=10000):
    # Split model sizes if passed as a comma-separated string
    if isinstance(model_sizes, str):
        model_sizes = model_sizes.split(',')

    # Base command that includes the script to run and the fixed number of steps
    basic_args = [sys.executable, "test_train_simple.py", "--num_steps", str(num_steps)]
    
    # Loop over all specified model sizes and run the training script for each one
    for model_size in model_sizes:
        subprocess.run(basic_args + ["--model_size", model_size], check=True)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
