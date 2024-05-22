import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def visualize(config_path):
    with open(config_path, "r") as f:
        paths = json.load(f)
    csv_path = paths["csv_path"]

    data = pd.read_csv(csv_path)
    steps = data['step']
    training_losses = data['train_loss']
    validation_losses = data['validation_loss']

    plt.figure(figsize=(12, 6))
    plt.plot(steps, training_losses, linestyle='-', color='green', alpha=0.8, label='Training Loss')
    plt.plot(steps, validation_losses, linestyle='-', color='blue', alpha=0.8, label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Step')
    plt.ylabel('Loss (%)')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <path_to_json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    visualize(config_path)
