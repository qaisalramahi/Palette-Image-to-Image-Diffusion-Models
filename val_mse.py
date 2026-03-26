import re
import sys
import matplotlib.pyplot as plt

def plot_val_mse(log_path):
    epochs = []
    mse_values = []
    current_epoch = None

    with open(log_path, 'r') as f:
        for line in f:
            epoch_match = re.search(r'epoch:\s*(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            mse_match = re.search(r'val/mse:\s*([\d.]+)', line)
            if mse_match:
                mse_values.append(float(mse_match.group(1)))
                epochs.append(current_epoch if current_epoch is not None else len(mse_values))

    if not mse_values:
        print("No val/mse entries found in the log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_values, marker='o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE over Training')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_path.replace('train.log', 'val_mse.png'), dpi=150)
    plt.show()
    print(f"Epochs: {epochs}")
    print(f"Val MSE: {mse_values}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        log_path = input("Enter path to train.log: ")
    plot_val_mse(log_path)
