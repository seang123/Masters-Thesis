import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")

run_name = config['run']
out_path = os.path.join(config['log'], run_name, 'eval_out')


custom_name = 'gradient_check2'
out_path = os.path.join(config['log'], custom_name, 'eval_out')


if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating folder: {out_path}")

def plot_train_loss_with_reg(csv_file_path, out_path):

    df = pd.read_csv(csv_file_path)
    #headers: loss,reg_loss,accuracy,val_loss,val_reg_loss,val_accuracy

    fig, ax = plt.subplots(3, 1, figsize=(15,15), sharex=True)

    # Cross entropy loss
    ax[0].plot(df.loss, label = 'train')
    ax[0].plot(df.val_loss, label = 'val')
    ax[0].set_title('Cross-entropy Loss')
    ax[0].legend()

    ax[1].set_title("L2 loss")
    ax[1].plot(df.reg_loss[:], label = 'train')
    ax[1].plot(df.val_reg_loss[:], label = 'val')
    ax[1].legend()

    # Accuracy
    ax[2].axhline(0.50, color = 'k', linestyle = '--')
    if max(df.accuracy.values) > 0.9 or max(df.val_accuracy.values) > 0.9:
        ax[2].axhline(1.00, color = 'k', linestyle = ':')
    ax[2].plot(df.accuracy, label = 'train')
    ax[2].plot(df.val_accuracy, label = 'val')
    ax[2].set_title('Categorical Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].legend()

    plt.savefig(f'{out_path}/training_loss.png')
    plt.close(fig)

def plot_train_loss(csv_file_path, out_path):

    df = pd.read_csv(csv_file_path)

    fig, ax = plt.subplots(2, 1, sharex=True)

    # Cross entropy loss
    ax[0].plot(df.loss, label = 'train')
    ax[0].plot(df.val_loss, label = 'val')
    ax[0].set_title('Cross-entropy Loss')
    ax[0].legend()

    # Accuracy
    ax[1].axhline(0.50, color = 'k', linestyle = '--')
    ax[1].axhline(1.00, color = 'k', linestyle = ':')
    ax[1].plot(df.accuracy, label = 'train')
    ax[1].plot(df.val_accuracy, label = 'val')
    ax[1].set_title('Categorical Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    plt.savefig(f'{out_path}/training_loss.png')
    plt.close(fig)


if __name__ == '__main__':
#    plot_train_loss_with_reg(f"{config['log']}/{config['run']}/training_log.csv", out_path)
    plot_train_loss_with_reg(f"{config['log']}/{custom_name}/training_log.csv", out_path)
