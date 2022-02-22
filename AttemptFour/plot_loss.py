import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='darkgrid')
import numpy as np
import pandas as pd
import yaml
import os
import argparse
import pandas as pd
import tbparse

parser = argparse.ArgumentParser(description='Evaluate NIC model')
parser.add_argument('--dir', type=str, required=True)
args = parser.parse_args()

with open(f"./Log/{args.dir}/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded:\n\t{f.name}")

run_name = config['run']
out_path = os.path.join(config['log'], args.dir, 'eval_out')

# Output dir
if not os.path.exists(out_path):
    os.makedirs(out_path)
    print(f"Creating folder: {out_path}")

# Load loss history
df = pd.read_csv(f'./Log/{args.dir}/loss_history.csv')

log_dir = './tb_logs/scalars/no_attn_loss/2022-02-19_14-49-49/'
reader = tbparse.SummaryReader(log_dir, pivot=False)
df = reader.scalars # [step, tag, value]

def plot_tb_acc():

    epoch_acc = df['value'].loc[df['tag'] == 'epoch_accuracy'].loc[~df['value'].isna()].values
    train = []
    val = []
    c = 0
    for i in range(len(epoch_acc)):
        x = epoch_acc[i]
        if c % 2 == 0:
            train.append(x)
        else:
            val.append(x)
        c += 1
    acc = np.stack((train,val), axis=0)

    palette = sns.color_palette("mako_r", 6)

    fig, ax = plt.subplots(1,1, figsize=(16,9))  # w * h
    sns.lineplot(ax = ax, data = acc[0], palette = palette, label='train')
    sns.lineplot(ax = ax, data = acc[1], palette = palette, label='val')
    plt.title("Categorical Accuracy")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"{out_path}/accuracy.png", bbox_inches='tight')
    plt.close(fig)
    return

def plot_tb_loss():
    """ Pulls data from TB and plots that 
    Looks slighly different to just plotting the keras.history data 
    """
    epoch_loss = df['value'].loc[df['tag'] == 'epoch_loss'].loc[~df['value'].isna()].values

    loss = []
    val_loss = []
    c = 0
    for i in range(len(epoch_loss)):
        x = epoch_loss[i]
        if c % 2 == 0:
            loss.append(x)
        else:
            val_loss.append(x)
        c += 1
    loss = np.stack((loss,val_loss), axis=0)

    palette = sns.color_palette("mako_r", 6)

    fig, ax = plt.subplots(1,1, figsize=(16,9))  # w * h
    sns.lineplot(ax = ax, data = loss[0], palette = palette, label='train')
    sns.lineplot(ax = ax, data = loss[1], palette = palette, label='val')
    plt.title("Cross entropy loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"{out_path}/loss.png", bbox_inches='tight')
    plt.close(fig)
    return

def sb(df):

    def smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        return smoothed

    # Data 
    epoch_df = df.groupby(['epoch']).mean()
    epoch_loss = epoch_df['loss']

    batch_loss = smooth(list(df['loss'].dropna()), 0.90)
    batch_loss_val = smooth(list(df['val_loss'].dropna()), 0.90)

    # Plot
    palette = sns.color_palette("mako_r", 6)

    fig, ax = plt.subplots(2,1, figsize=(15,15))
    sns.lineplot(ax = ax[0], data = epoch_loss, palette = palette)
    sns.lineplot(ax = ax[0], data = epoch_df['val_loss'], palette=palette)

    sns.lineplot(ax = ax[1], data = batch_loss, palette=palette)
    sns.lineplot(ax = ax[1].twiny(), data = batch_loss_val, palette=palette)

    plt.savefig(f'{out_path}/training_loss.png')
    plt.close(fig)
    return

def plot_epoch_loss_acc(df, out_path):
    """ Plot loss and accuracy """

    fig, ax = plt.subplots(2, 1, figsize=(15,15), sharex=True)

    # Group batches by epoch and average
    epoch_df = df.groupby(['epoch']).mean()
    epoch_loss = epoch_df['loss']
    epoch_val_loss = epoch_df['val_loss']

    epoch_acc = epoch_df['accuracy']
    epoch_val_acc = epoch_df['val_accuracy']

    ax[0].plot(epoch_loss, label='train')
    ax[0].plot(epoch_val_loss, label='val')
    #sns.lineplot(ax=ax[0], x=range(len(epoch_loss)), y=epoch_loss, label='train')
    #sns.lineplot(ax=ax[0], x=range(len(epoch_val_loss)), y=epoch_val_loss, label='val')
    ax[0].set_title('Cross-entropy loss')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(epoch_acc, label='train')
    ax[1].plot(epoch_val_acc, label='val')
    #sns.lineplot(ax=ax[0], x=range(len(epoch_acc)), y=epoch_acc, label='train')
    #sns.lineplot(ax=ax[0], x=range(len(epoch_val_acc)), y=epoch_val_acc, label='val')
    ax[1].axhline(0.50, color = 'k', linestyle = '--')
    ax[1].set_title('Accuracy')
    ax[1].set_ylabel('%')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()
    ax[1].grid()

    plt.savefig(f'{out_path}/training_loss.png')
    plt.close(fig)
    return

def plot_batch_loss(df, out_path):

    fig, ax = plt.subplots(2, 1, figsize=(15,15), sharex=True)

    #ax[0].plot(list(df['loss'].dropna()), label='train')
    sns.lineplot(ax=ax[0], x=range(len(list(df['loss'].dropna()))), y=list(df['loss'].dropna()), label='train')
    ax2 = ax[0].twiny()
    #ax2.plot(list(df['val_loss'].dropna()), label='val')
    sns.lineplot(ax=ax2, x=range(len(list(df['val_loss'].dropna()))), y=list(df['val_loss'].dropna()), label='train')
    ax[0].grid()
    ax[0].legend()
    
    #ax[1].plot(list(df['accuracy'].dropna()), label='train')
    sns.lineplot(ax = ax[1], x=range(len(list(df['accuracy'].dropna()))), y=list(df['accuracy'].dropna()), label='train')
    ax2 = ax[1].twiny()
    #ax2.plot(list(df['val_accuracy'].dropna()), label='val')
    sns.lineplot(ax = ax2, x=range(len(list(df['val_accuracy'].dropna()))), y=list(df['val_accuracy'].dropna()), label='train')
    ax[1].grid()
    ax[1].legend()
    plt.savefig(f'{out_path}/training_loss_batch.png')
    plt.close(fig)


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
    #plot_train_loss_with_reg(f"{config['log']}/{config['run']}/training_log.csv", out_path)
    #plot_train_loss_with_reg(f"{config['log']}/{custom_name}/training_log.csv", out_path)

    plot_tb_loss()
    plot_tb_acc()
#    plot_epoch_loss_acc(df, out_path)
    #plot_batch_loss(df, out_path)

    #sb(df)


