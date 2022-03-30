import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

## List of loss directories to plot
loss_dir = {
        0: "no_attn_loss_const_lr2",
        1: "large_lstm",
        2: "layer_norm_lstm",
        3: "multi_sub_sep_enc",
        4: "no_stopwords_large_lstm",
        5: "no_attn_loss",
        6: "no_attn_loss_subject_1",
        7: "attn_loss",
        8: "multi_sub_sep_enc2",
        9: "multi_sub_same_enc",
}

# ['batch_L2' 'batch_accuracy' 'batch_attention' 'batch_loss' 'batch_lr' 'epoch_L2' 'epoch_accuracy' 'epoch_attention' 'epoch_loss' 'epoch_lr']

def get_loss(model_name: str):

    df = pd.read_csv(f"./Log/{model_name}/loss_history.csv")

    ## Example for getting loss or val_loss
    #loss = df.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    #loss = [loss[i][-1] for i in range(len(loss))]
    #val = df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    #val = [val[i][-1] for i in range(len(val))]

    return df

def plot_loss(df1):

    fig = plt.figure(figsize=(16,9))

    loss = df1.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    val = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    val = [val[i][-1] for i in range(len(val))]
    plt.plot(loss, label='train')
    plt.plot(val, label='val')

    # plot parameters
    #plt.ylim(ymin=1.0)
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Cross-entropy loss")
    plt.savefig(f"./Eval/loss.png")

    plt.close(fig)

def plot_():
    # plot const_lr2 vs multi_sub_sep_enc
    df1 = get_loss(loss_dir[0])
    df2 = get_loss(loss_dir[3])

    fig = plt.figure(figsize=(16,9))

    ## Const lr
    # Train loss
    loss = df1.dropna(subset=['loss']).groupby('epoch')['loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, label='train')
    # val loss
    loss = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]

def plot_2():
    df1 = get_loss(loss_dir[8])
    df2 = get_loss(loss_dir[0])
    df3 = get_loss(loss_dir[9])

    fig = plt.figure(figsize=(16,9))
    
    # Loss (const lr2)
    loss = df2.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))][:100]
    plt.plot(loss, label='single subject')
    # Loss (multi sup same enc)
    loss = df3.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, label='multi subject (same enc)')
    # Loss (multi sup sep enc3)
    loss = df1.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].apply(list)
    loss = [loss[i][-1] for i in range(len(loss))]
    plt.plot(loss, label='multi subject (sep enc)')

    plt.xlim(xmin = -(.05 * plt.gca().get_xlim()[1]))
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Cross-entropy val loss comparing single subject (2) vs multi subject (1 & 2)\nMulti-subject runs compared when using a single encoder or one per subject")
    plt.savefig(f"./Eval/msse3_vs_constlr2.png")

    plt.close(fig)






if __name__ == '__main__':
#    df = get_loss(loss_dir[3])
#    plot_loss(df)

    plot_2()

