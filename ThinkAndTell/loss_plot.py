import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse

"""

Given a folder containing a trial (loss_data, config.txt) plot the loss 

folder should be given by name, and be located in /data

"""

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--name", type=str, default="")

p_args = parser.parse_args()

assert p_args.name != ""
print("name:", p_args.name)

folder_path = f"./data/{p_args.name}/"

with open(f"{folder_path}config.txt", "r") as f:
    config = f.read()
    config = json.loads(config)
print("Config loaded")


L2_val = config['L2']
alpha = config['LR']

loss_file_path = folder_path + "loss_data.npz"

loss_data = np.load(loss_file_path)

print("loss data loaded: ", loss_data.files)

train_loss = loss_data['xtrain']
test_loss = loss_data['xtest']

#
# COMPUTE AVERAGES
#

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

train_avg = moving_average(train_loss, 100)
test_avg = moving_average(test_loss, 100)


#
# PLOTTING
#

fig, ax1 = plt.subplots(figsize=(15,5)) # (width, height) in inches

epochs = config['EPOCHS']
batch_size = config['BATCH_SIZE']
    
# Plot data
ax1_, = ax1.plot(np.linspace(0, epochs, len(train_avg)), train_avg, color='k', label="train avg")
# Plot data on new axis
ax2 = ax1.twiny()
ax2_, = ax2.plot(np.linspace(0, epochs, len(test_avg)), test_avg, color='g', label="test avg")

ax1.set_xlabel("Batch")
ax1.set_ylabel("Loss")
ax1.set_title(f"Loss across {epochs} epochs | L2:{L2_val} | LR:{alpha}")

ax1.set_xlabel("Batch")
ax2.set_xlabel("", color = 'tab:green')
ax2.tick_params(axis='x', labelcolor='tab:green')


# ax_train.spines['top'].set_position(('outward', 30))

lns = [ax1_, ax2_]
ax1.legend(handles=lns, loc='upper left')

ax1.yaxis.label.set_color(ax1_.get_color())
ax2.yaxis.label.set_color(ax2_.get_color())

ax1.grid(True)

plt.savefig(f"{folder_path}loss_plot.png")
plt.close()
