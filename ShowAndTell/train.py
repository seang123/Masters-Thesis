
import numpy as np
import tensorflow as tf
import time
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
sys.path.append('/home/seagie/sandbox/Tensorgram/')
import tensorbot as tb
import utils
import h5py
import collections
from model import Decoder as RNN
import datetime
from dataclass import Dataclass

## Start telegram bot
bot = tb.Tensorbot()
gpu_var   = bot.register_variable("GPU", "", autoupdate=True)
epoch_var = bot.register_variable("", "", autoupdate=True)
err_var   = bot.register_variable("ERROR:", "", autoupdate=True)

top_k = 5000
dataclass = Dataclass(73000, top_k)

tokenizer = dataclass.get_tokenizer()

img_name_train, cap_train, img_name_val, _ = dataclass.train_test_split(0.95)


## Parameters
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512 # recurrent units
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
EPOCHS = 5
save_checkpoints = True


## load image features data into memory
img_features = np.load('./img_features_vgg16').astype(np.float32)

## TF dataset
def map_func(img_idx, cap):
    """
    Map image features into tf.dataset
    """
    img_tensor = img_features[int(img_idx),:]
    return img_tensor, cap

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

## Optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


## Init Model
decoder = RNN(embedding_dim, units, vocab_size, tokenizer, optimizer, loss_object)

## Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Current time string
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_start_time = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')


## Checkpoints handler
checkpoint_path = f"./checkpoints/train"
ckpt = tf.train.Checkpoint(decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

if start_epoch != 0:
    print(f"Checkpoint loaded. Starting from epoch {start_epoch}")
else:
    print(f"No checkpoint loaded.")


parameter_string = f"Parameters:\nBatch Size: {BATCH_SIZE} | embedding dim: {embedding_dim} | units: {units} | vocab size: {vocab_size} | nr batches: {num_steps} | train set: {len(img_name_train)} | test set: {len(img_name_val)}"
print("###################")
print(parameter_string)
print("###################")


## Main Loop
def main(gpu = 2):
    print(f"Training for {EPOCHS} epochs")
    epoch_var.update(f"Starting Training for {EPOCHS} epochs")

    training_loss = []
    training_batch_loss = []

    with tf.device(f'/gpu:{gpu}'):
        for epoch in range(start_epoch, EPOCHS):
            start = time.time()

            total_loss = 0
            
            pre_batch_time = 0
            for (batch, (img_tensor, target)) in dataset.enumerate():

                batch_loss, t_loss = decoder.train_step(img_tensor, target)
                total_loss += t_loss

                training_loss.append(t_loss)
                training_batch_loss.append(batch_loss)

                if batch % 100 == 0:
                    print(f"Epoch {epoch} | Batch {batch:4} | Loss {(t_loss):.4f} | {(time.time()-start-pre_batch_time):.2f} sec")
                    pre_batch_time = time.time() - start

            print(f"Epoch {epoch} done.")
            epoch_var.update(f"TRAIN: epoch {epoch} done\ntotal_loss: {total_loss:.4f}\ntotal time: {(time.time()-start):.2f} sec")


            print(f"Epoch {epoch} | Loss {(total_loss/num_steps):.6f} | Total Time {(time.time() - start):.2f} sec\n")

            if save_checkpoints:
                ckpt_manager.save()

        print("## Training Complete. ##")
        epoch_var.update("## Training Complete. ##")

        return training_loss, training_batch_loss

def save_loss(train_loss, train_batch_loss):
    t_loss = np.array(train_loss)
    t_b_loss = np.array(train_batch_loss)
    with open('./loss_data.npy', 'wb') as f:
        np.savez(f, x=t_loss, y=t_b_loss)

def save_model_sum():
    with open('./modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            decoder.encoder.summary()
            decoder.summary()
        f.write("\n")
        f.write(parameter_string)
        f.write("\n")
        f.write(f"Total training epochs: {EPOCHS}")
        f.write(f"\nTraining started at: {train_start_time}")
        f.write(f"\nTraining completed at: {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}")


if __name__ == '__main__':
    try:
        train_loss, train_batch_loss = main()
    except Exception as e:
        err_str = f"Caught error in main training loop. {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}"
        print(err_str)
        err_var.update(err_str)
        bot.kill()
        raise e
    except KeyboardInterrupt as e:
        print("Caught Keyboard Interrupt")
        print("Saving partial data")


    try:
        save_loss(train_loss, train_batch_loss)
        save_model_sum()
        print("Data saved!")
    except Exception as e:
        print("error saving loss data and model summary") 
        err_var.update("svaing loss data and model summary")
        bot.kill()

    bot.kill()
    print("Done.")
