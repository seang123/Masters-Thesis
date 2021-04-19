import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from model import Encoder, Decoder, CaptionGenerator
import datetime
from dataclass import Dataclass
import traceback
from contextlib import redirect_stdout 


gpu_to_use = 2

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

## Start telegram bot
bot = tb.Tensorbot()
gpu_var   = bot.register_variable("GPU", "", autoupdate=True)
epoch_var = bot.register_variable("", "", autoupdate=True)
err_var   = bot.register_variable("ERROR", "", autoupdate=True)

top_k = 6000
dataclass = Dataclass(73000, top_k)

tokenizer = dataclass.get_tokenizer()
max_length = dataclass.max_length()

img_name_train, cap_train, img_name_val, cap_val = dataclass.train_test_split(0.95)


## Parameters
BATCH_SIZE = 128
BUFFER_SIZE = 1000
embedding_dim = 512 
units = 512 # recurrent units
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
num_steps_test = len(img_name_val) // BATCH_SIZE
EPOCHS = 10
save_checkpoints = True 
save_data = True


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
    num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

## validation dataset
dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
dataset_val = dataset_val.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
dataset_val = dataset_val.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


## Optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
metric_object = tf.keras.metrics.SparseCategoricalCrossentropy()


## Init Model
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)
model = CaptionGenerator(encoder, decoder, tokenizer, max_length)
model.compile(optimizer, loss_object, metric_object, run_eagerly=True)

## Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# Current time string
current_time = datetime.datetime.now().strftime("%H%M%S-%d%m%Y")
train_start_time = datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')

# Loggers for Tensorboard
log_dir = 'logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(log_dir)


data_path = f"./data/"
## Checkpoints handler
checkpoint_path = f"./checkpoints/train/"
ckpt = tf.train.Checkpoint(encoder=encoder,
                        decoder=decoder,
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

## load loss data if it exists
training_loss, training_batch_loss, testing_loss, testing_batch_loss = dataclass.load_loss(f"{data_path}loss_data.npz")

## Store validation data
dataclass.save_val_keys(f"{data_path}img_name_val.txt", img_name_val)

## Main Loop
def main(gpu = 2):
    print(f"Training for {EPOCHS}({start_epoch}) epochs")
    epoch_var.update(f"Starting Training for {EPOCHS}({start_epoch}) epochs")

    #training_loss = []
    #training_batch_loss = []

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()

        total_loss = 0
        
        pre_batch_time = 0
        for (batch, (img_tensor, target)) in dataset.enumerate():

            #with tf.device(f'/gpu:{gpu}'):
            losses = model.train_step((img_tensor, target))
            sum_loss, t_loss = losses.values()
            total_loss += t_loss

            training_loss.append(t_loss)
            training_batch_loss.append(sum_loss)

            if batch % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch:4} | Loss {(t_loss):.4f} | {(time.time()-start-pre_batch_time):.2f} sec")
                pre_batch_time = time.time() - start

        #print(f"Epoch {epoch} done.")
        print(f"Training    | Loss {(total_loss/num_steps):.6f} | Total Time: {(time.time() - start):.2f} sec")
        epoch_var.update(f"TRAIN: epoch {epoch} done\navg loss: {(total_loss/num_steps):.4f}\ntotal time: {(time.time()-start):.2f} sec")

        if save_checkpoints:
            ckpt_manager.save()


        print("Testing ...")
        start_test = time.time()
        total_loss_test = 0
        for (batch, (img_tensor, target)) in dataset_val.enumerate():
            losses = model.test_step((img_tensor, target))
            sum_loss, t_loss = losses.values()
            total_loss_test += t_loss 

            testing_loss.append(t_loss)
            testing_batch_loss.append(sum_loss)

        epoch_var.update(f"TEST: epoch {epoch} done\navg loss: {(total_loss_test/num_steps_test):.4f}\ntotal time: {(time.time()-start_test):.2f} sec")

        print(f"Testing     | Loss {(total_loss_test/num_steps_test):.6f} | Total Time: {(time.time() - start_test):.2f} sec\n", end = '\r') 
        print(f"Epoch {epoch} done.")

        try:
            gen_send_plot(epoch)
        except Exception:
            pass


    print("## Training Complete. ##")
    epoch_var.update("## Training Complete. ##")

    return #training_loss, training_batch_loss

def gen_send_plot(epoch, send_plot=True):
    """Generates a plt.plot and saves it as png
    Telegram bot send that png
    """
    # create loss plot
    fig = plt.figure()
    plt.title(f"Total normalised loss per batch. Epoch {epoch}")
    plt.ylabel("loss")
    plt.xlabel("batch")
    plt.plot(training_loss)
    plt.savefig("./Figures/training_loss_so_far.png")
    plt.close(fig)
    # send the generated png
    if send_plot:
        bot.send_plot("./Figures/training_loss_so_far.png")


def save_loss():
    t_loss = np.array(training_loss)
    t_b_loss = np.array(training_batch_loss)
    test_loss = np.array(testing_loss)
    test_b_loss = np.array(testing_batch_loss)
    with open(f'{data_path}loss_data.npz', 'wb') as f:
        np.savez(f, xtrain=t_loss, ytrain=t_b_loss, xtest=test_loss, ytest=test_b_loss)

def save_model_sum():
    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            encoder.summary()
            decoder.summary()
        f.write("\n")
        f.write(parameter_string)
        f.write("\n")
        f.write(f"Total training epochs: {EPOCHS}")
        f.write(f"\nTraining started at: {train_start_time}")
        f.write(f"\nTraining completed at: {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}")


if __name__ == '__main__':
    try:
        #train_loss, train_batch_loss = 
        main()
        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,profile_batch='100, 105')
        #model.fit(dataset, epochs = 1, steps_per_epoch=num_steps, callbacks=[tb_callback])
    except Exception as e:
        err_str = f"Caught error in main training loop. {datetime.datetime.now().strftime('%H:%M:%S - %d/%m/%Y')}"
        print(err_str)
        err_var.update(err_str)
        traceback.print_exc(file=open('ERROR_file.txt', 'w'))
        bot.kill()
        raise e
    except KeyboardInterrupt as e:
        print("\nKeyboard Interrupt")
        print("Saving partial data")

    if save_data:
        try:
            save_model_sum()
            print("Model summary saved")
        except Exception as e:
            print("Failed to store model summary")
            traceback.print_exc(file=open('ERROR_file.txt', "a"))

        try:
            save_loss()
            print("Data saved!")
        except Exception as e:
            print("error saving loss data") 
            err_var.update("failt to save loss data")
            traceback.print_exc(file=open('ERROR_file.txt', 'a'))

    bot.kill()
    print("Done.")
