import logging
import time
import pandas as pd
import os, sys
import csv
import tensorflow as tf
from tensorflow.keras.optimizers import schedules
import tensorflow_addons as tfa
from tensorflow.keras.utils import Progbar
import numpy as np
#from Model import lc_NIC
from Model import ms2_NIC as lc_NIC
#from Model import tmp_lc_NIC as lc_NIC
from DataLoaders import load_avg_betas as loader
#from DataLoaders import data_generator as generator
#from DataLoaders import data_generator_guse as generator
from DataLoaders import data_generator_multisub as generator
from Callbacks import BatchLoss, EpochLoss, WarmupScheduler, Predict
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
from datetime import datetime
import subprocess
import yaml

gpu_to_use = 0
print(f"Running on GPU: {gpu_to_use}")

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

## Load the configuration file
with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")

run_name = config['run']
run_path = os.path.join(config['log'], run_name)

if not os.path.exists(run_path):
    os.makedirs(run_path)
    print(f"Creating training Log folder: {run_path}")
else:
    print(f"Training Log will be saved to: {run_path}")

with open(f"{run_path}/config.yaml", "w+") as f:
    yaml.dump(config, f)

logging.basicConfig(filename=f'{run_path}/log.log', filemode='w', level=logging.DEBUG)

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

# Copy Model file to run_path folder for record
subprocess.run(["cp", "./Model/ms2_NIC.py", f"{run_path}/ms2_NIC.py"], shell=False, check=True)
print(f"Model file copied to {run_path} for record")

## Parameters
vocab_size = config['top_k'] + 1

#
## Load data
#
train_keys_one, val_keys_one, test_keys_one = loader.get_nsd_keys('1')
print("subject 1")
print("train_keys:", train_keys_one.shape)
print("val_keys:", val_keys_one.shape)
train_keys, val_keys, test_keys = loader.get_nsd_keys('2')
print("subject 2")
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)

# TODO: get test set and remove from val keys
#test = loader.get_test_set()
#print(test.shape)

# Keep only validation split  (rest is test data)
#val_split = np.loadtxt("./TrainData/val_split.txt", dtype=np.int32)
#val_keys = val_keys[val_split]

train_pairs_one= np.array(loader.create_pairs(train_keys_one, '1'))
val_pairs_one = np.array(loader.create_pairs(val_keys_one, '1'))
print("subject 1")
print("train_pairs:", train_pairs_one.shape)
print("val_pairs:  ", val_pairs_one.shape)

train_pairs_two = np.array(loader.create_pairs(train_keys, '2'))
val_pairs_two = np.array(loader.create_pairs(val_keys, '2'))
print("subject 2")
print("train_pairs:", train_pairs_two.shape)
print("val_pairs:  ", val_pairs_two.shape)

train_pairs = [train_pairs_one, train_pairs_two]
val_pairs = [val_pairs_one, val_pairs_two]

print(len(train_pairs), len(train_pairs[0]))
print(len(val_pairs), len(val_pairs[0]))

#tokenizer, _ = loader.build_tokenizer(np.arange(1, 73001), config['top_k'])
tokenizer = loader.load_tokenizer()

def lr_schedule(step):
    # final lr = initial_lr * decay_rate
    decay_steps = 10
    decay_rate = 0.01 
    inital_lr = 0.01
    final_lr = 0.0001
    return 0.0001
    #return max(inital_lr * decay_rate ** (step / decay_steps), final_lr)

# Setup optimizer 
if config['optimizer'] == 'Adam':
    print(f"Using Adam optimizer with lr: {config['alpha']}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['alpha'], beta_1 = 0.9, beta_2=0.98, epsilon=10.0e-9, clipnorm=config['clipnorm'])
    #optimizer = tf.keras.optimizers.Adam(learning_rate=config['alpha'])
    #optimizer = tfa.optimizers.AdamW(0.001, config['alpha'], beta_1 = 0.9, beta_2 = 0.98, epsilon = 10.0e-09)
    print(f"Using optimizer: Adam")
elif config['optimizer'] == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=config['alpha'], momentum=0.9, nesterov=False)
    print(f"Using optimizer: SGD")
else:
    print("No optimizer specified")

# Loss function
loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction='none'
)

# Setup Model
model = lc_NIC.NIC(
        #loader.get_groups(config['embedding_features'])[0], 
        #loader.get_groups(config['embedding_features'])[1],
        #loader.get_groups(config['group_size'])[0], 
        #loader.get_groups(config['group_size'])[1],
        loader.get_groups(config['group_size'], separate_hemi=True),
        config['units'], 
        config['embedding_features'], 
        config['embedding_text'],
        config['attn_units'],
        vocab_size,
        config['max_length'],
        config['dropout_input'],
        config['dropout_features'],
        config['dropout_text'],
        config['dropout_attn'],
        config['dropout_lstm'],
        config['dropout_out'],
        config['input_reg'],
        config['attn_reg'],
        config['lstm_reg'],
        config['output_reg']
        )
#Compile
model.compile(optimizer, loss_object, run_eagerly=True)

## The following relates to pre-loading LSTM weights 
init_generator = generator.DataGenerator(
        train_pairs, 
        config['batch_size'], 
        tokenizer, 
        config['units'], 
        config['max_length'], 
        vocab_size, 
        pre_load_betas=False,
        shuffle=False, training=True)

build_time = time.perf_counter()
model(init_generator.__getitem__(0)[0])
print(f"Model build time: {(time.perf_counter() - build_time):.3f}")
print(model.summary())

if False:
    # 1. Make one pass through the model 
    model(init_generator.__getitem__(0)[0])
    print(model.summary())
    # 2. Load weights
    pre_trained_weights = np.load('./Log/vgg16_all_samples/pre_train_weights.npy', allow_pickle=True)
    for i in pre_trained_weights:
        print(i.shape)
    lstm_weights = pre_trained_weights[-5:-2]
    time_dist_weights = pre_trained_weights[-2:]
    # 3. Set LSTM layer weights
    model.get_layer('lstm').set_weights(lstm_weights)
    model.get_layer('time_distributed_softmax').set_weights(time_dist_weights)
    print("Weights loaded for: LSTM & Time-distributed-softmax")



# Setup Checkpoint handler
checkpoint_path = f"{run_path}/model/" 
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path_best = f"{checkpoint_path}" + "model-ep{epoch:03d}.h5"
checkpoint_best = ModelCheckpoint(
        checkpoint_path_best,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1
)
checkpoint_path_latest = f"{checkpoint_path}model-latest.h5"
checkpoint_latest = ModelCheckpoint(
        checkpoint_path_latest,
        monitor='val_loss',
        verbose=0, 
        save_weights_only = True, 
        save_best_only = False,
        mode = 'min',
        period=1
)

#
## Callbacks
#
#batch_loss_writer = BatchLoss.BatchLoss(f"{run_path}/batch_training_log.csv", f"{run_path}")
#epoch_loss_writer = EpochLoss.EpochLoss(f"{run_path}/training_log.csv")
loss_history = EpochLoss.LossHistory(f"{run_path}/loss_history.csv", f"{run_path}")

#early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=10, min_delta=0.005, min_lr=0.0001)

logdir = f"./tb_logs/scalars/{config['run']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
tensorboard_callback = TensorBoard(
        log_dir=logdir, 
        update_freq='batch',
        #histogram_freq=1,
        #write_graph=True,
        #write_images=True,
        #embeddings_freq=1,
        #profile_batch='200,220',
        )
#file_writer = tf.summary.create_file_writer(logdir)

# Init a generator used during the predict callback
#val_generator_pred = create_generator(val_pairs, False)
#predict_callback = Predict.Predict(val_generator_pred, tokenizer, file_writer, config['max_length'], config['units'])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule, verbose = 0)

_callbacks = [
        #batch_loss_writer, 
        #epoch_loss_writer, 
        #lr_scheduler,
        loss_history,
        tensorboard_callback, 
        #reduce_lr,
        checkpoint_latest,
        checkpoint_best,
        #predict_callback,
#        early_stop
]
callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=model)

logs = {}
start_epoch = 0

def dotfit():
    logging.info("training with .fit()")

    train_generator = generator.DataGenerator(
            train_pairs, 
            config['batch_size'], 
            tokenizer, 
            config['units'], 
            config['max_length'], 
            vocab_size, 
            pre_load_betas=False,
            shuffle=True, training=True)
    val_generator = generator.DataGenerator(
            val_pairs, 
            config['batch_size'], 
            tokenizer, 
            config['units'], 
            config['max_length'], 
            vocab_size, 
            pre_load_betas=False,
            shuffle=False, training=True)

    """
    model_name = './Log/no_attn_loss_const_lr2/model/model-ep114.h5'
    start_epoch = 20
    print(f"loading weights from:\n\t {model_name}")
    print(f"starting training from epoch: {start_epoch}")
    model.load_weights(model_name,by_name=True,skip_mismatch=True)
    """

    model.fit(
            train_generator,
            epochs = config['epochs'],
            steps_per_epoch = train_pairs[0].shape[0]//config['batch_size'],
            batch_size = config['batch_size'],
            callbacks = _callbacks,
            validation_data = val_generator,
            validation_steps = val_pairs[0].shape[0]//config['batch_size'],
            initial_epoch = start_epoch,
    )
    return




def custom_train_loop():
    print(f"------\nRunning custom training loop")
    print(f"for {config['epochs'] - start_epoch} epochs\n------")
    logging.info("training with custom training loop")

    train_generator = generator.DataGenerator(
            train_pairs, 
            config['batch_size'], 
            tokenizer, 
            config['units'], 
            config['max_length'], 
            vocab_size, 
            pre_load_betas=False,
            shuffle=True, training=True)
    val_generator = generator.DataGenerator(
            val_pairs, 
            config['batch_size'], 
            tokenizer, 
            config['units'], 
            config['max_length'], 
            vocab_size, 
            pre_load_betas=False,
            shuffle=False, training=True)

    print("len train generator:", len(train_generator))
    print("len val generator:", len(val_generator))

    # Train for N epochs
    callbacks.on_train_begin(logs=logs)
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch, logs=logs)
        
        # Progress bar
        pb = Progbar(train_pairs[0].shape[0]/config['batch_size'])#, stateful_metrics=['loss', 'l2', 'accuracy'])
        pb2 = Progbar(val_pairs[0].shape[0]/config['batch_size'])#, stateful_metrics=['val-loss', 'val-l2', 'val-accuracy'])

        # Training
        for (batch_nr, data) in enumerate(train_generator):
            callbacks.on_batch_begin(epoch, logs=logs)
            losses = model.train_step(data)

            values = list(losses.items())
            pb.add(1, values=values)

            callbacks.on_train_batch_end(batch_nr, logs=losses)


        # Validation 
        for (batch_nr, data) in enumerate(val_generator):
            
            losses = model.test_step(data)

            #for key, v in losses.items():
            #    batch_val_loss[key].append(v)

            values = list(losses.items())
            pb2.add(1, values=values)

            callbacks.on_test_batch_end(batch_nr, logs=losses)
        
        # On-Epoch-End
        callbacks.on_epoch_end(epoch, logs=logs)

    # On-Train-End
    callbacks.on_train_end(logs=logs)

    val_generator.on_epoch_end()
    train_generator.on_epoch_end()

    return

if __name__ == '__main__':
    try:
        #custom_train_loop()
        dotfit()
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")
    finally:
        print(f"Done.")

