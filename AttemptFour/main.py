import yaml
import logging
import time
import pandas as pd
import os, sys
import csv
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import Progbar
import numpy as np
from Model import NIC, lc_NIC
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator as generator
from Callbacks import BatchLoss, EpochLoss, WarmupScheduler, Predict
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
from datetime import datetime

gpu_to_use = 2

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
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

## Parameters
vocab_size = config['top_k'] + 1

#
## Load data
#
tokenizer, _ = loader.build_tokenizer(config['dataset']['captions_path'], config['top_k'])

nsd_keys, _ = loader.get_nsd_keys(config['dataset']['nsd_dir'])
shr_nsd_keys = loader.get_shr_nsd_keys(config['dataset']['nsd_dir'])

train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
val_keys = shr_nsd_keys

train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'])
val_pairs   = loader.create_pairs(val_keys, config['dataset']['captions_path'])
print(f"train_pairs: {len(train_pairs)}")
print(f"val_apirs  : {len(val_pairs)}")

print(train_pairs[0])

print("set train_pairs", len(list(set([int(i[0]) for i in train_pairs]))))
print("set val_pairs", len(list(set([int(i[0]) for i in val_pairs]))))

raise

# Returns a generator object 
create_generator = lambda pairs, training: loader.lc_batch_generator(pairs, 
            config['dataset']['betas_path'],
            config['dataset']['captions_path'],
            tokenizer,
            config['batch_size'],
            config['max_length'],
            vocab_size,
            config['units'],
            training=training
        )

print("data loaded successfully")



# Setup optimizer 
if config['optimizer'] == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['alpha'], beta_1 = 0.9, beta_2=0.98, epsilon=10.0e-9)#, clipnorm=0.2)
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
        loader.get_groups(config['embedding_dim'])[0], 
        loader.get_groups(config['embedding_dim'])[1],
        config['units'], 
        config['embedding_dim'], 
        vocab_size,
        config['max_length'],
        config['dropout_input'],
        config['dropout_features'],
        config['dropout_text'],
        config['input_reg'],
        config['lstm_reg'],
        config['output_reg']
        )

model.compile(optimizer, loss_object, run_eagerly=True)


# Setup Checkpoint handler
checkpoint_path = f"{run_path}/model/" 
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path_best = f"{checkpoint_path}" + "model-ep{epoch:03d}.h5"
checkpoint_best = ModelCheckpoint(checkpoint_path_best,
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        mode='min',
        period=1
        )
checkpoint_path_latest = f"{checkpoint_path}model-latest.h5"
checkpoint_latest = ModelCheckpoint(checkpoint_path_latest,
        monitor='val_loss',
        verbose=1, 
        save_weights_only = True, 
        save_best_only = False,
        mode = 'min',
        period=1)

#
## Callbacks
#
batch_loss_writer = BatchLoss.BatchLoss(f"{run_path}/batch_training_log.csv", f"{run_path}")
epoch_loss_writer = EpochLoss.EpochLoss(f"{run_path}/training_log.csv")

warmup = WarmupScheduler.WarmupScheduler(1, 0.00001, config['alpha'])

early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=10, min_delta=0.005, min_lr=0.00001)

logdir = f"./tb_logs/scalars/{config['run']}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
tensorboard_callback = TensorBoard(
        log_dir=logdir, 
        #histogram_freq=1,
        #write_graph=True,
        #write_images=True,
        #embeddings_freq=1,
        #profile_batch='200,220',
        )
file_writer = tf.summary.create_file_writer(logdir)

# Init a generator used during the predict callback
#val_generator_pred = create_generator(val_pairs, False)
#predict_callback = Predict.Predict(val_generator_pred, tokenizer, file_writer, config['max_length'], config['units'])


_callbacks = [
        batch_loss_writer, 
        epoch_loss_writer, 
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

    # tf.data Generator
    def create_train_gen():
        return loader.lc_batch_generator(train_pairs,
            config['dataset']['betas_path'],
            config['dataset']['captions_path'],
            tokenizer,
            config['batch_size'],
            config['max_length'],
            vocab_size,
            config['units'],
            training=True
            )
    def create_val_gen():
        return loader.lc_batch_generator(val_pairs,
            config['dataset']['betas_path'],
            config['dataset']['captions_path'],
            tokenizer,
            config['batch_size'],
            config['max_length'],
            vocab_size,
            config['units'],
            training=False
            )
    #train_generator = tf.data.Dataset.from_generator(create_train_gen,
    #        output_types=((np.float32, np.int32, np.float32, np.float32), np.int32))
    #val_generator = tf.data.Dataset.from_generator(create_val_gen,
    #        output_types=((np.float32, np.int32, np.float32, np.float32), np.int32, np.int32))

    # Raw generator
    #train_generator = create_generator(train_pairs, True)
    #val_generator = create_generator(val_pairs, False)

    train_generator = generator.DataGenerator(train_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, load_to_memory=True, shuffle=True, training=True)
    val_generator = generator.DataGenerator(val_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, load_to_memory=True, shuffle=True, training=True)

    model.fit(
            train_generator,
            epochs = config['epochs'],
            steps_per_epoch = len(train_pairs)//config['batch_size'],
            batch_size = config['batch_size'],
            callbacks = _callbacks,
            validation_data = val_generator,
            validation_steps = len(val_pairs)//config['batch_size'],
            initial_epoch = 0,
    )
    return




def custom_train_loop():
    print(f"------\nRunning custom training loop")
    print(f"for {config['epochs'] - start_epoch} epochs\n------")
    logging.info("training with custom training loop")

    train_generator = generator.DataGenerator(train_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, shuffle=True, training=True)
    val_generator = generator.DataGenerator(val_pairs, config['batch_size'], tokenizer, config['units'], config['max_length'], vocab_size, shuffle=True, training=True)

    grads = []

    # Train for N epochs
    callbacks.on_train_begin(logs=logs)
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch, logs=logs)
        
        # Reshuffle train/val pairs
        #train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'], seed = shuffle_seed)
        #val_pairs   = loader.create_pairs(val_keys,   config['dataset']['captions_path'], seed = shuffle_seed)
        # Instantiate new generator
        #train_generator = create_generator(train_pairs, True)
        #val_generator = create_generator(val_pairs, False)

        #batch_train_loss = defaultdict(list)
        #batch_val_loss = defaultdict(list)

        # Progress bar
        pb = Progbar(len(train_pairs)/config['batch_size'])#, stateful_metrics=['loss', 'l2', 'accuracy'])
        pb2 = Progbar(len(val_pairs)/config['batch_size'])#, stateful_metrics=['val-loss', 'val-l2', 'val-accuracy'])

        # Training
        for (batch_nr, data) in enumerate(train_generator):
            callbacks.on_batch_begin(epoch, logs=logs)
            #target = data[1]
            #target = tokenizer.sequences_to_texts(np.argmax(target, axis=2))

            # data -> ([betas, cap_vector, a0, c0], target)
            #print( "tf.executing_eagerly()", tf.executing_eagerly() )
            losses = model.train_step(data)

            grads.append(grad)

            #for key, v in losses.items():
            #    batch_train_loss[key].append(v)

            values = list(losses.items())
            pb.add(1, values=values)

            callbacks.on_train_batch_end(batch_nr, logs=losses)


        # Validation 
        for (batch_nr, data) in enumerate(val_generator):
            
            losses_val = model.test_step(data)

            #for key, v in losses.items():
            #    batch_val_loss[key].append(v)

            values = list(losses_val.items())
            pb2.add(1, values=values)

            callbacks.on_test_batch_end(batch_nr, logs=losses_val)
        
        # On-Epoch-End
        callbacks.on_epoch_end(epoch, logs=logs)
        #model.save_weights(f"{config['log']}/{config['run']}/model/checkpoints/checkpoint_latest")

    # On-Train-End
    callbacks.on_train_end(logs=logs)

    #df = pd.DataFrame(grads)
    #df.to_csv(f'{run_path}/df_grads.csv')
    #df.to_pickle(f'{run_path}/df_grads.csv')

    return

if __name__ == '__main__':
    try:
        #custom_train_loop()
        dotfit()
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")
    finally:
        print(f"Done.")

