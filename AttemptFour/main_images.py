import yaml
import logging
import time
import os, sys
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np

from Model import img_NIC
#from DataLoaders import load_images as loader
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_image as generator

from Callbacks import BatchLoss, EpochLoss, WarmupScheduler, Predict
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

gpu_to_use = 1

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


with open("./config_img.yaml", "r") as f:
    config = yaml.safe_load(f)
    print(f"Config file loaded.")

run_name = config['run']
run_path = os.path.join(config['log'], run_name)

if not os.path.exists(run_path):
    os.makedirs(run_path)
    print(f"Creating training Log folder: {run_path}")
else:
    print(f"Training Log will be saved to: {run_path}")

with open(f"{run_path}/config_img.yaml", "w+") as f:
    yaml.dump(config, f)


logging.basicConfig(filename=f'{run_path}/log.log', filemode='w', level=logging.DEBUG)

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

## Parameters
vocab_size = config['top_k'] + 1

# Create train-val keys
train_keys, val_keys, test_keys = loader.get_nsd_keys('2')
print("train_keys:", train_keys.shape)
print("val_keys:", val_keys.shape)
print("test_keys:", test_keys.shape)

# Create train pairs
train_pairs = np.array(loader.create_pairs(train_keys))
val_pairs = np.array(loader.create_pairs(val_keys))
print("train_pairs:", train_pairs.shape)
print("val_pairs:  ", val_pairs.shape)
print("Captions created")

#tokenizer, _ = loader.build_tokenizer(all_keys, config['top_k'])
tokenizer = loader.load_tokenizer()
print("Tokenizer built")


def create_generator(data, cap, keys, training):
    return loader.data_generator(data, cap, keys, config['units'], vocab_size, config['batch_size'], training=training)

print("Caption data successfully loaded")

def lr_schedule(step):
    decay_steps = 10
    decay_rate = 0.01 
    inital_lr = 0.01
    final_lr = 0.0001
    #return max(inital_lr * decay_rate ** (step / decay_steps), final_lr)
    return 0.0001

# Setup optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.9, beta_2=0.98, epsilon=10.0e-9, clipnorm=config['clipnorm'])
loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction='none'
)

# Setup Model
model = img_NIC.NIC(
        config['group_size'],
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
model.compile(optimizer, loss_object, run_eagerly=True)


# Setup Checkpoint handler
"""
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager_best = tf.train.CheckpointManager(
            checkpoint, 
            directory=f"{run_path}/model",
            checkpoint_name = "model-best",
            max_to_keep=None)
manager_latest = tf.train.CheckpointManager(
            checkpoint, 
            directory=f"{run_path}/model", 
            checkpoint_name = "model-latest",
            max_to_keep=1)

status = checkpoint.restore(manager_latest.latest_checkpoint)
start_epoch = manager_latest.latest_checkpoint
if start_epoch == None:
    start_epoch = 0
else:
    start_epoch = int(start_epoch.split("-")[1])
"""
start_epoch = 0



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

logdir = f"./tb_logs/scalars/{config['run']}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
tensorboard_callback = TensorBoard(
        log_dir=logdir, 
        update_freq='batch',
        #histogram_freq=1,
        #write_graph=True,
        #write_images=True,
        #embeddings_freq=1,
        )

loss_history = EpochLoss.LossHistory(f"{run_path}/loss_history.csv", f"{run_path}")

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lr_schedule, verbose = 0)

_callbacks = [
        loss_history,
        checkpoint_best, 
        checkpoint_latest,
        tensorboard_callback,
        lr_scheduler,
]

callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=model)

logs = {}

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

    build_time = time.perf_counter()
    model(val_generator.__getitem__(0)[0])
    print(f"Model build time: {(time.perf_counter() - build_time):.3f}")
    print(model.summary())

    model.fit(
            train_generator,
            epochs=config['epochs'],
            steps_per_epoch= len(train_pairs)//config['batch_size'],
            batch_size=config['batch_size'],
            callbacks=_callbacks,
            validation_data = val_generator,
            validation_steps = len(val_pairs)//config['batch_size'],
            initial_epoch = 0
    )
    """
    print("--- Weights ---")
    m_weights = model.get_weights()
    np.save(f"{run_path}/pre_train_weights.npy", m_weights, allow_pickle=True)
    #print(model.get_layer('lstm'))
    """
    return


def custom_train_loop():
    print(f"------\nRunning custom training loop")
    print(f"Running for {config['epochs'] - start_epoch} epochs\n------")

    callbacks.on_train_begin(logs=logs)
    # Train for N epochs
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()
        callbacks.on_epoch_begin(epoch, logs=logs)
        
        # Instantiate new generator
        train_generator = create_generator(data_train, train_vector, train_keys)
        val_generator = create_generator(data_val, val_vector, val_keys)

        epoch_ce_loss     = []
        epoch_l2_loss     = []
        epoch_ce_loss_val = []
        epoch_l2_loss_val = []

        epoch_accuracy     = []
        epoch_accuracy_val = []

        pb = Progbar(data_train.shape[0]/config['batch_size'], stateful_metrics=['loss', 'l2', 'accuracy'])
        pb2 = Progbar(data_val.shape[0]/config['batch_size'], stateful_metrics=['val-loss', 'val-l2', 'val-accuracy'])

        # Training
        for (batch_nr, data) in enumerate(train_generator):
            callbacks.on_batch_begin(epoch, logs=logs)

            # data -> ([betas, cap_vector, a0, c0], target)
            losses = model.train_step(data)
            ce_loss, l2_loss, accuracy = losses.values()

            epoch_ce_loss.append( ce_loss )
            epoch_l2_loss.append( l2_loss )
            epoch_accuracy.append( accuracy )

            values = list(losses.items())
            pb.add(1, values=values)

            callbacks.on_train_batch_end(batch_nr, logs=losses)

        # Validation 
        for (batch_nr, data) in enumerate(val_generator):
            
            losses_val = model.test_step(data)
            ce_loss, l2_loss, accuracy = losses_val.values()

            epoch_ce_loss_val.append( ce_loss )
            epoch_l2_loss_val.append( l2_loss )
            epoch_accuracy_val.append( accuracy )

            values = list(losses_val.items())
            pb2.add(1, values=values)

            callbacks.on_test_batch_end(batch_nr, logs=losses_val)

        # On-Epoch-End
        callbacks.on_epoch_end(epoch, logs=logs)

    # On-Train-End
    callbacks.on_train_end(logs=logs)

    return

if __name__ == '__main__':
    try:
        #custom_train_loop()
        dotfit()
    except KeyboardInterrupt as e:
        print("--Keyboard Interrupt--")
    finally:
        print(f"Done.")


