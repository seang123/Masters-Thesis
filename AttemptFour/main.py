import yaml
import time
import os, sys
import tensorflow as tf
from tensorflow.keras.utils import Progbar
import numpy as np
from Model import NIC
from DataLoaders import load_avg_betas as loader
from Callbacks import BatchLoss
from tensorflow.keras.callbacks import CSVLogger

gpu_to_use = 2

# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for i in range(0, len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')


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


# Load data
tokenizer, _ = loader.build_tokenizer(config['dataset']['captions_path'], config['top_k'])

nsd_keys, _ = loader.get_nsd_keys(config['dataset']['nsd_dir'])
shr_nsd_keys = loader.get_shr_nsd_keys(config['dataset']['nsd_dir'])

train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
val_keys = shr_nsd_keys

train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'])
val_pairs   = loader.create_pairs(val_keys, config['dataset']['captions_path'])

print(f"train_pairs: {len(train_pairs)}")
print(f"val_apirs  : {len(val_pairs)}")

train_generator = loader.batch_generator(train_pairs,
        config['dataset']['betas_path'],
        config['dataset']['captions_path'],
        tokenizer,
        config['batch_size'],
        config['max_length'],
        config['top_k'],
        config['units']
)

val_generator = loader.batch_generator(val_pairs,
        config['dataset']['betas_path'],
        config['dataset']['captions_path'],
        tokenizer,
        config['batch_size'],
        config['max_length'],
        config['top_k'],
        config['units']
)
print("data loaded successfully")

def gen_train():
    return train_generator
def gen_val():
    return val_generator


train_dataset = tf.data.Dataset.from_generator(loader.batch_generator, args=[
    val_pairs,
    config['dataset']['betas_path'],
    config['dataset']['captions_path'],
    tokenizer,
    config['batch_size'],
    config['max_length'],
    config['top_k'],
    config['units']
    ],
    output_types = ((tf.float32, tf.int32, tf.float32, tf.float32), tf.int32)
)


print("tf.data.Dataset 's generated")

# Setup optimizer 
optimizer = tf.keras.optimizers.Adam(learning_rate=config['alpha'])
loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        reduction='none'
)

# Setup Checkpoint handler
# TODO
start_epoch = 0


# Setup Model
nic = NIC.NIC(
        config['input']['vc'],
        config['units'], 
        config['embedding_dim'], 
        config['top_k'],
        config['max_length'],
        config['dropout'],
        config['input_reg'],
        config['lstm_reg'],
        config['output_reg']
        )
model = NIC.CaptionGenerator(nic)
model.compile(optimizer, loss_object, run_eagerly=True)
#print(model.summary())

"""
model.fit(train_dataset,
        epochs = config['epochs'],
        verbose = 1,
        callbacks = [],
        validation_data = val_generator,
        initial_epoch = start_epoch,
        steps_per_epoch = len(train_pairs)/config['batch_size'],
        validation_steps = len(val_pairs)/config['batch_size'],
)
"""

batch_loss_writer = BatchLoss.BatchLoss(f"{run_path}/batch_training_log.csv")
csv_logger = CSVLogger(f"{run_path}/training_log.csv")
_callbacks = [batch_loss_writer]

callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=True, model=model)

logs = {}
callbacks.on_train_begin(logs=logs)


def custom_train_loop():
    print(f"------\nRunning custom training loop")
    print(f"Running for {config['epochs'] - start_epoch} epochs\n------")

    # Train for N epochs
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nepoch {epoch+1}/{config['epochs']}")
        epoch_start_time = time.time()

        callbacks.on_epoch_begin(epoch, logs=logs)

        epoch_ce_loss     = []
        epoch_l2_loss     = []
        epoch_ce_loss_val = []
        epoch_l2_loss_val = []

        epoch_accuracy     = []
        epoch_accuracy_val = []

        pb = Progbar(len(train_pairs)/config['batch_size'], stateful_metrics=['loss', 'l2', 'accuracy'])

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
            
            losses = model.test_step(data)
            ce_loss, l2_loss, accuracy = losses.values()

            epoch_ce_loss_val.append( ce_loss )
            epoch_l2_loss_val.append( l2_loss )
            epoch_accuracy_val.append( accuracy )

        #callbacks.on_epoch_end(epoch, logs=logs)

    #callbacks.on_train_end(logs=logs)

    return

if __name__ == '__main__':
    custom_train_loop()
    print(f"Done.")

