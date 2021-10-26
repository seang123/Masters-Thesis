import argparse
import os

from filelock import FileLock

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
import yaml
import tensorflow as tf
from Model import NIC, lc_NIC
from DataLoaders import load_avg_betas as loader
from Callbacks import BatchLoss, EpochLoss, WarmupScheduler
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from collections import defaultdict

"""
    Hyperparameter tune
    Creates an instance of the NIC model and trials several hp combinations
"""

gpu_to_use = 1
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_to_use)

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


vocab_size = config['top_k'] + 1

tokenizer, _ = loader.build_tokenizer(config['dataset']['captions_path'], config['top_k'])

nsd_keys, _ = loader.get_nsd_keys(config['dataset']['nsd_dir'])
shr_nsd_keys = loader.get_shr_nsd_keys(config['dataset']['nsd_dir'])

train_keys = [i for i in nsd_keys if i not in shr_nsd_keys]
val_keys = shr_nsd_keys

train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'])
val_pairs   = loader.create_pairs(val_keys, config['dataset']['captions_path'])

print(f"train_pairs: {len(train_pairs)}")
print(f"val_apirs  : {len(val_pairs)}")

## Function to create a data generator instance
create_generator = lambda pairs, units, training: loader.lc_batch_generator(pairs, 
            config['dataset']['betas_path'],
            config['dataset']['captions_path'],
            tokenizer,
            config['batch_size'],
            config['max_length'],
            vocab_size,
            units,
            training=training
        )
print("data loaded successfully")







def train_NIC(tune_config):
    """ Builds and and trains model """

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in range(0, len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')

    batch_size = tune_config['batch_size']

    optimizer = tf.keras.optimizers.Adam(learning_rate=tune_config['lr'], beta_1=0.9, beta_2=0.98, epsilon=10.0e-9,
            #clipvalue=tune_config['clipvalue'], 
            clipnorm=tune_config['clipnorm'],
    )
    loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction='none'
    )

    units = tune_config['units']
    embedding_dim = tune_config['embedding_dim']

    # Build model
    model = lc_NIC.NIC(
            loader.get_groups(embedding_dim)[0],
            loader.get_groups(embedding_dim)[1],
            units,
            embedding_dim,
            vocab_size,
            config['max_length'],
            tune_config['dropout_input'],
            tune_config['dropout_features'],
            tune_config['dropout_text'],
            tune_config['l2_in'],
            tune_config['l2_lstm'],
            tune_config['l2_out']
            )
    model.compile(optimizer, loss_object, run_eagerly=True)

    # Init generators
    train_generator = create_generator(train_pairs, units, True)
    val_generator = create_generator(val_pairs, units, False)

    model.fit(train_generator,
            epochs = tune_config['epochs'],
            verbose=0,
            steps_per_epoch = len(train_pairs)//config['batch_size'],
            batch_size = config['batch_size'],
            validation_data = val_generator,
            validation_steps = len(val_pairs)//config['batch_size'],
            initial_epoch = 0,
            callbacks=[TuneReportCallback({
                "mean_loss": "loss",
                "mean_val_loss": "val_loss",
                "mean_l2": "L2",
                "mean_accuracy": "accuracy",
                "mean_val_accuracy": "val_accuracy"
            })]
    )




def tune_NIC(num_training_iterations):
    """ Handles the ray.tune settings """
    
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        train_NIC,
        name="NIC_concat_lr",
        local_dir="./tb_logs/ray_results",
        scheduler=sched,
        metric="mean_loss",
        mode="min",
        stop={
            #"mean_loss": 1.0,
            "training_iteration": num_training_iterations
        },
        num_samples=50,
        resources_per_trial={
            "cpu": 3,
            "gpu": 0.20,
        },
        config={
            "batch_size": 128,
            "epochs": 20,
            "lr": tune.choice([0.001, 0.0001, 0.00001]),
            "clipnorm": 0.2, #tune.loguniform(0.001, 1),
            "clipvalue": 0, #tune.loguniform(0.1, 100),
            "embedding_dim": 64, #tune.choice([64, 128, 256]),
            "units": 512, #tune.choice([512]),
            "l2_in": 0.002, #tune.loguniform(0.00001, 1000),
            "l2_lstm": 0.0003, #tune.loguniform(0.00001, 100),
            "l2_out": 1.3e-5, #tune.loguniform(0.00001, 100), 
            "dropout_features": tune.uniform(0, 0.8),
            "dropout_input": tune.uniform(0, 0.5),
            "dropout_text": tune.uniform(0, 0.5),
        }
    )
    print("Best hyperparameters found were: ", analysis.best_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client."
    )
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=4)
    elif args.server_address:
        ray.init(f"ray://{args.server_address}")

    tune_NIC(num_training_iterations=5 if args.smoke_test else 300)


