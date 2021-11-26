import argparse
import os
import numpy as np

from filelock import FileLock

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
import yaml
import tensorflow as tf
from Model import NIC, lc_NIC
from DataLoaders import load_avg_betas as loader
from DataLoaders import data_generator_guse as generator
from Callbacks import BatchLoss, EpochLoss, WarmupScheduler
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from collections import defaultdict

"""
    Hyperparameter tune
    Creates an instance of the NIC model and trials several hp combinations
"""

gpu_to_use = 0
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

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])

vocab_size = config['top_k'] + 1

tokenizer, _ = loader.build_tokenizer(config['dataset']['captions_path'], config['top_k'])

nsd_keys, shr_nsd_key = loader.get_nsd_keys(config['dataset']['nsd_dir'])

train_keys = nsd_keys
val_keys   = shr_nsd_key

train_pairs = loader.create_pairs(train_keys, config['dataset']['captions_path'])
val_pairs   = loader.create_pairs(val_keys, config['dataset']['captions_path'])

print(f"train_pairs: {len(train_pairs)}")
print(f"val_pairs  : {len(val_pairs)}")

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
            #clipnorm=tune_config['clipnorm'],
    )
    loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction='none'
    )

    units = tune_config['units']
    embedding_dim = tune_config['embedding_dim']
    group_size = tune_config['group_size']

    # Build model
    model = lc_NIC.NIC(
            #loader.get_groups(config['embedding_features'])[0], 
            #loader.get_groups(config['embedding_features'])[1],
            loader.get_groups(group_size)[0], 
            loader.get_groups(group_size)[1],
            tune_config['units'], 
            embedding_dim, #tune_config['embedding_features'], 
            embedding_dim, #tune_config['embedding_text'],
            tune_config['attn_units'],
            vocab_size,
            config['max_length'],
            config['dropout_input'],
            config['dropout_features'],
            config['dropout_text'],
            config['dropout_attn'],
            config['dropout_lstm'],
            tune_config['input_reg'],
            tune_config['attn_reg'],
            tune_config['lstm_reg'],
            tune_config['output_reg']
            )
    model.compile(optimizer, loss_object, run_eagerly=True)


    train_generator = generator.DataGenerator(
            train_pairs, 
            tune_config['batch_size'], 
            tokenizer, 
            units,
            config['max_length'], 
            vocab_size, 
            nsd_keys = train_keys,
            pre_load_betas=False,
            shuffle=True, training=True)
    val_generator = generator.DataGenerator(
            val_pairs, 
            tune_config['batch_size'], 
            tokenizer, 
            units, 
            config['max_length'], 
            vocab_size, 
            nsd_keys = val_keys,
            pre_load_betas=False,
            shuffle=True, training=True)

    model.fit(train_generator,
            epochs = tune_config['epochs'],
            verbose=0,
            steps_per_epoch = len(train_pairs)//config['batch_size'],
            batch_size = config['batch_size'],
            validation_data = val_generator,
            validation_steps = len(val_pairs)//config['batch_size'],
            initial_epoch = 0,
            callbacks=[TuneReportCallback({
                "loss": "loss",
                "val_loss": "val_loss",
                "L2": "L2",
                "val_l2": "val_L2",
                "accuracy": "accuracy",
                "val_accuracy": "val_accuracy"
            })]
    )




def tune_NIC(num_training_iterations):
    """ Handles the ray.tune settings """
    
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20
    )

    analysis = tune.run(
        train_NIC,
        name="NIC_attention_fixed_idx",
        local_dir="./tb_logs/ray_results",
        scheduler=sched,
        metric="loss",
        mode="min",
        stop={
            #"mean_loss": 1.0,
            "training_iteration": num_training_iterations
        },
        num_samples=100,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0.20,
        },
        config={
            "batch_size": 64,
            "epochs": 30,
            "lr": 0.001, # tune.choice([0.001, 0.0001]),
            "clipnorm": 0, # 0.2, #tune.loguniform(0.001, 1),
            "clipvalue": 0, #tune.loguniform(0.1, 100),
            "group_size": 32, # tune.choice([32, 64]),
            #"embedding_features": 512, #tune.choice([64, 128, 256]),
            #"embedding_text": 512,
            "embedding_dim": tune.choice([128, 256, 512]),
            "units": 512, # tune.choice([256, 512, 1024]),
            "attn_units": tune.choice([8, 16, 32]),
            "input_reg": tune.loguniform(1.0e-5, 1),
            "lstm_reg": tune.loguniform(1.0e-5, 1),
            "output_reg": tune.loguniform(1.0e-5, 1), 
            "attn_reg": tune.loguniform(1.0e-5, 1),
            "dropout_features": 0, # tune.uniform(0, 0.8),
            "dropout_input": 0, # tune.uniform(0, 0.5),
            "dropout_text": 0, # tune.uniform(0, 0.5),
            "dropout_attn": 0, # tune.uniform(0, 0.5),
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


