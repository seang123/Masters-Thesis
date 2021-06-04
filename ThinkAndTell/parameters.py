parameters = {
        "EPOCHS": 50,
        "BATCH_SIZE": 128,
        "LR": 0.0001, # learning rate
        "BUFFER_SIZE": 1000,
        "max_length": 15, # sentence length 
        "embedding_dim": 256,
        "units": 256,
        "L2": 0.1,
        "L2_lstm": 0.0001,
        "init_method": 'glorot_uniform',
        "dropout_fc": 0.2,
        "dropout_lstm": 0.2,
        "top_k": 5000,
        "data_path": "./data/PCA/",
        "checkpoint_path": "./checkpoints/"
        }

# optimal values according to gridsearch on pca (non-exhaustive)
# units = 256
# embedding_dim = 128
