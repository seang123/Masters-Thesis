parameters = {
        "EPOCHS": 15,
        "BATCH_SIZE": 128,
        "LR": 0.001, # learning rate
        "BUFFER_SIZE": 1000,
        "max_length": 15, # sentence length 
        "embedding_dim": 512,
        "units": 512,
        "L2": 1.0,
        "top_k": 5000,
        "data_path": "./data/fc_lstm_L2/",
        "checkpoint_path": "./checkpoints/"
        }

# optimal values according to gridsearch on pca (non-exhaustive)
# units = 256
# embedding_dim = 128
