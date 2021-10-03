parameters = {
        "EPOCHS": 2,
        "BATCH_SIZE": 128,
        "LR": 0.0001, # learning rate
        "BUFFER_SIZE": 2000,
        "max_length": 15, # sentence length 
        "embedding_dim": 512,
        "units": 512,
        "L2": 1e-4, # 1e-1,
        "L2_lstm": 1e-4, # 1e-3,
        "init_method": 'random_uniform', #'glorot_uniform',
        "dropout_fc": 0.3, # 0.5,
        "dropout_lstm": 0.3,
        "top_k": 5000,
        "data_path": "./data/jit_compile/",
        "checkpoint_path": "./checkpoints/",
        "SAVE_CHECKPOINT": True,
        "SAVE_DATA": True
        }

# optimal values according to gridsearch on pca (non-exhaustive)
# units = 256
# embedding_dim = 128
