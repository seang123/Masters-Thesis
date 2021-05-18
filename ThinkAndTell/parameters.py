parameters = {
        "EPOCHS": 10,
        "BATCH_SIZE": 64,
        "BUFFER_SIZE": 15000,
        "max_length": 15, # sentence length 
        "embedding_dim": 512, # was 512
        "units": 512, # was 512
        "top_k": 5000,
        "data_path": "./data/",
        "checkpoint_path": "./checkpoints/"
        }

# optimal values according to gridsearch on pca (non-exhaustive)
# units = 256
# embedding_dim = 128
