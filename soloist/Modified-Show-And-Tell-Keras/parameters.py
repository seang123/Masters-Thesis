params_dir = {
        "data_dir": "./data/anti_overfit/",
        "epochs": 1000, # 5 hours for 100 epochs @ bs=128
        "batch_size": 64,
        "max_length": 10, #20,

        "input": 5000, # MSCOCO = 4096 | VC = 62756 | PCA = 5000
        "top_k": 5000,
        "units": 512,
        "embedding_dim": 512,

        "L2_reg": 0, #1e-4,
        "LR": 0.0001,
        "lr_decay": 0, # 1e-4,
        }
