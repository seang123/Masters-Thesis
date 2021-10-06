params_dir = {
        "data_dir": "./data/temp/",
        "epochs": 1, # 5 hours for 100 epochs @ bs=128
        "batch_size": 64,
        "max_length": 10, #20,

        "input": 62756, # MSCOCO = 4096 | VC = 62756 | PCA = 5000
        "top_k": 5000,
        "units": 128,
        "embedding_dim": 128,

        "L2_reg": 1, #0.01,# 1e-4,
        "LSTM_reg": 0,
        "out_reg": 0,
        "LR": 0.001,
        "lr_decay": 0, # 1e-4,
        }
