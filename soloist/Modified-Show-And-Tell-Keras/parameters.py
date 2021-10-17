params_dir = {
        "data_dir": "./data/betas_ian/",
        "epochs": 10, # 5 hours for 100 epochs @ bs=128
        "batch_size": 128,
        "max_length": 10, #20,

        "input": 62756, # MSCOCO = 4096 | VC = 62756 | PCA = 5000
        "top_k": 5000,
        "units": 128,
        "embedding_dim": 128,

        "L2_reg": 0, #0.01,# 1e-4,
        "LSTM_reg": 0,
        "out_reg": 0,
        "LR": 0.001,
        "lr_decay": 0, # 1e-4,
        }
