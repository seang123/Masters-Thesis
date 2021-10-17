import tensorflow as tf
import numpy as np

class WarmupScheduler(tf.keras.callbacks.Callback):

    def __init__(self, n_warmup: int, initial_lr: float, final_lr: float):
        """
        Parameters
        ----------
            n_warmup : int
                how many batches to warm up for
            initial_lr : float
                the initial learning rate
            final_lr : float
                learning rate after warmup
        """
        super(WarmupScheduler, self).__init__()

        self.n_warmup = n_warmup
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        self.step_value = np.linspace(self.initial_lr, self.final_lr, n_warmup)

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer has no 'lr' attribute.")
        
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Simple constant warmup to final lr schedule
        if batch > n_warmup:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.final_lr)
        else:
            tf.keras.backend_set_value(self.model.optimizer.lr, self.step_value[batch])
            print("LR:", self.step_value[batch])




