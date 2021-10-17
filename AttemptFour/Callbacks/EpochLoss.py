import numpy as np
from tensorflow.keras.callbacks import Callback
import csv


class EpochLoss(Callback):

    def __init__(self, file_name):
        self.f = open(file_name, "w+", newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['loss', 'reg_loss', 'accuracy', 'val_loss', 'val_reg_loss', 'val_accuracy'])


    def on_epoch_begin(self, epoch, logs=None):
        self.batch_loss     = []
        self.batch_reg_loss = []
        self.batch_acc      = []

        self.batch_loss_val = []
        self.batch_reg_loss_val = []
        self.batch_acc_val  = []

    def on_epoch_end(self, epoch, logs=None):
        self.writer.writerow(
            [
                np.mean(self.batch_loss), 
                np.mean(self.batch_reg_loss), 
                np.mean(self.batch_acc),
                np.mean(self.batch_loss_val),
                np.mean(self.batch_reg_loss_val),
                np.mean(self.batch_acc_val)
            ]
        )
        
    def on_train_batch_end(self, batch, logs=None):
        self.batch_loss.append( logs['loss'] )
        self.batch_reg_loss.append( logs['L2'] )
        self.batch_acc.append( logs['accuracy'] )

    def on_test_batch_end(self, batch, logs=None):
        self.batch_loss_val.append( logs['loss'] )
        self.batch_reg_loss_val.append( logs['L2'] )
        self.batch_acc_val.append( logs['accuracy'] )
        
    def on_train_end(self, logs=None):
        self.f.close()
