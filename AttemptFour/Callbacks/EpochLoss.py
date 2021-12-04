import numpy as np
from tensorflow.keras.callbacks import Callback
import csv
import pandas as pd
import logging
from contextlib import redirect_stdout 

loggerA = logging.getLogger(__name__ + '.LossHistory')

class LossHistory(Callback):

    def __init__(self, file_name, summary_path):
        self.file_name = file_name
        self.summary_path = summary_path
        self.df = pd.DataFrame()
        self.cur_epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        logs['epoch'] = self.cur_epoch
        self.df = self.df.append(logs, ignore_index=True)

    def on_test_batch_end(self, batch, logs=None):
        logs = {f"val_{k}":v for k,v in logs.items()} # append val_ to name
        logs['epoch'] = self.cur_epoch
        
        self.df = self.df.append(logs, ignore_index=True)

    def on_epoch_end(self, epoch, logs=None):
        self.cur_epoch += 1

        self.df.to_csv(self.file_name, index=False)
        loggerA.info('saving loss history to csv')

        if epoch == 0:
            with open(f'{self.summary_path}/modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    self.model.summary()
                    loggerA.info(f"saving model summary to: {f.name}")

    def on_train_end(self, logs=None):
        self.df.to_csv(self.file_name, index=False) 



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
