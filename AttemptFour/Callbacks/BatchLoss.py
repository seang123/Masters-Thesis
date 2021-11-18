from tensorflow.keras.callbacks import Callback
import csv
from contextlib import redirect_stdout 


class BatchLoss(Callback):

    def __init__(self, file_name, summary_path):
        self.f = open(file_name, "w+", newline='')
        self.writer = csv.writer(self.f)
        #self.writer.writerow(['loss', 'reg_loss', 'accuracy'])
        self.summary_path = summary_path


    def on_epoch_begin(self, epoch, logs=None):
        self.batch_losses = []

    def on_epoch_end(self, epoch, logs=None):
        max_batch_loss = max(self.batch_losses)
        min_batch_loss = min(self.batch_losses)
        argmax = lambda array: max(zip(array, range(len(array))))[1]
        argmin = lambda array: min(zip(array, range(len(array))))[1]
        max_batch_loss_am = argmax(self.batch_losses)
        min_batch_loss_am = argmin(self.batch_losses)

        #print(f"\nBatch Loss:\n\t{max_batch_loss:.4f} ({max_batch_loss_am})\n\t{min_batch_loss:.4f} ({min_batch_loss_am})")

        if epoch == 0:
            with open(f'{self.summary_path}/modelsummary.txt', 'w') as f:
                with redirect_stdout(f):
                    self.model.summary()

    def on_train_batch_end(self, batch, logs=None):
        if batch == 0:
            self.writer.writerow(list(logs.keys()))
        self.batch_losses.append(logs['loss'])
        #self.writer.writerow(
        #    [logs['loss'], logs['L2'], logs['accuracy']]
        #)
        self.writer.writerow(
                list(logs.values())
        )
        
    def on_train_end(self, logs=None):
        self.f.close()
