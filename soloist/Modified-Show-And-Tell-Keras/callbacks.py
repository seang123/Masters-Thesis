import keras
import numpy as np
import csv

class EarlyStoppingByLossVal(keras.callbacks.Callback):
        def __init__(self, monitor='val_loss', value=0.00001, verbose=1):
            super(keras.callbacks.Callback, self).__init__()
            self.monitor = monitor
            self.value = value
            self.verbose = verbose

        def on_epoch_end(self, epoch, logs={}):
            current = logs.get(self.monitor)
            if current is None:
                print(f"Early stopping requires {self.monitor} be available!")

            if current < self.value:
                if self.verbose > 0:
                    print(f"Epoch {epoch}: early stopping. Loss < {self.value}")
                self.model.stop_training = True

class Predict(keras.callbacks.Callback):
    pass

class EpochHistory(keras.callbacks.Callback):
    
    def __init__(self, file_location):
        self.f = open(f"{file_location}/training_log.csv")
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        pass


class BatchLoss(keras.callbacks.Callback):

    def __init__(self, file_name, location):

        # training store
        self.f = open(f"{location}/{file_name}", 'w+', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(['loss', 'accuracy'])

        # validation store
        self.g = open(f"{location}/val_{file_name}", 'w+', newline='')
        self.val_writer = csv.writer(self.g)
        self.val_writer.writerow(['val_loss', 'val_accuracy'])

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_losses = []

    def on_epoch_end(self, epoch, logs=None):
        max_batch_loss = max(self.batch_losses)
        min_batch_loss = min(self.batch_losses)
        argmax = lambda array: max(zip(array, range(len(array))))[1]
        argmin = lambda array: min(zip(array, range(len(array))))[1]
        max_batch_loss_idx = argmax(self.batch_losses)
        min_batch_loss_idx = argmin(self.batch_losses)
        print(f"Max batch loss: {max_batch_loss:.4f} ({max_batch_loss_idx})\nMin batch loss: {min_batch_loss:.4f} ({min_batch_loss_idx})\n")

    def on_train_batch_end(self, batch, logs=None):
        loss = logs['loss']
        accuracy = logs['accuracy']
        self.batch_losses.append(loss)

        self.writer.writerow([loss, accuracy])

    def on_test_batch_end(self, batch, logs=None):
        loss = logs['loss']
        accuracy = logs['accuracy']

        self.val_writer.writerow([loss, accuracy])


    def on_train_end(self, logs=None):
        self.f.close()
        self.g.close()
