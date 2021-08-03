import keras

class BatchHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        
        # per-batch loss
        self.loss = []
        # per-batch acc
        self.accuracy = []

        # per-epoch loss
        self.epoch_loss = []
        self.epoch_val_loss = []

        # per_epoch accuracy 
        self.epoch_accuracy = []
        self.epoch_val_accuracy = []

    def on_batch_end(self, batch, logs={}):
        """ Store loss after each batch
        Only valid for the training loss, not validation loss
        """
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        """ Store loss value at the end of each epoch
        """
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_accuracy.append(logs.get('accuracy'))

        self.epoch_val_loss.append(logs.get('val_loss'))
        self.epoch_val_accuracy.append(logs.get('val_accuracy'))

        # TODO: store json loss data after each epoch

    
    def get_loss(self):
        """ Return a dictionary of the losses 
        """
        loss_dict = {}
        loss_dict['epoch_loss'] = self.epoch_loss
        loss_dict['accuracy'] = self.accuracy
        loss_dict['epoch_loss'] = self.epoch_loss
        loss_dict['epoch_val_loss'] = self.epoch_val_loss
        loss_dict['epoch_accuracy'] = self.epoch_accuracy
        loss_dict['epoch_val_accuracy'] = self.epoch_val_accuracy

        return loss_dict
