import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np


class Predict(Callback):
    """
    Makes predictions on the validation dataset during training.
    """

    def __init__(self, batch_generator, tokenizer, file_writer, max_len, units):
        """
        batch_generator 
            generator object that yields betas, targets
        tokenizer 
            keras.tokenizer
        """
        self.batch_generator = batch_generator
        self.tokenizer = tokenizer
        self.file_writer = file_writer
        self.max_len = max_len
        self.units = units

        # Call the generator once initially to get some val samples
        self.samples = self.batch_generator.__next__()

    def on_epoch_end(self, epoch, logs=None):

        features, _, a0, c0 = self.samples[0]
        target = self.samples[1]
        nsd_keys = self.samples[2]

        start_seq = np.repeat([self.tokenizer.word_index['<start>']], features.shape[0])

        self.model.greedy_predict(features, tf.convert_to_tensor(a0), tf.convert_to_tensor(c0), start_seq, self.max_len, self.units)

        outputs = np.squeeze(outputs, axis = 2) # (10, 128, 5001)
        captions = np.argmax(outputs, axis = 2) # (10, 128)
        captions = np.transpose(captions, axes=[1,0]) # (128, 10)
        captions = self.tokenizer.sequences_to_texts(captions)

        target_sentences = targets_to_sentences(target) 

        for k, v in captions:
            with self.file_writer.as_default():
                tf.summary.text("Sample", f"Candidate: {v}\nTarget:  {target_sentences[k]}", step=0)


    def targets_to_sentences(self, targets: np.array) -> list:
        """ Returns a list of target sentences

        Parameters
        ----------
            target : ndarray
                the target caption as one-hot encoded words 
        Returns:
            sentence : [string]
        """
        return self.tokenizer.sequences_to_texts(np.argmax(targets, axis=2))
