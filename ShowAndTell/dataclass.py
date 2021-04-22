from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import numpy as np
import collections
import sys, os
import utils
import random
import tensorflow as tf

class Dataclass():
    """
    Class for handling loading of data for Show and Tell model    
    """

    def __init__(self, nr_training_samples, vocab_size):
        print("Loading annotations data...")
        self.nr_training_samples = nr_training_samples
        self.top_k = vocab_size
        self.annt_dict = utils.load_json("../modified_annotations_dictionary.json")     
        self.train_captions = self.get_annotations_list()

        self.tokenizer, self.train_seqs, self.cap_vector = self.create_tokenizer()


    def save_val_keys(self, path, data):
        ## save the validation set for later analysis
        with open(path, 'w') as f:
            for i in data:
                f.write("%s\n" % i) 
        return

    def get_tokenizer(self):
        return self.tokenizer

    def train_test_split(self, split_percentage = 0.95):
        """
        Split captions into train-test sets
        """
        
        img_name_vector = []
        for k, v in self.annt_dict.items():
            keys = [k for i in range(0, len(v))]
            img_name_vector.extend(keys)

        img_to_cap_vector = collections.defaultdict(list)
        for img, cap in zip(img_name_vector, self.cap_vector):
            img_to_cap_vector[img].append(cap)

        # shuffle keys
        img_keys = list(img_to_cap_vector.keys())
        random.shuffle(img_keys)


        slice_index = int(len(img_keys) * split_percentage)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

        ## Training Set
        img_name_train = [] # hold img id's for caption
        cap_train      = [] # hold captions
        for img in img_name_train_keys:
            cap_len = len(img_to_cap_vector[img])
            img_name_train.extend([img] * cap_len)
            cap_train.extend(img_to_cap_vector[img])

        ## Validation set
        img_name_val = [] 
        cap_val      = [] 
        for img in img_name_val_keys:
            cap_len = len(img_to_cap_vector[img])
            img_name_val.extend([img] * cap_len)
            cap_val.extend(img_to_cap_vector[img])

        return img_name_train, cap_train, img_name_val, cap_val

    def get_annotations_list(self):
        train_captions = []
        for i in range(0, self.nr_training_samples):
            train_captions.extend(self.annt_dict[str(i)])

        return train_captions


    def create_tokenizer(self):
        """
        Create the text tokenizer

        Return:
            tokenizer  - object
            train_seqs - captions as integer sequences
            cap_vector - padded caption integer sequences
        """
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

        tokenizer.fit_on_texts(self.train_captions)

        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        train_seqs = tokenizer.texts_to_sequences(self.train_captions)

        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=self.max_length(), padding='post')

        print("Tokenizer created!")
        return tokenizer, train_seqs, cap_vector

    def max_length(self):
        # return length of longest caption
        return 75
        #return max(len(i) for x in self.annt_dict.values() for i in x)

    @staticmethod
    def load_loss(path):
        if os.path.exists(path):
            x = np.load(path)
            return list(x['xtrain']), list(x['ytrain']), list(x['xtest']), list(x['ytest'])
        else:
            return [], [], [], []
