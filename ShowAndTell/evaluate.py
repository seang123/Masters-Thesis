import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nsd_access import NSDAccess
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils
import pandas as pd
from model import Encoder, Decoder, CaptionGenerator
import tensorflow as tf


# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

class Evaluate():
    """
    Class for generating captions for a given image from the NSD dataset

    Loads the pre-trained Encoder/Decoder networks 
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.nsd_loader = NSDAccess("/home/seagie/NSD")
        self.nsd_loader.stim_descriptions = pd.read_csv(self.nsd_loader.stimuli_description_file, index_col=0)
        
        # load annotations
        self.annt_dict = utils.load_json("../modified_annotations_dictionary.json")
    
        # load validation set
        if os.path.exists("images_val_keys.txt"):
            self.val_keys = np.loadtxt("images_val_keys.txt", dtype=np.int32)
        else:
            self.val_keys = np.arange(0, 73000, dtype=np.int32)

        embedding_dim = 256 # was 256
        units = 512 # was 512
        self.top_k = 5000
        vocab_size = self.top_k + 1

        # load tokenizer
        self.create_tokenizer()

        # init the Network
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, units, vocab_size)
        self.model = CaptionGenerator(self.encoder, self.decoder, self.tokenizer, self.max_length())

        # load Network weights
        self.load_checkpoint()

        # load features
        self.image_features = np.load("img_features_vgg16").astype(np.float32)


    def gen_prediction(self, image_id=None):
        """

        Return:
            image_id - image id 
            result   - generated caption
            real cap - actual captions
        """

        if image_id == None:
            image_id = np.random.choice(self.val_keys, 1)[0]
       
        hidden = self.decoder.reset_state(batch_size=1)
        
        # load CNN output
        image_features = self.get_img_feature(image_id)
        print("img features:", image_features.shape)
        #image_features = tf.reshape(image_features, (image_features.shape[0],-1,image_features.shape[3]))
        image_features = tf.reshape(image_features, (1, 4096))

        # pass through encoder
        features = self.model.encoder(image_features)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        max_length = self.max_length() 
        for i in range(0, max_length):
            predictions, hidden = self.model.decoder((dec_input, features, hidden))

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return image_id, result, self.get_captions(image_id)

            dec_input = tf.expand_dims([predicted_id], 0)

        return image_id, result, self.get_captions(image_id) 

    def create_tokenizer(self):
        """
        Create the tokenizer
        """

        test_captions = []

        # Put all captions into a single list
        for i in range(0, len(self.val_keys)):
            test_captions.extend(self.annt_dict[str(i)])

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

        self.tokenizer.fit_on_texts(test_captions)

        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'


        # Create the tokenized vectors
        test_seqs = self.tokenizer.texts_to_sequences(test_captions)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        max_length = self.max_length()
        self.cap_vector = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, maxlen=max_length, padding='post')

    def max_length(self):
        # returns length of longest caption
        return max(len(i) for x in self.annt_dict.values() for i in x)


    def get_img_feature(self, image_id):
        # for a given image_id get its features from img_features_binary 
        return self.image_features[image_id,:]

    def save_fig(self, image_id):
        # save given image to a .png file
        img = self.nsd_loader.read_images(image_id)
        fig = plt.figure()
        plt.imshow(img)
        plt.title(f"img: {image_id}")
        plt.savefig(f"{self.save_dir}/test_img_{image_id}.png")
        plt.close(fig)

    def get_captions(self, img_id: np.int32):
        # returns all captions for a given image id
        return self.annt_dict[str(img_id)]

    def load_checkpoint(self):

        
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')


        checkpoint_path = f"./checkpoints/train"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                           decoder=self.decoder,
                           optimizer = optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

            print("Checkpoint loaded!")
