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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 everything, 3 nothing
import tensorflow as tf
import json


# Allow memory growth on GPU devices 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in range(0, len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

class Evaluate():
    """
    Class for generating captions for a given image from the NSD dataset

    Loads the pre-trained Encoder/Decoder networks 
    """

    def __init__(self, save_dir, use_vgg16 = False):
        self.save_dir = save_dir
        self.use_vgg16 = use_vgg16
        self.nsd_loader = NSDAccess("/home/seagie/NSD")
        self.nsd_loader.stim_descriptions = pd.read_csv(self.nsd_loader.stimuli_description_file, index_col=0)
        
        # load annotations
        self.annt_dict = utils.load_json("../modified_annotations_dictionary.json")
    
        # load validation set
        if os.path.exists("images_val_keys.txt"):
            self.val_keys = np.loadtxt("images_val_keys.txt", dtype=np.int32)
        else:
            self.val_keys = np.arange(0, 73000, dtype=np.int32)

        # Model Parameter settings TODO: load automatically from modelsummary file
        embedding_dim = 512 # was 256
        self.units = 512 # was 512
        self.top_k = 5000
        vocab_size = self.top_k + 1

        # load tokenizer
        self.create_tokenizer()

        # init the Network
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, self.units, vocab_size, use_stateful=True)
        self.model = CaptionGenerator(self.encoder, self.decoder, self.tokenizer, self.max_length())

        # load Network weights
        self.load_checkpoint()

        # load features
        self.image_features = np.load("img_features_vgg16").astype(np.float32)
        if use_vgg16:
            image_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
            new_input = image_model.input
            hidden_layer = image_model.layers[-2].output
            self.vgg16 = tf.keras.Model(new_input, hidden_layer)

    def load_image_vgg16(self, idx: int):
        if not isinstance(idx, list):
            idx = [idx]
        img = self.nsd_loader.read_images(idx)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img


    def gen_prediction(self, image_id=None):
        """

        Return:
            image_id - image id 
            result   - generated caption
            real cap - actual captions
        """

        if image_id == None:
            image_id = np.random.choice(self.val_keys, 1)[0]

        print(f"Generating caption for image {image_id}.")
       
        hidden = self.decoder.reset_state(batch_size=1)

        if self.use_vgg16:
            print("Passing image through vgg16")
            img = self.load_image_vgg16(image_id)
            image_features = self.vgg16(img)
            print("img features:", image_features.shape)
        else: 
            # load CNN output
            image_features = self.get_img_feature(image_id)
            print("img features:", image_features.shape)
            #image_features = tf.reshape(image_features, (image_features.shape[0],-1,image_features.shape[3]))
            image_features = tf.reshape(image_features, (1, 4096))

        max_length = self.max_length() 

        # input should start as 1xN vector of zeros with only the first idx having the words '<start>'
        dec_input = [self.tokenizer.word_index['<start>']]
        paddings = tf.constant([[0,max_length-1]])
        dec_input = tf.pad(dec_input, paddings, "CONSTANT")
        dec_input = tf.expand_dims(dec_input, 0)

        ## first pass should include the image 
        features = self.model.encoder(image_features)


        result = []

        for i in range(0, max_length):
            # 1. pass in the image
            # 1.1 get hidden state
            
            # 2. create new lstm with hidden state
            # 2.1 get prediction, new hidden state
            # 2.2 repeat from 2.0
            if i == 0:
                pred, hidden, carry = self.model.decoder((dec_input, features), training=True) # (1,75,5001)(1,512)(1,512)
                state = pred[0, i, :]
                state = tf.expand_dims(state, 0)
            else:
                lstm_in = self.model.decoder.embedding(dec_input)
                pred, hidden, carry = tf.keras.layers.LSTM(units=self.units, return_sequences=True, return_state=True)(lstm_in, initial_state=[hidden, carry]) # Not yet tested with return_sequences=True
                pred = self.model.decoder.fc1(pred)
                pred = self.model.decoder.fc2(pred)

            #state = tf.expand_dims(state, 0)
            pred_id = tf.random.categorical(state, 1)[0][0].numpy()
            # add the last predicted word to the input
            dec_input = dec_input.numpy()
            dec_input[0, i] = pred_id
            dec_input = tf.convert_to_tensor(dec_input)
            pred_word = self.tokenizer.index_word[pred_id]
            result.append(pred_word)
            if pred_word == '<end>':
                return image_id, result, self.get_captions(image_id)

        return image_id, result, self.get_captions(image_id)

    def gen_prediction2(self, image_id=None):

        if image_id == None:
            image_id = np.random.choice(self.val_keys, 1)[0]

        # load CNN output
        image_features = self.get_img_feature(image_id)
        print("img features:", image_features.shape)
        #image_features = tf.reshape(image_features, (image_features.shape[0],-1,image_features.shape[3]))
        image_features = tf.reshape(image_features, (1, 4096))
        features = self.model.encoder(image_features)

        dec_input = [self.tokenizer.word_index['<start>']]
        dec_input = tf.expand_dims(dec_input, 0)
        print("dec_input", dec_input)
        
        result = []
        for i in range(0, self.max_length()):
            pred, hidden, carry = self.model.decoder((dec_input, features), training=True)
            word = tf.expand_dims(pred[0, i, :], 0)

            pred_id = tf.random.categorical(word, 1)[0][0].numpy()
            dec_input = dec_input.numpy()
            print("pred_id", pred_id)
            dec_input = np.append(dec_input, np.array([[pred_id]]), 0)
            dec_input = tf.convert_to_tensor(dec_input)
            pred_word = self.tokenizer.index_word[pred_id]
            result.append(pred_word)
            if pred_word == '<end>':
                return image_id, result, self.get_captions(image_id)

        return image_id, result, self.get_captions(image_id)



    def create_tokenizer(self):
        """
        Create the tokenizer
        """

        test_captions = []

        # Put all captions into a single list
        for i in range(0, len(self.val_keys)):
            test_captions.extend(self.annt_dict[str(i)])


        if os.path.exists("./tokenizer_config.txt"):
            with open('./tokenizer_config.txt') as json_file:
                json_string = json.load(json_file)
                self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
                print("tokenizer loaded from config file")
        else:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.top_k, oov_token='<unk>', filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~\t\n ')

            self.tokenizer.fit_on_texts(test_captions)

            self.tokenizer.word_index['<pad>'] = 0
            self.tokenizer.index_word[0] = '<pad>'
            print("new tokenizer created")


        # Create the tokenized vectors
        test_seqs = self.tokenizer.texts_to_sequences(test_captions)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        max_length = self.max_length()
        self.cap_vector = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, maxlen=max_length, padding='post')

    def max_length(self):
        # returns length of longest caption
        return 75 
        #return max(len(i) for x in self.annt_dict.values() for i in x)


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
            # expect_partial() hides the warning that the optimizer state wasn't loaded. This is fine since we arn't interested in training here.

            print("Checkpoint loaded!")
