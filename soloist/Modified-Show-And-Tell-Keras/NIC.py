'''
File to define the model structure of NIC, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

model: Define the NIC training model

greedy_inference_model: Define the model used in greedy search.
                        Please initialize it with trained NIC model by load_weights()

image_dense_lstm: Define the encoding part of model used in beam search
                  Please initialize it with trained NIC model by load_weights()

text_emb_lstm: Define the decoding part of model used in beam search
               Please initialize it with trained NIC model by load_weights()
'''

import numpy as np
from keras import backend as K
from keras import regularizers, initializers
from keras.initializers import RandomUniform
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda, TimeDistributed, Bidirectional, LeakyReLU, ReLU)
from keras.models import Model
from parameters import params_dir

unit_size = params_dir['units']

def model(vocab_size, input_size, max_len, reg):

    # Image embedding
    #inputs1 = Input(shape=(2048,)) # was (2048,)
    inputs1 = Input(shape=(input_size,))
    #X_img = Dropout(0.1)(inputs1) # was 0.5
    X_img = Dense(unit_size, use_bias = True, # was False
                        kernel_regularizer=regularizers.l2(reg),
                        kernel_initializer=RandomUniform(-0.08, 0.08),
                        name = 'dense_img')(inputs1)#(X_img)
    #X_img = Dropout(0.3)(X_img) # not here originally 
    #X_img = LeakyReLU(alpha=0.1)(X_img)
    #X_img = ReLU()(X_img)
    # X_img = BatchNormalization(name='batch_normalization_img')(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    # Text embedding
    inputs2 = Input(shape=(max_len,))
    X_text = Embedding(vocab_size, unit_size, mask_zero = True, name = 'emb_text')(inputs2)
    #X_text = Dropout(0.3)(X_text)

    # Initial States
    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    LSTMLayer= LSTM(unit_size, 
                    return_sequences = True, 
                    return_state = True, 
                    kernel_regularizer = regularizers.l2(reg),
                    bias_regularizer = regularizers.l2(reg),
                    #dropout = 0.5,
                    name = 'lstm') 

    # Take image embedding as the first input to LSTM
    _, a, c = LSTMLayer(X_img, initial_state=[a0, c0])

    A, _, _ = LSTMLayer(X_text, initial_state=[a, c])
    output = TimeDistributed(Dense(vocab_size, activation='softmax',
                                    kernel_regularizer = regularizers.l2(reg), 
                                    bias_regularizer = regularizers.l2(reg), 
                                    kernel_initializer = RandomUniform(-0.08, 0.08)
                            ),
                            name = 'time_distributed_softmax')(A)

    return Model(inputs=[inputs1, inputs2, a0, c0], outputs=output, name='NIC')


def greedy_inference_model(vocab_size, input_size, max_len, reg):
    
    EncoderDense = Dense(unit_size, use_bias=True, name = 'dense_img')
    EmbeddingLayer = Embedding(vocab_size, unit_size, mask_zero = True, name = 'emb_text')
    LSTMLayer = LSTM(unit_size, return_state = True, name = 'lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name = 'time_distributed_softmax')
    # BatchNormLayer = BatchNormalization(name='batch_normalization_img')

    # Image embedding
    inputs1 = Input(shape=(input_size,)) # 2048
    X_img = EncoderDense(inputs1)
    #X_img = LeakyReLU(alpha=0.1)(X_img)
    #X_img = ReLU()(X_img)
    # X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    # Text embedding
    inputs2 = Input(shape=(1,))
    X_text = EmbeddingLayer(inputs2)

    # Initial States
    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    x = X_text

    outputs = []
    for i in range(max_len):
        
        a, _, c = LSTMLayer(x, initial_state=[a, c])
        output = SoftmaxLayer(a)
        outputs.append(output)
        x = Lambda(lambda x : K.expand_dims(K.argmax(x)))(output)
        x = EmbeddingLayer(x)

    return Model(inputs=[inputs1, inputs2, a0, c0], outputs=outputs, name='NIC_greedy_inference_v2')


def image_dense_lstm():

    raise Exception("Don't use this") 

    EncoderDense = Dense(unit_size, use_bias = False, name = 'dense_img')
    BatchNormLayer = BatchNormalization(name = 'batch_normalization_img')
    LSTMLayer = LSTM(unit_size, return_state = True, name = 'lstm')

    inputs = Input(shape=(5000,))
    X_img = EncoderDense(inputs)
    X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    return Model(inputs=[inputs, a0, c0], outputs=[a, c])


def text_emb_lstm(vocab_size):

    raise Exception("Don't use this")

    EmbeddingLayer = Embedding(vocab_size, unit_size, mask_zero = True, name='emb_text')
    LSTMLayer = LSTM(unit_size, return_state = True, name='lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name='time_distributed_softmax')

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))
    cur_word = Input(shape=(1,))

    X_text = EmbeddingLayer(cur_word)
    a, _, c = LSTMLayer(X_text, initial_state=[a0, c0])
    output = SoftmaxLayer(a)

    return Model(inputs=[a0, cur_word, c0], outputs=[output, a, c])
