'''
File to define inference and BLEU evaluation method of NIC, 
including how to generate captions by given image use greedy or beam search, 

based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

'''

import math
import os

import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from NIC import *
from preprocessing.image import *
from preprocessing.text import *

#from data_loader import load_data_pca, load_data_img, data_generator
from data_loader import *
import my_utils as uu
import sys, os
from nsd_access import NSDAccess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from parameters import params_dir


#gpu_to_use = 0
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for i in range(0, len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[i], True)
#tf.config.set_visible_devices(physical_devices[gpu_to_use], 'GPU')

def load_filckr8k_features(dict_dir, dataset_dir):
    
    '''
    Helper function to load pre-extracted image features
    '''
    
    img_ids = []
    with open(dataset_dir, 'r') as f:
        for line in f.readlines():
            img_ids.append(os.path.splitext(line)[0])

    features = load_features(dict_dir, dataset_dir, repeat_times = 1)

    return img_ids, features


def decoder(inf_model, tokenizer, features, post_process = True):

    '''
    Helper funtion of greedy search
    '''

    assert(features.shape[0]>0 and features.shape[1] == 2048)

    N = features.shape[0]

    startseq = np.repeat([tokenizer.word_index['<start>']], N)
    a0 = np.zeros([N, unit_size])
    c0 = np.zeros([N, unit_size])

    # output dims: [32, N, 7378]
    y_preds = np.array(inf_model.predict([features, startseq, a0, c0], verbose = 1))

    # output dims: [N, 32, 7378]
    y_preds = np.transpose(y_preds, axes = [1,0,2])
    
    sequences = np.argmax(y_preds, axis = -1)
    sents = tokenizer.sequences_to_texts(sequences)

    if post_process:
        # post processing: 'endseq'
        sents_pp = []
        for sent in sents:
            if '<end>' in sent.split():
                words = sent.split()
                sents_pp.append(' '.join(words[:words.index('<end>')]))
            else:
                sents_pp.append(sent)
        sents = sents_pp

    return sents


def beam_search(decoder_model, a0 , c0, tokenizer, beam_width, max_len, alpha = 0.7):

    '''
    Helper funtion of beam search

    decoder_model: A NIC_text_emb_lstm like model
    '''

    assert(a0.shape == (1, unit_size) and c0.shape == (1, unit_size) and isinstance(beam_width, int) and
             beam_width > 0 and max_len > 0)

    # === first step ====
    start_word = np.array([tokenizer.word_index['startseq']])

    output, a, c = decoder_model.predict([a0, start_word, c0], verbose=0)

    assert(len(output.shape)==2 and beam_width<=output.shape[1])

    # === define data structure and initialization====
    
    seeds = np.argpartition(-output, beam_width, axis=-1)[0, :beam_width]
    start_words = np.array(seeds)
    next_activates = np.repeat(a, beam_width, axis = 0)
    next_cells = np.repeat(c, beam_width, axis = 0)

    scores = [math.log(output[0, i]) for i in seeds]
    routes = [[i] for i in seeds]
    res = {'scores':[], 'routes':[]}

    # === search ====
    for i in range(max_len-1):

        outputs, activations, cells = decoder_model.predict([next_activates, start_words, next_cells], 
                                                            verbose=0)

        # pick <beam_width> highest scores from every route as a candidate
        candidates = np.argpartition(-outputs, beam_width, axis=-1)[:,:beam_width]
        # r <----> i-th in scores and routes, c is the index of vocabulary
        candidates = [(r, c) for r in range(candidates.shape[0]) for c in candidates[r,:]]
        # calculate score according to the candidates
        candidates_scores = np.array([scores[r] + math.log(outputs[r, c]) for r, c in candidates])
        # consider the length of the current sentence
        #weigthed_scores = 1/(i+1)**alpha * candidates_scores
        if beam_width < len(candidates):
            choosen_candidates = np.argpartition(-candidates_scores, beam_width)[:beam_width]
        else:
            choosen_candidates = np.arange(0, len(candidates))

        # update scores, routes
        # construct new start_words, activations, cells
        start_words = []
        next_activates = []
        next_cells = []
        updated_scores = []
        updated_routes = []
        for idx in choosen_candidates:
            r, c = candidates[idx]
            if c == tokenizer.word_index['endseq']:
                res['routes'].append(routes[r])
                
                if i != 0:
                    res['scores'].append(1/len(routes[r])**alpha * candidates_scores[idx])
                else:
                    res['scores'].append(-math.inf)
                
                beam_width -= 1
            else:
                start_words.append(c)
                next_activates.append(activations[r, :])
                next_cells.append(cells[r, :])

                updated_scores.append(candidates_scores[idx])
                updated_routes.append(routes[r]+[c])

        start_words = np.array(start_words)
        next_activates = np.array(next_activates)
        next_cells = np.array(next_cells)
        scores = updated_scores
        routes = updated_routes

        if beam_width <= 0:
            break

    res['scores'] += [1/len(routes[i])**alpha * scores[i] for i in range(len(scores))]
    res['routes'] += routes

    return res


def bleu_evaluation_greedy(model_dir, tokenizer, test_references, test_features, max_len):

    vocab_size = tokenizer.num_words or (len(tokenizer.word_index)+1)

    # prepare inference model
    NIC_inference = greedy_inference_model(vocab_size, max_len)
    NIC_inference.load_weights(model_dir, by_name = True, skip_mismatch=True)

    test_candidates = decoder(NIC_inference, tokenizer, test_features, True)

    assert(len(test_references) == len(test_candidates))

    scores = {'BLEU-1':[], 'BLEU-2':[], 'BLEU-3':[], 'BLEU-4':[]}
    for i in range(len(test_candidates)):
        references = [r.lower().split() for r in test_references[i]]
        candidate = test_candidates[i].split()

        scores['BLEU-1'].append(sentence_bleu(references, candidate, weights=(1.0, 0, 0, 0), 
                                smoothing_function=SmoothingFunction().method1))
        scores['BLEU-2'].append(sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), 
                                smoothing_function=SmoothingFunction().method1))
        scores['BLEU-3'].append(sentence_bleu(references, candidate, weights=(0.333, 0.333, 0.333, 0), 
                                smoothing_function=SmoothingFunction().method1))
        scores['BLEU-4'].append(sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), 
                                smoothing_function=SmoothingFunction().method1))

    scores['BLEU-1'] = np.average(scores['BLEU-1'])
    scores['BLEU-2'] = np.average(scores['BLEU-2'])
    scores['BLEU-3'] = np.average(scores['BLEU-3'])
    scores['BLEU-4'] = np.average(scores['BLEU-4'])

    print('BLEU-1', scores['BLEU-1'])
    print('BLEU-2', scores['BLEU-2'])
    print('BLEU-3', scores['BLEU-3'])
    print('BLEU-4', scores['BLEU-4'])

    return test_candidates


def bleu_evaluation_beam_search(model_dir, tokenizer, test_references, test_features, max_len, beam_width, alpha):

    vocab_size = tokenizer.num_words or (len(tokenizer.word_index)+1)

    # prepare inference model
    NIC_text_emb_lstm = text_emb_lstm(vocab_size)
    NIC_text_emb_lstm.load_weights(model_dir, by_name = True, skip_mismatch=True)
    NIC_image_dense_lstm = image_dense_lstm()
    NIC_image_dense_lstm.load_weights(model_dir, by_name = True, skip_mismatch=True)

    feature_size = test_features.shape[0]
    a0, c0 = NIC_image_dense_lstm.predict([test_features, np.zeros([feature_size, unit_size]), np.zeros([feature_size, unit_size])])

    # generate candidate sentences
    test_candidates = []
    for i in range(feature_size):
        res = beam_search(NIC_text_emb_lstm, a0[i, :].reshape(1,-1), c0[i, :].reshape(1,-1), tokenizer, beam_width, max_len, alpha)
        best_idx = np.argmax(res['scores'])
        test_candidates.append(tokenizer.sequences_to_texts([res['routes'][best_idx]])[0])

    assert(len(test_references) == len(test_candidates))

    scores = {'BLEU-1':[], 'BLEU-2':[], 'BLEU-3':[], 'BLEU-4':[]}
    for i in range(len(test_candidates)):
        references = [r.split() for r in test_references[i]]
        candidate = test_candidates[i].split()
        scores['BLEU-1'].append(sentence_bleu(references, candidate, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1))
        scores['BLEU-2'].append(sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1))
        scores['BLEU-3'].append(sentence_bleu(references, candidate, weights=(0.333, 0.333, 0.333, 0), smoothing_function=SmoothingFunction().method1))
        scores['BLEU-4'].append(sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1))

    print('BLEU-1', np.average(scores['BLEU-1']))
    print('BLEU-2', np.average(scores['BLEU-2']))
    print('BLEU-3', np.average(scores['BLEU-3']))
    print('BLEU-4', np.average(scores['BLEU-4']))

    return test_candidates


def evaluate_one(model_dir, method='b', beam_width = 5, alpha = 0.7):
    '''
    To evaluate one model by BLEU in a directory and return relevant information

    inputs:

    b - beam_search
        beam_width and alpha only effective when method = 'b'
    g - gready search

    outputs:

    1. test_ids: relevant image file names
    2: test_references: relevant ground truth sentences
    3: candidates: sentences generated by NIC geedy or beam search inference model
    (the order is the same)

    examples:

    model_dir = './model-params/xxxx.h5' # a model weight file address
    img_ids, refs, cands = evaluate_all(model_dir, method='b', beam_width = 5, alpha = 0.7)
    '''

    #dict_dir = './datasets/features_dict.pkl'
    #train_dir = './datasets/Flickr8k_text/Flickr_8k.trainImages.txt'
    #test_dir = './datasets/Flickr8k_text/Flickr_8k.testImages.txt'
    #token_dir = './datasets/Flickr8k_text/Flickr8k.token.txt'

    max_len = 24

    tokenizer = create_tokenizer(train_dir, token_dir)
    filter_tokenizer = create_tokenizer(test_dir, token_dir, use_all=True)

    test_ids, test_features = load_filckr8k_features(dict_dir, test_dir)
    all_sents = load_token_text(token_dir)
    test_references = [all_sents[id] for id in test_ids]

    test_references = clean_test_sentences(filter_tokenizer, test_references)

    if method == 'g':
        candidates = bleu_evaluation_greedy(model_dir, tokenizer, test_references, test_features, max_len)
    elif method == 'b':
        candidates = bleu_evaluation_beam_search(model_dir, tokenizer, test_references, test_features, max_len, beam_width, alpha)
    
    return test_ids, test_references, candidates


def evaluate_all(models_dir, method='b', beam_width = 5, alpha = 0.7):
    '''
    To evaluate all models by BLEU in a directory and return relevant information

    inputs:

    b - beam_search
        beam_width and alpha only effective when method = 'b'
    g - gready search

    outputs:

    1. test_ids: relevant image file names
    2. model_ids: relevant model file names
    3: test_references: relevant ground truth sentences
    4: candidates_list: sentences generated by NIC geedy or beam search inference model
    (the order is the same)

    examples:

    models_dir = './model-params' # the directory with a lot of same model structure weights
    img_ids, mid, refs, cands_list = evaluate_all(models_dir, method='b', beam_width = 5, alpha = 0.7)
    '''

    dict_dir = './datasets/features_dict.pkl'
    train_dir = './datasets/Flickr8k_text/Flickr_8k.trainImages.txt'
    test_dir = './datasets/Flickr8k_text/Flickr_8k.testImages.txt'
    token_dir = './datasets/Flickr8k_text/Flickr8k.token.txt'

    max_len = 24

    tokenizer = create_tokenizer(train_dir, token_dir)
    filter_tokenizer = create_tokenizer(test_dir, token_dir, use_all=True)

    test_ids, test_features = load_filckr8k_features(dict_dir, test_dir)
    all_sents = load_token_text(token_dir)
    test_references = [all_sents[id] for id in test_ids]
    
    test_references = clean_test_sentences(filter_tokenizer, test_references)

    candidates_list = []
    model_ids = []
    for model_file in os.listdir(models_dir):
        print('----------', model_file)
        model_ids.append(model_file)
        model_path = models_dir + '/' + model_file
        if method == 'g':
            candidates = bleu_evaluation_greedy(model_path, tokenizer, test_references, test_features, max_len)
        elif method == 'b':
            candidates = bleu_evaluation_beam_search(model_path, tokenizer, test_references, test_features, max_len, beam_width, alpha)
        candidates_list.append(candidates)
    
    return test_ids, model_ids, test_references, candidates_list

def my_decoder(inf_model, tokenizer, features, input_size, unit_size, post_process = True):
    '''Calls the inference version of the training model for word-by-word prediction giving just the input image and a starting word
    '''

    assert(features.shape[0]>0 and features.shape[1] == input_size)

    # Batch size
    N = features.shape[0]

    # Initial word token and hidden state
    startseq = np.repeat([tokenizer.word_index['<start>']], N)
    a0 = np.zeros([N, unit_size])
    c0 = np.zeros([N, unit_size])

    ## Call inference model
    # output dims: [32, N, 7378]
    y_preds = np.array(inf_model.predict([features, startseq, a0, c0], verbose = 1)) # verbose = 1 for progress bar

    # Reshape output
    # output dims: (N, max_length, vocab_size) ie. (20, 20, 5000)
    y_preds = np.transpose(y_preds, axes = [1,0,2]) 

    # At every timestep the output is a (5000,) vector. The argmax of that vector is the most likely word. 
    # So a (20, 5000) vector gives the 20 most likely words at each timestep
    sequences = np.argmax(y_preds, axis = -1) # out: (N, max_length)
    sents = tokenizer.sequences_to_texts(sequences)

    if post_process:
        # post processing: 'endseq'
        sents_pp = []
        for sent in sents:
            if '<end>' in sent.split():
                words = sent.split()
                sents_pp.append(' '.join(words[:words.index('<end>')]))
            else:
                sents_pp.append(sent)
        sents = sents_pp

    return sents


def my_eval(model_dir, out_path):
    """Main evaluation function

    Loads the validation(test) data set and runs the decoder model 

    Parameters
    ==========
        model_dir  -  path location of the model weights file to load
    Return
    ======
        None
    Output
    ======
        print the candidate generated image captions
    """
    nsd_loader = NSDAccess("/home/seagie/NSD")
    nsd_loader.stim_descriptions = pd.read_csv(nsd_loader.stimuli_description_file, index_col=0)

    input_size = params_dir['input']
    max_len = params_dir['max_length']
    units = params_dir['units']
    vocab_size = params_dir['top_k']
    max_len = params_dir['max_length']

#    data_train, train_vector, data_val, val_vector, tokenizer = load_data_pca(_max_length=20)
    data_train, train_vector, data_val, val_vector, tokenizer, _, _, train_keys, val_keys = load_data_img2(_max_length=20)

    val_generator = data_generator(data_val, val_vector, _unit_size = units, _vocab_size = vocab_size, _batch_size = 10, training=False)
    #val_generator =  data_generator(data_train, train_vector, _unit_size= units, _vocab_size = vocab_size, _batch_size = 10, training=False)
    

    NIC_inference = greedy_inference_model(vocab_size, input_size, max_len)
    NIC_inference.load_weights(model_dir, by_name = True, skip_mismatch=True)

    [test_features, words, a0, c0], target = val_generator.__next__()

    """
    candidates = []
    for i in val_generator:
        [test_features, words, a0, c0], target = i
        print("target shape:", target.shape) # (10, 20, 5000)
        out_candidates = my_decoder(NIC_inference, tokenizer, test_features, input_size, units, True)
        
        most_likely = np.argmax(target, axis = -1) # out: (N, max_length)
        print("most_likely:", most_likely.shape)
        print(tokenizer.sequences_to_texts(most_likely))

        print(out_candidates)
        break

    sys.exit(0)
    """
    print("val_vector:", val_vector.shape)

    """
    vec = val_vector[0]
    word = []
    for w in vec:
        word.append( tokenizer.index_word[w] ) 
    print(word)
    img = nsd_loader.read_images(val_keys[100])
    fig = plt.figure()
    plt.imshow(img)
    plt.title(f"{word}")
    plt.savefig(f"./img_{val_keys[100]}.png")
    plt.close(fig)
    """

        

    print("test_features:", test_features.shape)
    print("words:", words.shape)

    test_candidates = my_decoder(NIC_inference, tokenizer, test_features, input_size, units, True)

    target_sentences = np.argmax(target, axis = -1)
    target_sentences = tokenizer.sequences_to_texts(target_sentences)

    ## Save images with generated captions
    for k,v in enumerate(test_candidates):
        #print(v, " - ", target_sentences[k], " - ", val_keys[k])
        print("Candidate:", v)
        print("Target:", target_sentences[k])
        print(val_keys[k], "\n")

        img = nsd_loader.read_images(val_keys[k])
        fig = plt.figure()
        plt.imshow(img)
        plt.title(f"{v}")
        plt.savefig(f"{out_path}/test_img_{val_keys[k]}.png")
        plt.close(fig)


    print("Done.")
    
    return
    


if __name__ == '__main__':

    model_dir = './model-params-his/current_best.h5'
    model_dir = './model-params/current_best.h5'
    model_dir = './data/model-ep059-loss1.7367-val_loss1.6887.h5'
    model_dir = './data/img/model-ep024-loss1.7022-val_loss1.5508.h5'
    model_dir = './data/img_2/model-ep008-loss1.7451-val_loss1.6743.h5'
    model_dir = './data/img_3/model-ep019-loss1.6904-val_loss1.6130.h5'
    model_dir = './data/img/model-ep068-loss1.6716-val_loss1.4749.h5' # 200 epochs trained on images
    model_dir = './data/bs256/model-ep170-loss1.6274-val_loss1.5146.h5'
    model_dir = './data/new_loader/model-ep002-loss1.8410-val_loss1.7445.h5' # new load method

    #img_ids, test_references, candidates = evaluate_one(model_dir, method='b', beam_width = 5, alpha = 0.6)

    out_path = params_dir['data_dir'] + '/eval_out/'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        print("> created data folder:", out_path)

    my_eval(model_dir, out_path)




