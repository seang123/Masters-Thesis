'''
File to define inference and BLEU evaluation method of NIC, 
including how to generate captions by given image use greedy or beam search, 

based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

'''

import math
import os
import click

import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from NIC import *
from preprocessing.image import *
from preprocessing.text import *

#from data_loader import load_data_pca, load_data_img, data_generator
from DataLoaders import data_loader
from DataLoaders import load_betas
import my_utils as uu
import sys, os
from nsd_access import NSDAccess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from parameters import params_dir
import my_utils as uu

# set np seed
np.random.seed(42)


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

    assert(features.shape[0] > 0), "batch size needs to be greater than 0"
    assert(features.shape[1] == input_size), f"feature size {features.shape[1]} doesn't match expected feature size {input_size}"

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

    return sents, y_preds

def plot_loss(data_dir, out_path):
    """ Create a loss plot from the training_log.csv file """

    df = pd.read_csv(data_dir + '/training_log.csv')
    df2 = pd.read_csv(data_dir + '/batch_training_log.csv')

    # Epoch loss
    fig, axs = plt.subplots(2, 1, sharex=True) 

    axs[0].plot(df.loss[1:], label = 'train')
    axs[0].plot(df.val_loss[1:], label = 'val')
    axs[0].set_title('Cross-entropy Loss')
    axs[0].legend()

    axs[1].axhline(0.50, color = 'k', linestyle = '--')
    axs[1].axhline(1.00, color = 'k', linestyle = ':')
    axs[1].plot(df.accuracy, label = 'train')
    axs[1].plot(df.val_accuracy, label = 'val')
    axs[1].set_title('Categorical Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.savefig(f"{out_path}/training_loss.png")
    plt.close(fig)

    # Batch loss
    fig, axs = plt.subplots(2, 1, sharex=True) 

    axs[0].plot(df2.loss[600:], label = 'train')
    axs[0].set_title('Categorical Cross-entropy Loss')
    axs[0].legend()

    axs[1].axhline(0.50, color = 'k', linestyle = '--')
    axs[1].plot(df2.accuracy, label = 'tain')
    axs[1].set_title('Categorical Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.savefig(f"{out_path}/batch_training_loss.png")
    plt.close(fig)

    return

def categorical_accuracy(y_pred, y_true):
    """ Compute categorical accuracy between two matricies

    Parameters
    ----------
        y_pred : ndarray
            predictions - one-hot encoded
        y_true : ndarray
            true labels - one-hot encoded 

    Returns
    -------
        z : float
            categorical accuracy
    """

    assert y_pred.ndim == 3 and y_true.ndim == 3, "Arrays need to be 3 dimensional (batch, n_words, vocab_size)"

    y_pred_am = np.argmax(y_pred, axis=2)
    y_true_am = np.argmax(y_true, axis=2)

    z = y_pred_am == y_true_am

    z = np.sum(z, axis=1) / y_pred_am.shape[1]
    return z


def my_bleu(candidate: str, img_key: int):
    """ Calculates the BLEU score for a given candidate sentence and all 5 of its respective possible target captions

    Parameters
    ----------
        candidate : str
            the model generated caption
        img_key : int
            the corresponding NSD key of this trial

    Returns
    -------
        bleu : float
            the BLEU score

    """

    candidate = candidate.split(" ")

    annt_dict = uu.load_json('../../modified_annotations_dictionary.json')

    targets = annt_dict[str(img_key)] # list of strings

    references = []
    for i in targets:
        words = i.split(" ")
        words = words[1:len(candidate)] + ['<pad>']
        words = [w.lower() for w in words]
        references.append(words)

    chencherry = SmoothingFunction()
    bleu = sentence_bleu(references, candidate, smoothing_function=chencherry.method4)

    return bleu


def my_eval(model_dir, out_path, BATCH_SIZE_EVAL = 15, GEN_IMG=True):
    """Main evaluation function

    Loads the validation(test) data set and runs the decoder model 

    Parameters
    ==========
        model_dir : str
            path location of the model weights file to load
        out_path : str
            where to store outputs
        BATCH_SIZE_EVAL : int
            number of samples to evaluate
        GEN_IMG : bool
            whether to output images with the candidate captions
            
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


    # Data Loader
    #data_train, train_vector, data_val, val_vector, tokenizer, ext_train_keys, ext_val_keys = data_loader.load_data_img(_max_length = max_len, train_test_split = 0.9)
    # _, _, data_val, val_vector, tokenizer, _, ext_val_keys = load_betas.load_data_betas_partial(load_train=False, load_val=True, shuffle_data=True, _max_length = max_len)

    data_train, train_vector, data_val, val_vector, tokenizer, ext_train_keys, ext_val_keys = load_betas.load_data_betas(_max_length = max_len)

    train_keys_set = list(set(ext_train_keys)) # unq keys
    train_keys_set_split = int(len(train_keys_set) * 0.9) # split unq keys
    train_keys_set_1 = train_keys_set[:train_keys_set_split] # 8100
    train_keys_set_2 = train_keys_set[train_keys_set_split:] # 900

    print("train_keys_set_1", len(train_keys_set_1))
    print("train_keys_set_2", len(train_keys_set_2))


    #train_x = np.array([np.where(train_keys == i)[0] for i in train_keys_set_1]) # (8100,)
    #train_y = np.array([np.where(train_keys == i)[0] for i in train_keys_set_2]) # (900,)

    train_x = []
    train_y = []
    for k, v in enumerate(ext_train_keys):
        if v in train_keys_set_1:
            train_x.append(k)
        elif v in train_keys_set_2:
            train_y.append(k)
        else:
            raise Exception("oops")


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print("train_x", train_x.shape)
    print("train_y", train_y.shape)

    data_val = data_train[train_y]
    val_vector = train_vector[train_y]
    ext_val_keys = train_y

    data_train = data_train[train_x]
    train_vector = train_vector[train_x]
    ext_train_keys = train_x


    #data_train     = data_train[:1000,:]
    #train_vector   = train_vector[:1000,:]
    #ext_train_keys = ext_train_keys[:1000]

    print("data_train", data_train.shape)
    print("train_vector", train_vector.shape)
    print("ext_trian_keys", ext_train_keys.shape)

    #with open( 'eval_keys.txt', 'w') as f:
    #    for i in range(0, 1000):
    #            f.write(f"{ext_train_keys[i]}\n")

    #sentences = tokenizer.sequences_to_texts( train_vector )
    #print("sentences", len(sentences))
    #ts = []
    #for k, v in enumerate(sentences):
    #    ts.append(v)
    
    #for i in range(10):
    #    print(ts[i])

    #c = 0
    #with open( 'eval_captions.txt', 'w' ) as f:
    #    for item in ts:
    #        c += 1
    #        f.write( "%s\n" % item )
    #print(c)
    #sys.exit(0)

    # Data Batch Generator
#    val_generator = data_loader.data_generator(data_train, train_vector, ext_train_keys, _unit_size = units, _vocab_size = vocab_size, _batch_size = BATCH_SIZE_EVAL, training=False)
    val_generator = data_loader.data_generator(data_val, val_vector, ext_val_keys, _unit_size = units, _vocab_size = vocab_size, _batch_size = BATCH_SIZE_EVAL, training=False)

    ## Model
    NIC_inference = greedy_inference_model(vocab_size, input_size, max_len, params_dir['L2_reg'] )
    NIC_inference.load_weights(model_dir, by_name = True, skip_mismatch=True)

    [test_features, words, a0, c0], target, img_keys = val_generator.__next__()

    print("img keys:", img_keys) 
    print("test_features:", test_features.shape)
    print("words:", words.shape)

    ## Decoder
    test_candidates, y_preds = my_decoder(NIC_inference, tokenizer, test_features, input_size, units, True)

    target_sentences = np.argmax(target, axis = -1)
    target_sentences = tokenizer.sequences_to_texts(target_sentences)

    ## categorical accuracy
    cat_acc = categorical_accuracy(y_preds, target)

    # ts = []
    # for k, v in enumerate(test_candidates):
    #     target_cap = target_sentences[k]
    #     ts.append(target_cap)
    #
    # with open( 'target_caps.txt', 'w' ) as f:
    #     for item in ts:
    #         f.write( "%s\n" % item )
    # sys.exit(0)

    avg_bleu = 0
    ## Save images with generated captions
    for k,v in enumerate(test_candidates):
        bleu = my_bleu(v, img_keys[k])
        avg_bleu += bleu
        print("Candidate:", v)
        target_cap = target_sentences[k].partition("<end>")
        target_cap = target_cap[0] + target_cap[1]
        print("Target   :", target_cap)
        print("Accuracy: ", cat_acc[k])
        print("BLEU:     ", bleu)
        print(img_keys[k], "\n")

        if GEN_IMG:
            img = nsd_loader.read_images(img_keys[k])
            fig = plt.figure()
            plt.imshow(img)
            plt.title(f"{v}")
            plt.savefig(f"{out_path}/test_img_{img_keys[k]}.png")
            plt.close(fig)

    print(f"mean BLEU score: {(avg_bleu / len(test_candidates)):.4f}")
    print(f"mean categorical-accuracy: {np.mean(cat_acc):.4f}")
    print("Done.")
    
    return
    


if __name__ == '__main__':

    out_path = params_dir['data_dir'] + '/eval_out/'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        print("> created data folder:", out_path)

    plot_loss(params_dir['data_dir'], out_path)

    sys.exit(0)

    model_dir = params_dir['data_dir'] + '/latest-model.h5'
#    model_dir = './data/pca_short_lowLR/model-ep046-loss3.0974-val_loss3.0367.h5'

    # how many images to evaluate
    BATCH_SIZE_EVAL = 5
    GEN_IMG = True


    my_eval(model_dir, out_path, BATCH_SIZE_EVAL, GEN_IMG)



