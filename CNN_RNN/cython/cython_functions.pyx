## To compile the .pyx file
# python setup.py build_ext --inplace

## To use the cython file in the parent folders, we need to create a __init__.py, however to compile the cython code we need to delete this file
# and then remake it

## Holds cython functions

cimport numpy as np
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import cython

if cython.compiled:
    print("Yep, I'm compiled.")
else:
    print("Just a lowly interpreted script.")

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double bleu_score_c(np.ndarray[DTYPE_t, ndim=2] predictions, np.ndarray[DTYPE_t, ndim=2] target, tokenizer):
   

    assert predictions.dtype == DTYPE and target.dtype == DTYPE

    cdef int i
    cdef int pred_shape_1 = predictions.shape[1]
    cdef int score = 0
    cdef int end_token = tokenizer.word_index['<end>']

   
    for i in range(0, pred_shape_1):
        sentence_idx = predictions[:,i]

        #end_idx = np.where(sentence_idx == end_token, sentence_idx, -1)
        #end_idx = 
        end_idx = np.where(sentence_idx == end_token)[0]
        if len(end_idx) != 0:
            sentence_idx[end_idx[0]+1:] = 0 

        sentence_words = [tokenizer.index_word[j] for j in sentence_idx if j != 0]

        reference_words = [tokenizer.index_word[j] for j in target[i,:] if j != 0]

        score += sentence_bleu(reference_words, sentence_words, smoothing_function=SmoothingFunction().method2)

    return (score/pred_shape_1), score

