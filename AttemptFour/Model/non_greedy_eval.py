"""

    Implements a non-greedy evaluation
    Given a single beta sample, generate a caption
using non-greedy word selection. (top_k or top_p)

    Top_k just selects the top k words at any time step
    Top_p selects the top k words such that sum(k_probs) >= p

"""

class Word:

    def __init__(self, word, probability):
        self.word = word
        self.probability = probability
        self.nodes = []

    def add_node(self, word, probability):
        self.nodes.append( Word(word, probability) )

    def __repr__(self):
        print(f"Word: {self.word}  ({self.probability})")


def non_greedy_eval(data, start_word,  model, tokenizer, max_len, units):
    raise NotImplementedError
