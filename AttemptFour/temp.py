from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
import numpy as np

x = [(1, 'apple'), (1, 'banana'), (2, 'orange'), (3, 'grape')]
print("x:", x)

def remove_dup(x):
    keys = []
    new_list = []
    for i, v in enumerate(x):
        key = v[0]
        if key in keys:
            continue
        else:
            new_list.append(v)
            keys.append(key)
    return new_list


y = remove_dup(x)
print("y:", y)

z = list({v[0]:v for v in x}.values())
print("z:", z)

### ----

x = ["hi how are you", "this is an apple tree", "donde esta"]
y = [1, 2, 3]

z = []
z.append( list(zip(x, y)) )
print(z)

### ----


x = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
y = ['the', 'quick', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad', 'pad']

print(f"{x} : {len(x[0])}\n{y} : {len(y)}")

bleu = sentence_bleu(x, y, weights = (1, 0, 0, 0))
print("bleu:", bleu)


