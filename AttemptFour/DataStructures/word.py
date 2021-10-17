class Word:

    def __init__(self, word, probability):
        self.word = word
        self.probability = probability
        self.nodes = []

    def add_node(self, word, probability):
        self.nodes.append( Word(word, probability) )

    def __repr__(self):
        print(f"Word: {self.word}  ({self.probability})")
