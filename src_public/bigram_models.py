import numpy as np

class Vocab: # 0 is reserved for padding
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        for word in corpus:
            w2i.setdefault(word, len(w2i))
        return Vocab(w2i)

    def __len__(self):
        return len(self.w2i.keys())
    
    def insert(self, word):
        if word not in self.w2i:
            self.w2i[word] = len(self.w2i)
            self.i2w[len(self.i2w)] = word

def build_vocab(text):
    corpus = set()
    for exmp in text:
        for tok in exmp:
            corpus.add(tok)
    return Vocab.from_corpus(corpus)

class BigramModel:
    def __init__(self, train_text):
        self.retrain(train_text)
    
    def retrain(self, train_text):
        self.vocab = build_vocab(train_text)
        
        self.counts = np.zeros((len(self.vocab), len(self.vocab)))
        for exmp in train_text:
            for i in range(len(exmp)):
                if i == 0:
                    self.counts[self.vocab.w2i['.'], 
                                self.vocab.w2i[exmp[i]]] += 1
                else:
                    self.counts[self.vocab.w2i[exmp[i-1]], 
                                self.vocab.w2i[exmp[i]]] += 1
    
    def next_word_probabilities(self, text_prefix):
        """Return a list of probabilities for each word in the vocabulary."""
        counts = self.counts[self.vocab.w2i[text_prefix[-1]],:]
        return counts / np.sum(counts)

def generate_text(model, n=50, prefix=('.',)):
    prefix = list(prefix)
    for _ in range(n):
        probs = np.array(model.next_word_probabilities(prefix))
        probs[model.vocab.w2i['.']] = 0 # don't end the sentence
        if np.isnan(np.sum(probs)) or np.sum(probs) == 0:
            probs = np.array(model.next_word_probabilities(('.',)))
            probs[model.vocab.w2i['.']] = 0
        index = np.random.choice(len(model.vocab), p=probs/np.sum(probs))
        prefix.append(model.vocab.i2w[index])
    return prefix[1:]   
