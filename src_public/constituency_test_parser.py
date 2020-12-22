import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trees import PUNCT, detokenize, Tree

class ConstituencyTestParser(nn.Module):
    def __init__(self, grammar_model, tests, num_grad = 32):
        super().__init__()
        self.grammar_model = grammar_model
        self.test_functions = tests
        self.num_grad = num_grad

    def produce_tests(self, sent, i, j):
        """
        Returns a list of tests for the span, along with the name of each test.
        Sent should be a list of tokens. Each outputted test is also a list of
        tokens.
        """
        if len([s for s in sent[i:j] if s not in PUNCT]) <= 1 or j-i == len(sent):
            return [], []
        else:
            tests, labels = [], []
            for test_f in self.test_functions: # order of the tests should stay consistent.
                t, l = test_f(sent, i, j)
                tests.extend(t)
                labels.extend(l)
            return tests, labels
    
    def calc_posterior(self, sent, pgram):
        return np.sum(pgram) / len(pgram) # pgram has tensors in it, so we cannot use np.mean

    def is_constituent(self, sent_s, i_s, j_s):
        """
        For sent, i, j in zip(sent_s, i_s, j_s), tests whether sent[i:j] is a
        constituent by checking the grammaticality of constituency tests.
        
        prob_s - probability of each span
        label_s - string describing the result (for humans to read)
        """
        # produce list of tests, while keeping track of which input each test came from
        all_tests, all_labels = [], []
        idx_s = []
        for ind, (sent, i, j) in enumerate(zip(sent_s, i_s, j_s)):
            tests, labels = self.produce_tests(sent, i, j)
            idx_s.append((len(all_tests), len(all_tests) + len(tests)))
            all_tests.extend([detokenize(test) for test in tests])
            all_labels.extend(labels)
        
        # feed tests to grammar model
        # for num_grad random tests, use grad.
        shuffle_idx = np.arange(len(all_tests))
        np.random.shuffle(shuffle_idx)
        all_tests_shuffled = np.array(all_tests)[shuffle_idx] # shuffle tests to pick random ones to grad
        all_pgram = []
        if len(all_tests_shuffled) - self.num_grad < self.num_grad: # take grad through everything
            all_pgram = all_pgram + list(F.softmax(self.grammar_model(all_tests_shuffled), dim = -1)[:,1])
        else:
            if self.num_grad != 0: # take grad through num_grad
                all_pgram = all_pgram + list(F.softmax(self.grammar_model(all_tests_shuffled[:self.num_grad]), dim = -1)[:,1])
            if self.num_grad < len(all_tests):
                with torch.no_grad(): # compute the rest without grad
                    all_pgram = all_pgram + list(F.softmax(self.grammar_model(all_tests_shuffled[self.num_grad:]), dim = -1)[:,1].cpu().numpy())
        all_pgram = np.array(all_pgram)
        all_pgram[shuffle_idx] = all_pgram.copy() # unshuffle

        prob_s, label_s = [], [] # prob and label of each span (indexing is aligned with sent_s, i_s, j_s, idx_s)
        for sent, i, j, idx in zip(sent_s, i_s, j_s, idx_s):
            k, l = idx
            if l - k == 0: # no tests for this span
                prob_s.append(1)
                label_s.append('X')
            else:
                pgram = all_pgram[k:l]
                tests = all_tests[k:l]
                labels = all_labels[k:l]

                prob = self.calc_posterior(sent, pgram)
                best_ind = np.argmax(pgram)

                prob_s.append(prob)
                label_s.append('[{:.2f}]{}={:.2f}'.format(prob, labels[best_ind], pgram[best_ind]))

        return prob_s, label_s
    
    def forward(self, sent_all):
        """
        Parse all the sentences (algorithm is only run on sentences with length >= 3)
        """
        sent_len3 = []
        idx = []
        trees, probs = [None] * len(sent_all), [None] * len(sent_all)
        for i, sent in enumerate(sent_all):
            sent_stripped = [s for s in sent if s not in PUNCT]
            if len(sent_stripped) == 1:
                trees[i] = Tree(sent_stripped[0], None, sent_stripped[0])
            elif len(sent_stripped) == 2:
                trees[i] = Tree('S', [Tree(sent_stripped[0], None, sent_stripped[0]),
                     Tree(sent_stripped[1], None, sent_stripped[1])], None)
            else:
                sent_len3.append(sent)
                idx.append(i)
        
        if len(sent_len3) > 0:
            tree_all, prob_all = self.cky_len3(sent_len3)
            for tree, prob, ind in zip(tree_all, prob_all, idx):
                trees[ind] = tree
                probs[ind] = prob
        
        return trees, probs
    
    def cky_len3(self, sent_all):
        """      
        choose tree with maximum expected number of constituents,
        or max \sum_{(i,j) \in tree} p((i,j) is constituent)
        
        sent_all - list of sentences, which must be at least length 3.
            Each sentence is a list of tokens.
        """
        def backpt_to_tree(sent, backpt, label_table):
            """
            backpt[i][j] - the best splitpoint for the span sent[i:j]
            label_table[i][j] - description for span sent[i:j] (for humans to read - the parser is unlabeled)
            """
            def to_tree(i, j):
                if j - i == 1:
                    return Tree(sent[i], None, sent[i])
                else:
                    k = backpt[i][j]
                    return Tree(label_table[i][j], [to_tree(i,k), to_tree(k,j)], None)
            return to_tree(0, len(sent))

        def to_table(value_s, i_s, j_s):
            """
            Creates table where value_s[k] is put in entry (i_s[k], j_s[k])
            """
            table = [[None for _ in range(np.max(j_s) + 1)] for _ in range(np.max(i_s) + 1)]
            for value, i, j in zip(value_s, i_s, j_s):
                table[i][j] = value
            return table

        # produce list of spans to pass to is_constituent, while keeping track of which sentence
        sent_s, i_s, j_s  = [], [], []
        idx_all = []
        for ind, sent in enumerate(sent_all):
            start = len(sent_s)
            for i in range(len(sent)):
                for j in range(i+1, len(sent) + 1):
                    sent_s.append(sent)
                    i_s.append(i)
                    j_s.append(j)
            idx_all.append((start, len(sent_s)))

        # feed spans to is_constituent
        prob_s, label_s = self.is_constituent(sent_s, i_s, j_s)

        # given span probs, perform CKY to get best tree for each sentence.
        tree_all, prob_all = [], []
        for sent, idx in zip(sent_all, idx_all):
            # first, use tables to keep track of things
            k, l = idx
            prob, label = prob_s[k:l], label_s[k:l]
            i, j = i_s[k:l], j_s[k:l]

            prob_table = to_table(prob, i, j)
            label_table = to_table(label, i, j)

            # perform cky using scores and backpointers
            score_table = [[None for _ in range(len(sent) + 1)] for _ in range(len(sent))]
            backpt_table = [[None for _ in range(len(sent) + 1)] for _ in range(len(sent))]
            for i in range(len(sent)): # base case: single words
                score_table[i][i+1] = 1
            for j in range(2, len(sent) + 1): 
                for i in range(j-2, -1, -1):
                    best, argmax = -np.inf, None
                    for k in range(i+1, j): # find splitpoint
                        score = score_table[i][k] + score_table[k][j]
                        if score > best:
                            best, argmax = score, k
                    score_table[i][j] = best + prob_table[i][j]
                    backpt_table[i][j] = argmax

            tree = backpt_to_tree(sent, backpt_table, label_table)
            tree_all.append(tree)
            prob_all.append(prob_table)

        return tree_all, prob_all