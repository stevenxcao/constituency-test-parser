DELETE = ('``', "''", '-NONE-', '.', '-none-')
PUNCT = (',', ';', ':', '--', '-', '.', '?', '!', '-LRB-', '-RRB-', '$', '#',
         '-LCB-', 'RCB-', "'", '-lrb-','-rrb-','-lcb-','-rcb-')

class Tree(object):
    def __init__(self, label, children, word):
        self.label = label
        self.children = children
        self.word = word
        
    def __str__(self):
        return self.linearize()
    
    def linearize(self):
        if not self.children:
            return "({} {})".format(self.label, self.word)
        return "({} {})".format(self.label, " ".join(c.linearize() for c in self.children))

    def linearize_latex_labeled_fromparser(self):
        """
        Returns string for tikz-qtree, with nodes labeled by their score under the parser
        """
        if not self.children:
            return "{}".format(self.word)
        if self.label == 'X':
            prob = '1.00'
        else:
            prob = self.label[1:5]
        return "[ .{} {} ]".format(prob , " ".join(c.linearize_latex_labeled() for c in self.children))
    
    def linearize_latex_labeled_fromtreebank(self):
        """
        Returns string for tikz-qtree, use this method for trees from load_ptb
        """
        if not self.children:
            return "{}".format(self.word)
        return "[ .{} {} ]".format(self.label , " ".join(c.linearize_latex_labeled_2() for c in self.children))
    
    def linearize_latex_labeled_colored(self, gold_spans, start = 0):
        """
        Returns string for tikz-qtree. 
        Given this tree and a list of gold spans, color-codes the brackets:
            red - crossing, blue - correct, dashed blue - consistent with gold tree
        Note: the parser removes some punctuation. for the spans to align, both trees should have
        the same punctuation.
        """
        if not self.children:
            return "{}".format(self.word)
        if self.label == 'X':
            prob = '1.00'
        else:
            prob = self.label[1:5]
            
        position = start
        starts = []
        for c in self.children:
            starts.append(position)
            cspans = c.spans(start = position)
            position = cspans[0][1]
        span = (start, position)
        
        if span in gold_spans:
            edge = "\\edge[draw=blue];"
        elif crossing(span, gold_spans):
            edge = "\\edge[draw=red];"
        else:
            edge = "\\edge[dashed,draw=blue];"
        child_strings = [c.linearize_latex_labeled_colored(gold_spans, start) for c, start in zip(self.children, starts)]
        assert len(child_strings) == 2, 'Tree not binary'
        return "[ .{} {} {} {} {} ]".format(prob , edge, child_strings[0], edge, child_strings[1])
    
    def sent(self):
        if not self.children:
            return [self.word]
        return sum([c.sent() for c in self.children], [])
    
    def spans(self, start = 0):
        if not self.children:
            return [(start, start + 1)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans(start = position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position)] + span_list
        
    def spans_labels(self, start = 0):
        if not self.children:
            return [(start, start + 1, self.label)]
        span_list = []
        position = start
        for c in self.children:
            cspans = c.spans_labels(start = position)
            span_list.extend(cspans)
            position = cspans[0][1]
        return [(start, position, self.label)] + span_list
    
    def copy(self):
        return unlinearize(self.linearize())
 
def crossing(span, constraints):
    """
    False if span is consistent will all spans in constraints, True if at least
    one span is crossing.
    """
    i, j = span
    for (k,l) in constraints:
        if (i > k and i < l and j > l) or (i < k and j > k and j < l):
            return True
    return False
    
def unlinearize(string):
    """
    (TOP (S (NP (PRP He)) (VP (VBD was) (ADJP (JJ right))) (. .)))
    """
    tokens = string.replace("(", " ( ").replace(")", " ) ").split()
    
    def read_tree(start):
        if tokens[start + 2] != '(':
            return Tree(tokens[start+1], None, tokens[start+2]), start + 4
        i = start + 2
        children = []
        while tokens[i] != ')':
            tree, i = read_tree(i)
            children.append(tree)
        return Tree(tokens[start+1], children, None), i + 1
    
    tree, _ = read_tree(0)
    return tree

def collapse_unary_chains(tree):
    if tree.children:
        for i, child in enumerate(tree.children):
            while child.children and len(child.children) == 1:
                child = child.children[0]
            tree.children[i] = collapse_unary_chains(child)
    return tree

def standardize_punct(tree, nopunct): # tree cannot have unary chains
    if nopunct:
        delete_list = DELETE + PUNCT
    else:
        delete_list = DELETE
    if tree.children:
        tree.children = [c for c in tree.children if not (c.label in delete_list)]
        for c in tree.children:
            standardize_punct(c, nopunct)
    return tree
    
def remove_labels(tree):
    tree.label = 'X'
    if tree.children:
        for c in tree.children:
            remove_labels(c)
    return tree
        
def clean_empty(tree):
    """
    After removing punctuation, we might have non-leaves with 0 children.
    This function removes those nodes.
    """
    if tree.children:
        tree.children = [c for c in tree.children if c.word or len(c.children) > 0]
        for c in tree.children:
            clean_empty(c)
    return tree

def load_ptb(fname, nopunct = False, remove_len1 = False, lower = False):
    trees = []
    with open(fname) as file:
        for line in file:
            line = line.strip('\n')
            if lower:
                line = line.lower()
            tree = collapse_unary_chains(unlinearize(line))
            tree = standardize_punct(tree, nopunct)
            tree = collapse_unary_chains(clean_empty(tree))
            if not remove_len1 or len([t for t in tree.sent() if t not in (DELETE+PUNCT)]) > 1:
                trees.append({'sent': tree.sent(), 'tree': tree, 'string': line})
    return trees
    
def detokenize(words):
    string = " ".join(words)
    string = string.replace('-LRB-', '(').replace('-RRB-', ')')
    string = string.replace('-LSB-', '[').replace('-RSB-', ']')
    string = string.replace('-LCB-', '{').replace('-RCB-', '}')
    return string.lower()

def get_leaves(tree):
    if not tree.children:
        return [tree]
    leaves = []
    for c in tree.children:
        leaves.extend(get_leaves(c))
    return leaves

def transfer_leaves(tree1, tree2):
    leaves1, leaves2 = get_leaves(tree1), get_leaves(tree2)
    for l1, l2 in zip(leaves1, leaves2):
        assert l1.word.lower() == l2.word.lower(), "{} =/= {}".format(l1.word, l2.word)
        l1.label = l2.label

def produce_right_branching(sent):
    if len(sent) == 1:
        return Tree(sent[0], None, sent[0]) # label, children, word
    return Tree('A', [produce_right_branching(sent[0:1]), produce_right_branching(sent[1:])], None) 

def binarize(tree): # non-destructive
    if not tree.children:
        return Tree(tree.label, None, tree.word)
    else:
        t = Tree(tree.label, [binarize(c) for c in tree.children], tree.word)
        while len(t.children) > 2:
            first, second = t.children[-2], t.children[-1]
            t.children = t.children[:-2]
            t.children.append(Tree("{}+{}".format(first.label, second.label), [first, second], None))
        return t