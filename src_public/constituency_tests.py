from trees import PUNCT

### Clefting tests

def produce_clefting(sent, i, j):
    """
    Moves sent[i:j] via clefting, returns resulting sentences.
    sent -> It is/was sent[i:j] that sent[:i]+sent[j:].
    """
    sent = sent.copy()
    sent_1 = ['it', 'is'] + sent[i:j] + ['that'] + sent[:i] + sent[j:]
    sent_2 = ['it', 'was'] + sent[i:j] + ['that'] + sent[:i] + sent[j:]
    return [sent_1, sent_2], ['is_cleft', 'was_cleft']
    
### Movement tests

def produce_movement_front(sent, i, j):
    """
    Move sent[i:j] to the front, with a comma.
    """
    if i != 0 and i != j:
        span = sent[i:j]
        while sent[-1] in PUNCT:
            sent = sent[:-1]
        while span[0] in PUNCT:
            span = span[1:]
        return [span + [','] + sent[:i] + sent[j:]], ['move_front']
    return [], []

def produce_movement_end(sent, i, j):
    """
    Move sent[i:j] to the end without a comma.
    """
    if j != len(sent) and i != j:
        span = sent[i:j]
        while sent[0] in PUNCT:
            sent = sent[1:]
        while span[-1] in PUNCT:
            span = span[:-1]
        return [sent[:i] + sent[j:] + sent[i:j]], ['move_end']
    return [], []
    
### Substitution tests
    
def produce_subs(sent, i, j):
    """
    Replaces sent[i:j] with subs, returns resulting sentences.
    """
    subs = ['ones', 'did so', 'it']
    sents = []
    labels = []
    for sub in subs:
        sents.append(sent[:i] + [sub] + sent[j:])
        labels.append(sub.replace(' ', '_'))
    return sents, labels

def produce_omission(sent, i, j):
    """
    Tests for PPs, appositives, conjunction+XP, ... 
    """
    return [sent[:i] + sent[j:]], ['omission']

### Coordination

def produce_coordination_repeat(sent, i, j):
    sent_1 = sent[:i] + sent[i:j] + ['and'] + sent[i:j] + sent[j:]
    return [sent_1], ['coord']
