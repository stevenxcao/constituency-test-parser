import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef

import os, time, argparse

from grammar_model import BinaryClassifier
import constituency_tests as ct
from trees import detokenize, PUNCT
from util import from_numpy, torch_load
from bigram_models import BigramModel, generate_text

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def load_cola(filename):
    examples = []
    with open(filename) as file:
        for line in file:
            line = line.split(sep='\t')
            sent = line[-1].replace("\n", "")
            if sent.endswith('.'):
                sent = sent[:-1]
            sent = sent.strip('"').lower()
            examples.append((sent, int(line[1])))
    return examples

def load_gigaword(directory, min_len = 2, max_len = 60, num_sent = 5e6):
    # removes ending periods and quotes
    sents = []
    for fname in os.listdir(directory):
        if 'README' not in fname:
            with open(os.path.join(directory, fname)) as file:
                for line in file:
                    line = line.strip('\n').replace('"','')
                    if line not in ('<doc>', '<doc/>'):
                        words = line.split()
                        if len(words) in range(min_len, max_len + 1):
                            # if words[-1] == '.':
                            #     words = words[:-1]
                            sents.append(words)
                    if len(sents) >= num_sent:
                        return sents
    return sents

def drop_words(sent):
    num_words = np.random.choice(len(sent) - 1)
    if num_words < 5:
        num_words = 5
    dropped = list(np.array(sent)[np.sort(np.random.choice(len(sent), size=num_words, replace=False))])
    return [dropped]

def shuffle_line(sent):
    num_words = np.random.choice(len(sent) - 1) + 2 # [2, n]
    idx = list(range(len(sent)))
    np.random.shuffle(idx)
    idx = idx[:num_words]
    mapping = dict((idx[i], idx[i - 1]) for i in range(num_words))
    shuffled = [sent[mapping.get(x, x)] for x in range(len(sent))]
    return [shuffled]

def random_span_test(sent, test_f):
    i = np.random.choice(len(sent) - 2) # [0, n-2)
    j = np.random.choice(len(sent) - (i+2)) + i + 2 # [i+2, n)
    if j - i > len(sent) - 5 and test_f == ct.produce_omission: # sent must end up at least length 5
        j = i + (len(sent) - 5)
    if len([s for s in sent[i:j] if s not in PUNCT]) <= 1 or j-1 == len(sent):
        return []
    tests = test_f(sent, i, j)[0]
    return tests

def swap(sent):
    i = np.random.choice(len(sent) - 1)
    j = np.random.choice(len(sent) - i - 1) + i + 1
    sent = sent.copy()
    sent[i], sent[j] = sent[j], sent[i]
    return [sent]

TESTS = [ct.produce_movement_end, ct.produce_movement_front, ct.produce_omission]

NEG = [drop_words, shuffle_line, swap] # bigram model is added in training code

def produce_negative_exmps(sentences):
    neg = []
    for sent in sentences:
        perms = []
        for func in NEG:
            perms.extend(func(sent))
        for test_f in TESTS:
            perms.extend(random_span_test(sent, test_f))
        neg.append(perms[np.random.choice(len(perms))])
    return neg
    
def train(train_cola_path, train_giga_path, dev_cola_path, model_path,
          batch_size = 32, lr_warmup_frac = 0.1, lr_base = 0.00003, epochs = 1, num_sent = 5e6,
          subbatch_size = 32, bert_model = 'roberta_base', reinit = False, resume_path = None):
    print()
    print("Working directory:", os.path.dirname(os.path.abspath(__file__)))
    train_cola = None
    if train_cola_path:
        print()
        print("Loading train cola from {}".format(train_cola_path))
        train_cola = load_cola(train_cola_path)
        print(train_cola[0])
    train_giga = None
    if train_giga_path:
        print()
        print("Loading train gigaword from {}".format(train_giga_path))
        if not resume_path:
            train_giga = load_gigaword(train_giga_path, min_len = 8, max_len = 40,
                                       num_sent = num_sent)
        else:
            train_giga = load_gigaword(train_giga_path, min_len = 8, max_len = 40,
                                   num_sent = num_sent*2)
            train_giga = train_giga[int(num_sent):]
        print(train_giga[0])
    print()
    print("Loading dev cola from {}".format(dev_cola_path))
    dev_cola = load_cola(dev_cola_path)
    print(dev_cola[0])
    
    model = BinaryClassifier(bert_model, reinit = reinit)
    if resume_path is not None:
        model.load_state_dict(torch_load(resume_path)['state_dict'])
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    trainer = torch.optim.Adam(trainable_parameters, lr=0, betas=(0.9, 0.999), eps=1e-6)
    def set_lr(new_lr): # hparams from roberta paper
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr
    
    print()
    print("Training...")
    start_time = time.time()
    processed = 0
    check_processed = 0
    num_checks = 25
    num_exmp = 0
    if train_cola:
        num_exmp = len(train_cola)
    if train_giga:
        num_exmp = max(num_exmp, len(train_giga))
        bigram_model = BigramModel(train_giga[:10000])
        NEG.append(lambda x: [generate_text(bigram_model, n=len(x))])
        
    for epoch in range(epochs):
        if train_cola:
            np.random.shuffle(train_cola)
        if train_giga:
            np.random.shuffle(train_giga)
        for i in range(0, num_exmp, batch_size):
            model.train()
            trainer.zero_grad()
            
            if processed % 10000 < batch_size and train_giga:
                bigram_model.retrain(train_giga[i:i+10000]) # use bigram model on chunks of 10k sentences
            
            pos, neg = [], []
            if train_cola:
                j = i % len(train_cola)
                cola_batch = train_cola[j:j+batch_size]
                for sent, label in cola_batch:
                    if label == 0:
                        neg.append(sent)
                    else:
                        pos.append(sent)
            if train_giga:
                j = i % len(train_giga)
                giga_batch = train_giga[j:j+batch_size]
                pos.extend([detokenize(s) for s in giga_batch])
                neg.extend([detokenize(s) for s in produce_negative_exmps(giga_batch)])
            
            if i == 0:
                print()
                print("Pos/neg examples:")
                for sent1, sent2 in zip(pos, neg):
                    print()
                    print(sent1)
                    print(sent2)
            
            processed += batch_size
            lr_ratio = min(1, processed / (lr_warmup_frac * num_exmp))
            set_lr(lr_ratio * lr_base)
            
            loss = F.cross_entropy(model(pos), from_numpy(np.array([1] * len(pos))))
            loss.backward()
            loss = F.cross_entropy(model(neg), from_numpy(np.array([0] * len(neg))))
            loss.backward()
            del loss
            
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, np.inf)
            trainer.step()
            
            check_processed += batch_size
            if i == 0 or check_processed > (num_exmp // num_checks):
                print()
                print("Checking dev...")
                check_processed -= (num_exmp // num_checks)
                
                model.eval()
                with torch.no_grad():
                    pred = model([exmp[0] for exmp in dev_cola])
                pred = np.argmax(pred.cpu().numpy(), axis = 1)
                mcc = matthews_corrcoef([exmp[1] for exmp in dev_cola], pred)
                
                print(
                    "epoch {:,} "
                    "sentences {:,}-{:,}/{:,} "
                    "grad-norm {:.4f} "
                    "dev-mcc {:.4f} "
                    "total-elapsed {}".format(
                        epoch,
                        i, i + batch_size, num_exmp,
                        grad_norm,
                        mcc,
                        format_elapsed(start_time)))
                
                print("Saving model to {}...".format(model_path))
                torch.save({
                    'bert_model': 'roberta-base',
                    'state_dict': model.state_dict()
                    }, model_path)
                print("Done saving.")

def run_train(args):
    train(args.train_cola_path, args.train_giga_path, args.dev_cola_path, args.model_path,
          batch_size = args.batch_size, epochs = args.epochs, num_sent = args.num_sent, 
          bert_model = args.bert_model, reinit = args.reinit, resume_path = args.resume_path)
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args))
    subparser.add_argument("--train-cola-path", default=None)
    subparser.add_argument("--train-giga-path", default=None)
    subparser.add_argument("--dev-cola-path", required=True)
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--batch-size", type=int, default=32)
    subparser.add_argument("--epochs", type=int, default=1)
    subparser.add_argument("--num-sent", type=int, default=5e6)
    subparser.add_argument("--bert-model", default='roberta-base')
    subparser.add_argument("--reinit", type=int, default=0)
    subparser.add_argument("--resume-path", default=None)
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    args.callback(args)
    
if __name__ == '__main__':
    main()
