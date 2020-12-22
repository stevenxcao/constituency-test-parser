import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, time, argparse
import re, subprocess, tempfile

from grammar_model import BinaryClassifier
import constituency_tests as ct
from constituency_test_parser import ConstituencyTestParser
from trees import load_ptb
from util import from_numpy, torch_load

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def compute_f1_evalb(gold_str, tree_str):
    print("WARNING: You are using EVALB to compute F1.")
    print("Numbers will not match those in the paper, which uses a different eval script (following past work).")
    evalb_path = '../EVALB'
    
    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "pred.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")
    
    with open(gold_path, "w") as outfile:
        outfile.write("\n".join(gold_str)) 
    with open(predicted_path, "w") as outfile:
        outfile.write("\n".join(tree_str)) 
    
    command = "{} -p {} {} {} > {}".format(
        evalb_path + '/evalb',
        'unlabeled.prm',
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)
    
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                return float(match.group(1))

def list_to_tensor_grad(lst):
    # lst has 0-dim tensors
    tensor = lst[0][None]
    for i in range(1,len(lst)):
        tensor = torch.cat((tensor, lst[i][None]))
    return tensor

def compute_estep_loss(trees, probs):
    probs_list = []
    labels_list = []
    for tree, prob in zip(trees, probs):
        if prob is not None:
            spans = tree.spans()
            for i in range(len(prob)):
                for j in range(len(prob[i])):
                    if isinstance(prob[i][j], torch.Tensor):
                        if (i,j) in spans:
                            labels_list.append(1)
                        else:
                            labels_list.append(0)
                        probs_list.append(prob[i][j])
    if len(probs_list) > 0:
        probs_list = list_to_tensor_grad(probs_list)
        return F.binary_cross_entropy(probs_list, from_numpy(np.array(labels_list)).float())
    else:
        return None

def train(train_ptb_path, model_path, resume_path, save_path,
          batch_size = 32, subbatch_size = 8, epochs = 3, max_len = 20,
          num_sent = 128, supervised = False,
          num_grad = 32, train = True, test = True):
    print()
    print("Working directory:", os.path.dirname(os.path.abspath(__file__)))
    print()
    print("Loading train ptb from {}, keeping length <={}".format(train_ptb_path, max_len))
    train_ptb = load_ptb(train_ptb_path)
    train_ptb = [exmp for exmp in train_ptb if len(exmp['sent']) <= max_len]
    train_ptb = train_ptb[:num_sent]
    print("Num train examples: {}".format(len(train_ptb)))
    print(train_ptb[0])
    
    model = BinaryClassifier('roberta-base')
    try:
        model.load_state_dict(torch_load(resume_path)['state_dict'])
    except Exception:
        model.bert.bert = nn.DataParallel(model.bert.bert) # parallelizing affects param names and therefore loading
        model.load_state_dict(torch_load(resume_path)['state_dict'])
    
    TESTS = [ct.produce_clefting, ct.produce_subs, ct.produce_movement_front, 
             ct.produce_movement_end, ct.produce_coordination_repeat]
    
    parser = ConstituencyTestParser(model, TESTS, num_grad = num_grad)
    
    if train:
        trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        trainer = torch.optim.Adam(trainable_parameters, lr=0.00003, betas=(0.9, 0.999), eps=1e-6)
        print()
        print("Training...")
        start_time = time.time()
        num_exmp = len(train_ptb)
        for epoch in range(epochs):
            print()
            print('===================================================')
            np.random.shuffle(train_ptb)
            pred_all = []
            for i in range(0, num_exmp, batch_size):
                model.train()
                trainer.zero_grad()
                
                ptb_batch = train_ptb[i:i+batch_size]
                pred_batch = []
                for j in range(0, len(ptb_batch), subbatch_size):
                    ptb_subbatch = ptb_batch[j:j+subbatch_size]
                    sent_subbatch = [exmp['sent'] for exmp in ptb_subbatch]
                    trees, probs = parser(sent_subbatch)
                    pred_batch.extend([str(t) for t in trees])
                    if supervised:
                        gold = [exmp['tree'] for exmp in ptb_subbatch]
                        loss = compute_estep_loss(gold, probs)
                    else:
                        loss = compute_estep_loss(trees, probs)
                    if loss is not None:
                        loss.backward()
                        del loss
                    del probs
                grad_norm = nn.utils.clip_grad_norm_(trainable_parameters, np.inf)
                trainer.step()
                pred_all.extend(pred_batch)
                
                print("epoch {:,} "
                      "sentences {:,}-{:,}/{:,} "
                      "grad-norm {:.4f} "
                      "total-elapsed {}".format(
                          epoch, 
                          i, i+len(ptb_batch), num_exmp,
                          grad_norm, 
                          format_elapsed(start_time)))                
                
            # check total f1
            gold_all = [exmp['string'] for exmp in train_ptb]
            f1_all = compute_f1_evalb(gold_all, pred_all)
            print()
            print('---------------------------------------------------')
            print("Epoch-f1 {:.4f}".format(f1_all))
            print("Saving model to {}...".format(model_path))
            torch.save({
                'bert_model': 'roberta_base',
                'state_dict': model.state_dict()
                }, model_path)
            print("Done saving.")
                    
    if test:
        # after training, parse ptb and write output to file
        train_ptb = load_ptb(train_ptb_path)
        train_ptb = [exmp for exmp in train_ptb if len(exmp['sent']) <= max_len]
        train_ptb = train_ptb[:num_sent]
        model.eval()
        with torch.no_grad():
            trees = []
            for i in range(0, len(train_ptb), batch_size):
                ptb_batch = train_ptb[i:i+batch_size]
                with torch.no_grad():
                    trees_batch, _, = parser([exmp['sent'] for exmp in ptb_batch])
                trees.extend(trees_batch)
            
        pred_all = [str(t) for t in trees]
        gold_all = [exmp['string'] for exmp in train_ptb]
        f1_all = compute_f1_evalb(gold_all, pred_all)
        print("F1: {}".format(f1_all))
            
        print("Writing trees to file...")
        with open(save_path, "w") as outfile:
            outfile.write("\n".join(pred_all))
                
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: 
        train(args.train_ptb_path, args.model_path, args.resume_path, args.save_path,
          args.batch_size, args.subbatch_size, args.epochs, args.max_len,
          args.num_sent, args.supervised,
          args.num_grad, args.train, args.test))
    subparser.add_argument("--train-ptb-path", default='../data/22.auto.clean')
    subparser.add_argument("--model-path", required=True)
    subparser.add_argument("--resume-path", required=True)
    subparser.add_argument("--save-path", required=True)
    subparser.add_argument("--batch-size", type=int, default=16)
    subparser.add_argument("--subbatch-size", type=int, default=8)
    subparser.add_argument("--epochs", type=int, default=3)
    subparser.add_argument("--max-len", type=int, default=10)
    subparser.add_argument("--num-sent", type=int, default=30)
    subparser.add_argument("--supervised", type=int, default=0)
    subparser.add_argument("--num-grad", type=int, default=32)
    subparser.add_argument("--train", type=int, default=1)
    subparser.add_argument("--test", type=int, default=1)
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    args.callback(args)
    
if __name__ == '__main__':
    main()
