import argparse
import trees as tr
from eval_for_comparison import pcfg_compute_f1

def eval_parser_output(fname):
    print("Note: this function uses scripts from past work to evaluate (NOT evalb).")
    print("These numbers should match those in the paper.")
    print(fname)
    
    treebank = tr.load_ptb('../data/23.auto.clean_fixed', lower = False)
    treebank_nopunct = tr.load_ptb('../data/23.auto.clean_fixed', nopunct = True, lower = False)
    treebank_compare = tr.load_ptb('../data/ptb-test.txt', lower = False)
    
    pred = tr.load_ptb(fname)
    # align ptb-test with 23.auto.clean
    treebank_compare_aligned = []
    sents1 = [exmp['sent'] for exmp in treebank_nopunct]
    sents2 = [exmp['sent'] for exmp in treebank_compare]
    for sent in sents1:
        if sent not in sents2:
            print(sent)
            sent = ['The', 'only', 'thing', 'you', 'do', "n't", 'have', 'he', 'said', 
                    'is', 'the', 'portfolio', 'insurance', 'phenomenon', 'overlaid', 'on', 'the', 'rest']
        ind = sents2.index(sent)
        treebank_compare_aligned.append(treebank_compare[ind])
        
    # remove punct from pred for comparison
    for tree1, tree2 in zip(pred, treebank):
        try:
            tr.transfer_leaves(tree1['tree'], tree2['tree'])
        except Exception as e:
            print(e)
            print(tree1['sent'], tree2['sent'])
        tree1['tree'] = tr.remove_labels(tr.collapse_unary_chains(tr.clean_empty(tr.standardize_punct(tree1['tree'], True))))
        if len(tree1['tree'].sent()) == 1:
            print(tree1['tree'].sent())
            tree1['tree'] = tr.Tree('X', [tree1['tree']], None)
    
    f1 = pcfg_compute_f1([str(exmp['tree']) for exmp in treebank_compare_aligned], [str(exmp['tree']) for exmp in pred])
    print(f1)
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("eval")
    subparser.set_defaults(callback=lambda args: 
        eval_parser_output(args.save_path))
    subparser.add_argument("--save-path", required=True)
    
    args = parser.parse_args()
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    args.callback(args)
    
if __name__ == '__main__':
    main()