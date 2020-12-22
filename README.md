# Unsupervised Parsing via Constituency Tests
This is code for the paper:  
[Unsupervised Parsing via Constituency Tests](https://www.aclweb.org/anthology/2020.emnlp-main.389/)  
Steven Cao, Nikita Kitaev, Dan Klein  
EMNLP 2020  
### Dependencies
This code was tested with `python 3.6`, `pytorch 1.1`, and `pytorch-transformers 1.2`.
### Data
The Penn Treebank and [CoLA](https://nyu-mll.github.io/CoLA/) data are contained in the `data` folder. The folder also contains a few sentences from Gigaword to show the formatting; for the full data please [download it from the LDC](https://catalog.ldc.upenn.edu/LDC2011T07).
### Running the code
To run the main experiment in the paper, see `run_full.sh`. To reduce the memory usage, reduce both `--subbatch-size` and `--num-grad` while ensuring that the ratio between them stays the same (`num-grad` divided by `subbatch-size` should be 16).
### Note regarding evaluation
The code contains two ways of computing parser F1: `evalb`, which is standard in supervised parsing evaluation, and a custom script used in past grammar induction work (see `eval_for_comparison.py`, taken from [the Compound PCFG github repo](https://github.com/harvardnlp/compound-pcfg)). The latter ignores punctuation (among other differences; see the paper for details) and typically results in higher F1 numbers.
### Citation
    @inproceedings{cao-etal-2020-unsupervised-parsing,
        title = "Unsupervised Parsing via Constituency Tests",
        author = "Cao, Steven  and
          Kitaev, Nikita  and
          Klein, Dan",
        booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.emnlp-main.389",
        doi = "10.18653/v1/2020.emnlp-main.389",
        pages = "4798--4808",
    }
