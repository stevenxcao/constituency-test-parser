conda activate pytorch1.1

stdbuf -oL -eL python -u train_grammar_model.py train \
  --model-path roberta_base_grammar_gigaword_2M.pt \
  --train-giga-path ../data/gigaword \
  --dev-cola-path ../data/in_domain_dev.tsv \
  --epochs 1 --bert-model roberta-base --reinit 0 --num-sent 2000000 \
  > out_train_grammar_model.txt

for (( i=0; i<4; i++ ));
do
  RUN=$i
  
  stdbuf -oL -eL python -u em_training_efficient.py train \
    --train-ptb-path ../data/02-21.10way.clean \
    --resume-path roberta_base_grammar_gigaword_2M.pt \
    --model-path roberta_base_grammar_gigaword_2M_emtrain_${RUN}.pt \
    --save-path None \
    --max-len 40 --num-sent 5000 --supervised 0 --train 1 --test 0 \
    --num-grad 64 --batch-size 32 --subbatch-size 4 --epochs 1 \
    > out_train_parser_emtrain_${RUN}.txt
    
  stdbuf -oL -eL python -u em_training_efficient.py train \
    --train-ptb-path ../data/23.auto.clean \
    --resume-path roberta_base_grammar_gigaword_2M_emtrain_${RUN}.pt \
    --model-path None \
    --save-path roberta_base_grammar_gigaword_2M_emtrain_emtest_output_${RUN}.txt \
    --max-len 100000 --num-sent 100000 --supervised 0 --train 0 --test 1 \
    --num-grad 64 --batch-size 32 --subbatch-size 4 --epochs 1 \
    > out_train_parser_emtrain_emtest_${RUN}.txt
    
  stdbuf -oL -eL python -u eval_output.py eval \
    --save-path roberta_base_grammar_gigaword_2M_emtrain_emtest_output_${RUN}.txt \
    > out_train_parser_emtrain_emtest_eval_${RUN}.txt
done
