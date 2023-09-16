### example scripts for epsilon=4. Hyperparameters are set according to Table 9.
### For 0-shot, set --num-valid 0
### For 4-shot (epsilon=0), set --num-private-train 0 and --num-private-train-splits 0
### For 4-shot (epsilon=infinity), set --sigma 0 and other parameters as Table 9.

# AGNEWS
for seed in 0 1 2 3 4
do
    python -m dp_few_shot_generation.run_exp_agnews \
    --sigma 0.39 \
    --openai-model "babbage" \
    --num-private-train 20 \
    --set-num-public-train 0 \
    --num-valid 4 \
    --num-private-train-splits 10 \
    --num-test 1000 \
    --use-dp-prompts \
    --sample-same-label-prompts \
    --subsample-per-token \
    --synth-seed $seed \
    --eval-seed $seed 
done

# DBPedia
for seed in 0 1 2 3 4
do
    python -m dp_few_shot_generation.run_exp_dbpedia \
    --sigma 0.45 \
    --openai-model "babbage" \
    --num-private-train 80 \
    --set-num-public-train 0 \
    --num-valid 4 \
    --num-private-train-splits 40 \
    --num-test 1000 \
    --use-dp-prompts \
    --sample-same-label-prompts \
    --subsample-per-token \
    --synth-seed $seed \
    --eval-seed $seed 
done

# TREC
for seed in 0 1 2 3 4
do
    python -m dp_few_shot_generation.run_exp_trec \
    --sigma 0.69 \
    --openai-model "babbage" \
    --num-private-train 80 \
    --set-num-public-train 0 \
    --num-valid 4 \
    --num-private-train-splits 80 \
    --num-test 1000 \
    --no-public-token \
    --use-dp-prompts \
    --sample-same-label-prompts \
    --subsample-per-token \
    --synth-seed $seed \
    --eval-seed $seed 
done

# MIT-G
for seed in 0 1 2 3 4
do
    python -m dp_few_shot_generation.run_exp_movie \
    --sigma 0.64 \
    --openai-model "babbage" \
    --num-private-train 80 \
    --set-num-public-train 0 \
    --num-valid 4 \
    --num-private-train-splits 20 \
    --num-test 1000 \
    --use-dp-prompts \
    --field-name Genre \
    --subsample-per-token \
    --synth-seed $seed \
    --eval-seed $seed 
done

# MIT-D
for seed in 0 1 2 3 4
do
    python -m dp_few_shot_generation.run_exp_movie \
    --sigma 0.77 \
    --openai-model "babbage" \
    --num-private-train 80 \
    --set-num-public-train 0 \
    --num-valid 4 \
    --num-private-train-splits 20 \
    --num-test 1000 \
    --use-dp-prompts \
    --field-name Director \
    --subsample-per-token \
    --synth-seed $seed \
    --eval-seed $seed 
done
