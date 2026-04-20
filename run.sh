#!/bin/bash

# Dev run (small sample)
python main.py \
    --data MMLU_sampled \
    --eval dev \
    --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
    --meta_llm qwen \
    --graph_pooling_method mean \
    --top_k 3 \
    --threshold 0.05 \
    --rounds 1 \
    --temperature 0.7 \
    --max_tokens 800 \
    --num_proc 80 \
    --seed 0

# Test run (full evaluation)
# python main.py \
#     --data MMLU_sampled \
#     --eval test \
#     --reference_models qwen,qwen_coder,mathstral,biomedical_llama,finance_llama,saul \
#     --meta_llm qwen \
#     --graph_pooling_method mean \
#     --top_k 3 \
#     --threshold 0.05 \
#     --rounds 1 \
#     --temperature 0.7 \
#     --max_tokens 800 \
#     --num_proc 80 \
#     --seed 0