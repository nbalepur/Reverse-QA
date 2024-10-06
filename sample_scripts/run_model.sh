#!/bin/bash

# activate your environment here!
source ...
conda activate ...

# dataset details
inference_split="train"
dataset_name="nbalepur/QG-vs-QA"

# name of the model (identified by the API)
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# model type (see enums.py). Currently supported: hf_chat (Huggingface), open_ai, cohere, anthropic
model_type="hf_chat"

# how to identify this run
run_name="default"

# API tokens
hf_token=... # huggingface read token (for downloading gated models)
open_ai_token=... # OpenAI token (for GPT models)
cohere_token=... # Cohere token (Command-R)
anthropic_token=... # Anthropic token (Claude)

# generation parameters
temperature=0.0
min_tokens=5
max_tokens=1000

device_map="auto" # device map ('cpu', 'cuda', 'auto')
partition="full"  # partition of the dataset. can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")

# experiment to run
# see all possible experiments in: /mcqa-artifacts/model/data_loader.py
experiments=("qa_selfcons")

# directory setup
res_dir=".../QG-vs-QA/results/" # Results folder directory
cache_dir="..." # Cache directory to save the model

experiments_str=$(IFS=" "; echo "${experiments[*]}")

# add the correct file below
# there are also flags for `load_in_4bit` and `load_in_8bit`
python3 .../QG-vs-QA/model/run_model.py \
--run_name="$run_name" \
--model_nickname="$model_name" \
--model_name="$model_name" \
--model_type="$model_type" \
--dataset_name="$dataset_name" \
--inference_split="$inference_split" \
--partition="$partition" \
--hf_token="$hf_token" \
--open_ai_token="$open_ai_token" \
--cohere_token="$cohere_token" \
--anthropic_token="$anthropic_token" \
--device_map="$device_map" \
--temperature="$temperature" \
--min_tokens="$min_tokens" \
--max_tokens="$max_tokens" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir" \
--cache_dir="$cache_dir"
