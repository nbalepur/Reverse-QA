# activate your environment here!
source ...
conda activate ...

# name of the model (identified by the API)
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
# run to extract
run_name="default"
# experiment to extract
experiments=("qa")
experiments_str=$(IFS=" "; echo "${experiments[*]}")
# results directory
res_dir=".../QG-vs-QA/results/"

python3 .../QG-vs-QA/results/parse_answer.py \
--run_name="$run_name" \
--model_name="$model_name" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir"