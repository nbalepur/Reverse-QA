# QG vs QA

This repository is the official implementation of "Question Answering Meets Answer Questioning: Exposing LLM Inconsistencies in Deductive and Abductive Reasoning", which will soon be uploaded to Arxiv

## Overview

This repository contains the code and dataset to compare the reliability of LLM abduction and deduction, evaluated through zero-shot question generation and question answering, respectively.

## Dataset

Our dataset contains 3450 question/answer pairs across four categories (Number, Number + Text, Easy Facts, Hard Facts), and can be accessed through Huggingface [here](https://huggingface.co/datasets/nbalepur/QG-vs-QA) 

## Setup

Python 3.10.0, pip 23.2.1, and CUDA Version: 12.4 were used when running the code in this repository. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:

```
pip install -r requirements.txt 
```

The files in this repository are organized as follows:
* `/model/`: Contains the code to run the three-step prompting pipeline
* `/evaluation/`: Contains the code to evaluate the accuracy and consistency of abduction and deduction
* `/results/`: Directory for storing model outputs and contains code for parsing model outputs
* `/sample_scripts/`: Scripts to easily run the inference code in `/model/`

## Model Inference

You can run inference on the Huggingface models with the following command: 
```bash
bash .../scripts/model.py
```
You can change the following parameters for each run:
* `dataset_name`: where the dataset can be accessed on huggingface/locally
* `inference_split`: split of the dataset to run inference on
* `model_name`: Name of the model on the API. String type
* `model_type`: Type of the model, i.e., where it is accessed from. All available models are listed in `model/enums.py` Currently supports "hf_chat" (HuggingFace chat models), "open_ai" (OpenAI models), "cohere" (Cohere models), and "anthropic" (Anthropic models). String type
* `run_name`: Identifier for the run. String type
*  `device_map`: Device map for the GPUs ("cpu", "cuda", "auto"). String type
*  `partition`: Partition of the dataset. can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")
*  `experiments`: List of strings denoting experiments to run. Currently supports "qg" (0-shot QG), "qg_cot" (0-shot QG with CoT), "qg_selfcheck", (LLM checks its own answer), "qg_fewshot" (few-shot QG, just for nuemrical entities),  "qa" (0-shot QA), and "qa_selfcons" (QA on LLM's own generated question)

API tokens (depending on the model) can be specified:
* `hf_token`: Huggingface read token (for downloading gated models and datasets). String type
* `open_ai_token`: OpenAI token (fGPT models). String type
* `cohere_token`: Cohere token (Command-R models). String type
* `anthropic_token`: Anthropic token (Claude models). String type

You can also specify the following generation hyperparameters:
* `temperature`: Model temperature. Float type
* `min_tokens`: Minimum tokens to generate. Integer type
* `max_tokens`: Maximum tokens to generate. Integer type

Finally, the following parameters set up the directories for storing model outputs:
* `res_dir`: Pointing to the folder where results are stored
* `cache_dir`: Pointing to the folder where model and dataset downloads can be cached through huggingface

## Evaluation

After running any question answering or generating prompt, you can parse the results with the following Python files in `/results/`:
* `parse_answer.py`
* `parse_question.py`

Both methods use the following parameters:
*  `experiments`: List of strings denoting experiments to run. Currently supports "qg" (0-shot QG), "qg_cot" (0-shot QG with CoT), "qg_selfcheck", (LLM checks its own answer), "qg_fewshot" (few-shot QG, just for nuemrical entities),  "qa" (0-shot QA), and "qa_selfcons" (QA on LLM's own generated question)
* `run_name`: Identifier for the run. String type
* `model_name`: Name of the model on the API. String type
* `res_dir`: Pointing to the folder where results are stored

Please note that to run the self-consistency check, you must perform a four-step process of:
1. `run_model.py` with a QG prompt
2. `parse_question.py` with the prompt from (1)
3. `run_model.py` with the QA prompt `qa_selfcons`
4. `parse_answer.py` with the QA prompt `qa_selfcons`

The relevant repositories for computing question difficulty and token count are below:
* Question Difficulty: [Prometheus LLM](https://github.com/prometheus-eval/prometheus-eval)
* Token Count: [Infini-Gram](https://huggingface.co/spaces/liujch1998/infini-gram)
If you would like to have these re-implemented on this repo, please raise an issue and let us know!

## Contact

If you have questions, please feel free to raise an issue or contact either of the following authors of the repository:
- [Nishant Balepur](mailto:nbalepur@umd.edu)
- [Feng Gu](mailto:fgu1@umd.edu)


