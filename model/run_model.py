# imports and directory setup
from data_loader import PromptCollator
from checkpoint_handler import Checkpoint
from model_loader import ModelFactory
from enums import ModelType, PromptType

import pickle
import datasets
import tqdm
import os
import copy
import argparse

# =========================================== Argument Setup ===========================================

def enum_type(enum):
    enum_members = {e.name: e for e in enum}

    def converter(input):
        out = []
        for x in input.split():
            if x in enum_members:
                out.append(enum_members[x])
            else:
                raise argparse.ArgumentTypeError(f"You used {x}, but value must be one of {', '.join(enum_members.keys())}")
        return out

    return converter

def setup():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="String to identify this run",
        default="",
    )
    parser.add_argument(
        "--model_nickname",
        type=str,
        help="Nickname of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model on huggingface/OpenAI",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model_type",
        type=enum_type(ModelType),
        help="Type of the model: hf_chat",
        default="hf_chat",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Local path or huggingface path pointing to the dataset",
        default="nbalepur/QG_vs_QA_v2",
    )
    parser.add_argument(
        "--inference_split",
        type=str,
        help="Split of the dataset to use",
        default="full",
    )
    parser.add_argument(
        "--load_in_8bit",
        action='store_true',
        help="Should we load the model in 8 bit?",
        default=False,
    )
    parser.add_argument(
        "--load_in_4bit",
        action='store_true',
        help="Should we load the model in 4 bit?",
        default=False,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature of the model to use",
        default=0.7,
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        help="Minimum number of tokens to generate",
        default=5,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum number of tokens to generate",
        default=200,
    )
    parser.add_argument(
        "--device_map",
        type=str,
        help="Where to load the model ('cuda', 'auto', 'cpu')",
        default="auto",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface read token for access to models/datasets",
        default="",
    )
    parser.add_argument(
        "--open_ai_token",
        type=str,
        help="OpenAI token for access to the model",
        default="",
    )
    parser.add_argument(
        "--cohere_token",
        type=str,
        help="Cohere token for access to the model",
        default="",
    )
    parser.add_argument(
        "--anthropic_token",
        type=str,
        help="Anthropic token for access to the model",
        default="",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=enum_type(PromptType), 
        help='Prompt types/experiments to run', 
        default=[]
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Which partition should be done",
        default="none",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Absolute directory of the prompt folder",
        default="./",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Absolute directory of the cache folder for models",
        default="./",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the output results folder",
        default="./",
    )
    args = parser.parse_args()
    print(args)
    return args

# =========================================== Main Method ===========================================

def main(args):

    # load model
    model_factory = ModelFactory()
    model = model_factory.get_model(args)

    # load checkpoints
    checkpoint_loader = Checkpoint(args)
    
    # load prompt collator
    prompt_collator = PromptCollator(args)

    for pt in args.prompt_types[0]:

        # set output directories
        checkpoint_loader.set_directories(pt) 

        # get prompts and dataset size
        prompts = prompt_collator.get_prompts(pt, checkpoint_loader)
        
        # set up inference bounds
        start, end = checkpoint_loader.setup_partition(len(prompts))

        # load current save state and adjust starting point
        outputs = checkpoint_loader.load_checkpoint()
        start += len(outputs['raw_text'])

        # iterate through outputs and generate the response
        for idx in tqdm.tqdm(range(start, end)):
            prompt = prompts[idx]
            
            if prompt == None:
                outputs['raw_text'].append(None)
                outputs['prompt'].append(None)
                continue

            out_text = model.generate_text(prompt)

            outputs['raw_text'].append(out_text)
            outputs['prompt'].append(prompt)
            checkpoint_loader.save_checkpoint(outputs, False)

        # final save
        checkpoint_loader.save_checkpoint(outputs, True)

if __name__ == '__main__':
    
    args = setup()
    main(args)