import argparse
import pickle
from enum import Enum

class PromptType(Enum):
    qg = 'qg'
    qg_cot = 'qg_cot'
    qg_fewshot = 'qg_fewshot'
    qg_selfcheck = 'qg_selfcheck'

    qa = 'qa'
    qa_selfcons = 'qa_selfcons'

class ModelType(Enum):
    hf_chat = 'hf_chat'
    open_ai = 'open_ai'
    cohere = 'cohere'
    anthropic = 'anthropic'

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
        "--model_name",
        type=str,
        help="Name of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Directory where results are stored",
        default="./",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=enum_type(PromptType), 
        help='Prompt types/experiments to run', 
        default=[]
    )
    args = parser.parse_args()
    return args

def parse_answer(txt):
    delimiters = ['Answer:']
    if txt == None:
        return None
    lines = txt.split('\n')
    lines = [l for l in lines if sum([s in l for s in delimiters]) > 0]
    for out in lines:
        for token in delimiters:
            if token in out:
                candidate_a = out[out.index(token) + len(token):].strip()
                return None if len(candidate_a) == 0 else candidate_a
    return None

def main(args):

    run_name = args.run_name
    model_name = args.model_name
    res_dir = args.res_dir
    pt = args.prompt_types[0][0]

    f = f'{res_dir}{model_name}/{run_name}/{pt.value}.pkl'
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    raw_out = data['raw_text']
    
    parsed_as = []
    for out in raw_out:
        pa = parse_answer(out)
        if pa == None:
            continue
        parsed_as.append(pa)
    
    f = f'{res_dir}{model_name}/{run_name}/{pt.value}+answer.pkl'
    data['answer'] = parsed_as
    with open(f, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = setup()
    main(args)