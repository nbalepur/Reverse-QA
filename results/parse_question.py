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

def parse_question(txt):
    delimiters = ['Question:', 'question:', 'question is:']
    end_delimiters = ['...', '?', '_.']

    lines = txt.split('\n')
    lines = [l for l in lines if sum([s in l for s in delimiters]) > 0]
    for out in lines:
        for token in delimiters:
            if token in out:
                candidates = out.split(token)
                for candidate_q in candidates:
                    for end_delim in end_delimiters:
                        if end_delim in candidate_q:
                            candidate_q = candidate_q[:candidate_q.index(end_delim) + 1].strip()
                            return candidate_q
    return None

def main(args):

    pt = args.prompt_types[0][0]
    res_dir = args.res_dir
    model_name = args.model_name
    run_name = args.run_name

    f = f'{res_dir}{model_name}/{run_name}/{pt.value}.pkl'
    with open(f, 'rb') as handle:
        data = pickle.load(handle)
    raw_out = data['raw_text']

    parsed_qs = []

    for out in raw_out:
        pq = parse_question(out)
        if pq == None:
            continue
        parsed_qs.append(pq)

    f = f'{res_dir}{model_name}/{run_name}/{pt.value}+question.pkl'
    data['question'] = parsed_qs
    with open(f, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = setup()
    main(args)