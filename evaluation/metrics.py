import pickle
import datasets
import numpy as np
import dspy
import tqdm
import re
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate


# set these lists to what you want to evaluate!
true_questions = []
generated_questions = []
true_answers = []
generated_answers = []
answer_types = []

# point to the directory of the DSPy optimizer prompts (in `evaluation/dspy_prompts``)
dspy_prompt_dir = ''
# where to save the metrics
res_dir = ''

# ********************* Numerical Answer Equivalence (Deduction) *********************

def numerical_equivalence(a: str, b: str, category: str) -> float:
    match category:
        case "num":
            return int(a in b)

        case "num_text":
            num = re.findall(num_pattern, a)
            if len(num) == 0:
                return 0
            return int(str(int(float(num[0]) + 0.5)) in b)

        case _:
            raise ValueError(f"Invalid category for numerical equivalence: {category}")

# ********************* Textual Answer Equivalence (Deduction) *********************

class AnswerEquivalence(dspy.Signature):
    """Given two answers, your goal is to determine whether the two answers are semantically equivalent and thus equal to each other. Output 1 if the answers are equivalent and 0 if they are not equivalent."""
    answer1 = dspy.InputField(desc="First Answer", prefix="Answer 1:")
    answer2 = dspy.InputField(desc="Second Answer", prefix="Answer 2:")
    equivalent = dspy.OutputField(desc="Binary label denoting whether the two answers are equivalent. 1 means they are equivalent and 0 means they are not", prefix="Equivalent:")

class AnswerEquivalenceFewShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(AnswerEquivalence)

    def forward(self, answer1, answer2):
        return self.generate_answer(answer1=answer1, answer2=answer2)

# ********************* Question Verifier (Abduction) *********************

class AnswerVerifier(dspy.Signature):
    """Given a question and a candidate answer, your goal is to determine whether or not the candidate answer can correctly answer the question."""
    question = dspy.InputField(desc="Question", prefix="Answer 1:")
    candidate_answer = dspy.InputField(desc="Candidate answer for the question", prefix="Answer 2:")
    is_correct = dspy.OutputField(desc="Binary label denoting whether the candidate answer correctly answers the question. 1 means it is a correct answer and 0 means it is not", prefix="Is Correct:")

class AnswerVerifierFewShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(AnswerVerifier)

    def forward(self, question, candidate_answer):
        return self.generate_answer(question=question, candidate_answer=candidate_answer)

# ********************* Set up DSPy data *********************

ae_dspy_testset = []
ded_dspy_testset = []
abd_dspy_testset = []

text_idxs = []
idx = -1
for q_true, q_pred, a_true, a_pred, a_type in zip(true_questions, generated_questions, true_answers, generated_answers, answer_types):
    idx += 1
    if 'num' not in a_type: 
        ex = dspy.Example(answer1=a_true, answer2=a_pred, equivalent='1')
        ae_dspy_testset.append(ex.with_inputs("answer1", "answer2"))
        text_idxs.append(idx)

    ex = dspy.Example(question=q_true, candidate_answer=a_pred, is_correct='1')
    abd_dspy_testset.append(ex)

    ex = dspy.Example(question=q_pred, candidate_answer=a_pred, is_correct='1')
    ded_dspy_testset.append(ex)

# ********************* Run DSPy Inference *********************
def parse(o):
    if '1' in str(o) and '0' not in str(o):
        return 1
    return 0

ded_pred = []
def ded_metric(example, pred, trace=None):
    gold, pred = example.is_correct, pred.is_correct
    pred_text.append(parse(pred))

clf = AnswerVerifierFewShot()
clf.load(f"{dspy_prompt_dir}verifier.json")
evaluator = Evaluate(devset=ded_dspy_testset, num_threads=1, display_progress=True, display_table=0)
evaluator(clf, metric=ded_metric)

abd_pred = []
def abd_metric(example, pred, trace=None):
    gold, pred = example.is_correct, pred.is_correct
    abd_pred.append(parse(pred))

clf = AnswerVerifierFewShot()
clf.load(f"{dspy_prompt_dir}verifier.json")
evaluator = Evaluate(devset=abd_dspy_testset, num_threads=1, display_progress=True, display_table=0)
evaluator(clf, metric=abd_metric)

ae_pred = []
def ae_metric(example, pred, trace=None):
    gold, pred = example.equivalent, pred.equivalent
    ae_pred.append(parse(pred))

clf = AnswerVerifierFewShot()
clf.load(f"{dspy_prompt_dir}ae.json")
evaluator = Evaluate(devset=ae_dspy_testset, num_threads=1, display_progress=True, display_table=0)
evaluator(clf, metric=ae_metric)

# ********************* Save Final Outputs *********************

abd_accuracy = abd_pred
ded_accuracy = [numerical_equivalence(true_answers[idx], generated_answers[idx]) if idx in text_idxs else ae_pred[text_idxs.index(idx)] for idx in range(len(true_questions))]
answered_gen_q_correctly = ded_pred
with open(res_dir, 'wb') as handle:
    pickle.dump({'abduction_accuracy': abd_accuracy, 'deduction_accuracy': ded_accuracy, 'answered_own_question': answered_gen_q_correctly}, handle, protocol=pickle.HIGHEST_PROTOCOL)