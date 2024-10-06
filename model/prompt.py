from abc import ABC, abstractmethod
import random
import copy
from enums import PromptType

# Abstract base class for implementing zero-shot prompts
class ZeroShotPrompt(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def create_prompt(self, data):
        """Create a zero-shot prompt"""
        pass

# Question Generation Prompts
class QuestionGenerationVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]". If no possible question exists say "IDK".'
        return prompt

class QuestionGenerationCoT(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Think step by step and reason before generating the question. After reasoning, please format your final output as "Question: [insert generated question]".'
        return prompt

class QuestionGenerationCheckAnswer(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]". After generating a question, answer your own question to verify that the answer is "{answer}", formatted as "Answer: [insert answer to generated question]".'
        return prompt

class QuestionGenerationFewShot(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = 'Generate a one-sentence question with the answer: "{answer}". The only possible answer to the question must be "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]".'
        prompt += """

Answer: 328
Question: What is the sum of the first 15 prime numbers?

Answer: 710 survivors
Question: How many people survived the sinking of the RMS Titanic in 1912?

Answer: 648
Question: What is the product of 12 and 54?

Answer: 286 ayats
Question: How many verses are there in the longest chapter of the Quran, Surah Al-Baqarah?

Answer: 311
Question: What is the sum of the first three prime numbers greater than 100?

"""   
        prompt += f"Answer: {answer}\nQuestion:"
        return prompt

# Question Answering Prompts
class QuestionAnsweringVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        question = data['input']
        prompt = f'Generate the answer to the question: "{question}". Give just the answer and no explanation. Please format your output as "Answer: [insert generated answer]". If no possible answer exists say "IDK".'
        return prompt

class PromptFactory:

    def __init__(self):

        self.prompt_type_map = {
            PromptType.qg: QuestionGenerationVanilla,
            PromptType.qg_cot: QuestionGenerationCoT,
            PromptType.qg_fewshot: QuestionGenerationFewShot,
            PromptType.qg_selfcheck: QuestionGenerationCheckAnswer,

            PromptType.qa: QuestionAnsweringVanilla,
            PromptType.qa_selfcons: QuestionAnsweringVanilla,
        }

    def get_prompt(self, prompt_type):
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]()
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")
