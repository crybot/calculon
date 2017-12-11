import re
import math
import random
import os.path
from collections import Counter
from cobe.brain import Brain
from generation.cobe_generate import clean_text, read_file

class GenerativeModel(object):
    """ Abstract class for a generative model for text """
    def __init__(self, brain_name):
        self._model_name = brain_name
        self.brain_name = brain_name + ".brain"
        self.brain_questions_name = brain_name + "_questions.brain"
        self.brain = None
        self.brain_questions = None
        self.question_prob = 0.3
        self.similarity_min = 0.6

    @property
    def name(self):
        return self._model_name

    def train(self, corpus):
        self.brain_questions = self._learn_corpus(corpus, self.brain_questions_name, questions=True)
        self.brain = self._learn_corpus(corpus, self.brain_name, questions=False)
        return self

    def generate_start(self):
        start_seed = random.choice(['Hello', 'Hi'])
        line = self.generate(start_seed)
        return line

    def generate(self, context):
        u = random.random()
        while True:
            if self.brain_questions and u < self.question_prob:
                new_line = self.brain_questions.reply(context)
            else:
                new_line = self.brain.reply(context)

            #it gets stuck, trying without this condition
            #if new_line and get_cosine(text_to_vector(context), text_to_vector(new_line)) > self.similarity_min:
            return new_line

    def _learn_corpus(self, corpus_file, brain_name, questions=False):
        if not os.path.isfile(brain_name):
            brain = Brain(brain_name)
            print("- Training...")
            corpus = read_file(corpus_file)
            corpus = clean_text(corpus, get_questions=questions)

            for sent in corpus:
                brain.learn(sent)

        return Brain(brain_name)

    # cosine is calculated using
    # https://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)
