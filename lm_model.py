from collections import Counter
import numpy as np

"""
CS6120 Homework 2 - Nisharg Gosai
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS


def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
Args:
  tokens (list): a list of tokens as strings
  n (int): the length of n-grams to create

Returns:
  list: list of tuples of strings, each tuple being one of the individual n-grams
"""
    if n < 1:
        raise ValueError("The value of n must be greater than or equal to 1.")

    n_grams = []

    for i in range(0, len(tokens) - n + 1):
        n_gram = tokens[i:i + n]
        n_grams.append(tuple(n_gram))

    return n_grams


def read_file(path: str) -> list:
    """
Reads the contents of a file in line by line.
Args:
  path (str): the location of the file to read

Returns:
  list: list of strings, the contents of the file
"""

    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
Tokenize a single string. Glue on the appropriate number of
sentence begin tokens and sentence end tokens (ngram - 1), except
for the case when ngram == 1, when there will be one sentence begin
and one sentence end token.
Args:
  line (str): text to tokenize
  ngram (int): ngram preparation number
  by_char (bool): default value True, if True, tokenize by character, if
    False, tokenize by whitespace
  sentence_begin (str): sentence begin token value
  sentence_end (str): sentence end token value

Returns:
  list of strings - a single line tokenized
"""

    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int,
             by_char: bool = True,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
Tokenize each line in a list of strings. Glue on the appropriate number of
sentence begin tokens and sentence end tokens (ngram - 1), except
for the case when ngram == 1, when there will be one sentence begin
and one sentence end token.
Args:
  data (list): list of strings to tokenize
  ngram (int): ngram preparation number
  by_char (bool): default value True, if True, tokenize by character, if
    False, tokenize by whitespace
  sentence_begin (str): sentence begin token value
  sentence_end (str): sentence end token value

Returns:
  list of strings - all lines tokenized as one large list
"""

    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram_order: int):
        """
        Initializes an untrained LanguageModel.

        Args:
            n_gram_order (int): the n-gram order of the language model to create
        """
        # Attribute declaration
        self.input_tokens = None
        self.unknown_filtered = None
        self.n_gram_order = n_gram_order
        self.language_model = {}
        self.vocabulary = None
        self.vocabulary_size = 0

    def train(self, input_tokens: list = None, replace_unknown: bool = True) -> None:
        """
        Trains the language model on the given data.

        Args:
            input_tokens (list): (optional) tokens
            replace_unknown (bool): default value True, replace 1 frequency tokens with UNK during training
        """
        if input_tokens:
            self.input_tokens = input_tokens

        self.vocabulary = Counter(self.input_tokens)

        # UNK modification logic
        if replace_unknown:
            self.unknown_filtered = {k: v for k, v in self.vocabulary.items() if v <= 1}
            self.vocabulary = {k: v for k, v in self.vocabulary.items() if v > 1}

            if self.unknown_filtered:
                self.vocabulary[UNK] = sum(self.unknown_filtered.values())

                # Replacing tokens itself with UNK
                self.input_tokens = [UNK if x in self.unknown_filtered else x for x in self.input_tokens]

        self.vocabulary_size = len(set(self.vocabulary.keys()))

        n_grams = create_ngrams(self.input_tokens, self.n_gram_order)

        # Training the model with ngram window
        for n_gram_window in n_grams:
            context = n_gram_window[:-1]
            word = n_gram_window[-1]

            if context not in self.language_model:
                self.language_model[context] = {}
            if word not in self.language_model[context]:
                self.language_model[context][word] = 0

            self.language_model[context][word] += 1

    def score(self, sentence_tokens: list) -> float:
        """
        Calculates the probability score for a given string representing a single sequence of tokens.

        Args:
            sentence_tokens (list): a sentence token to be scored by this model

        Returns:
            float: the probability value of the given tokens for this model
        """
        # Replace unknown tokens with UNK
        sentence_tokens = [UNK if token in self.unknown_filtered else token for token in sentence_tokens]
        sentence_tokens = [UNK if token not in self.vocabulary else token for token in sentence_tokens]

        n_grams = create_ngrams(sentence_tokens, self.n_gram_order)
        probability = 1

        for token_set in n_grams:
            context = token_set[:-1]
            word = token_set[-1]

            # Some logic for UNK context or words
            c_sentence = self.language_model.get(context, {}).get(word, 0)

            # Laplace Smoothing
            set_probability = (c_sentence + 1) / (
                    sum(self.language_model.get(context, {}).values()) + self.vocabulary_size)

            probability *= set_probability

        return probability

    def generate_sentence(self) -> list:
        """
        Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
            list: the generated sentence as a list of tokens
        """
        sentence = ([SENTENCE_BEGIN] * (self.n_gram_order - 1)) if self.n_gram_order > 1 else [SENTENCE_BEGIN]
        word_chosen = None

        # Generate words until we get sentence end
        while word_chosen != SENTENCE_END:
            # Finding list of all possible words for given context
            window_start = len(sentence) - self.n_gram_order + 1
            n_gram_window = tuple(sentence[window_start:])
            words_possible = self.language_model.get(n_gram_window, {})

            # Filter out SENTENCE_BEGINS
            words_possible = {k: v for k, v in words_possible.items() if k != SENTENCE_BEGIN}
            total_context_words = sum(words_possible.values())

            # Probability calculation
            probabilities = [v / total_context_words for v in words_possible.values()]
            words_possible = list(words_possible.keys())

            # Choosing the word using Shannon technique
            words_possible_indices = list(range(len(words_possible)))
            word_chosen_index = np.random.choice(words_possible_indices, size=1, p=probabilities)

            word_chosen = words_possible[word_chosen_index[0]]
            sentence.append(word_chosen)

        sentence = sentence + ([SENTENCE_END] * (self.n_gram_order - 2))
        return sentence

    def generate(self, num_sentences: int) -> list:
        """
        Generates n sentences from a trained language model using the Shannon technique.

        Args:
            num_sentences (int): the number of sentences to generate

        Returns:
            list: a list containing lists of strings, one per generated sentence
        """
        return [self.generate_sentence() for _ in range(num_sentences)]

    def perplexity(self, token_sequence: list) -> float:
        """
        Calculates the perplexity score for a given sequence of tokens.

        Args:
            token_sequence (list): a tokenized sequence to be evaluated for perplexity by this model

        Returns:
            float: the perplexity value of the given sequence for this model
        """
        one_by_score = 1 / self.score(token_sequence)
        perplexity_value = one_by_score ** (1 / len(token_sequence))

        return perplexity_value
