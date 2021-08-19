import nltk
from nltk.stem.porter import PorterStemmer
import numpy

stemmer = PorterStemmer()


def Tokenize(sentence):
    return nltk.word_tokenize(sentence)


def Stem(word):
    return stemmer.stem(word.lower())


def Bag_O_Words_Func(tokenized_sentence, all_words):
    tokenized_sentence = [Stem(word) for word in tokenized_sentence]
    bag = numpy.zeros(len(all_words), dtype=numpy.float32)

    for (index, word) in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag
