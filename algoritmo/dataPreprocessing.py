import progressbar
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np


def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()


def create_dict(file):
    bar = progressbar.ProgressBar(max_value=400000, redirect_stdout=True)
    i = 0

    embeddings_dict = {}
    with open(file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float)
            embeddings_dict[word] = vector
            bar.update(i)
            i += 1

    bar.finish()
    return embeddings_dict


#glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove.6B.300d.txt")

glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove_dataset_300d.txt")

stop_words = list(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

remove_stop_words = lambda line: [word for word in line if word not in stop_words]

lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]

gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else np.zeros(300) for word in x]


def elaborate(plot, question):
    nplot = plot.replace("\n", '')
    phrases_tmp = nplot.split('.')
    phrasesa = [p for p in phrases_tmp if (len(p) is not 0)]
    phrases = [p for p in phrasesa if (len(p) is not 1)]
    #phrases = plot.split('.')
    np.seterr('raise')
    phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
    question_words = lemmatize_words(remove_stop_words(clean_split_string(question)))


    question_vect = gloveize_phrase(question_words)
    phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))

    question_emb = np.mean(question_vect, axis=0)
    phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))

    dizionario = list()
    for value in phrases_emb:
        dizionario.append(value)
    for val in phrases:
        dizionario.append(val)

    return [phrases_emb, question_emb, dizionario]
