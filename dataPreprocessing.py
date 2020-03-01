import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
from glove import Glove, Corpus


# nltk.download()
stop_words = list(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()


# lettura file
def read_file(file, el, off=0, n=500):
    data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < n+off:
                if line_count < off:
                    line_count += 1
                else:
                    line_count += 1
                    data.append(row[el])  # 0 question, 2 synopsis

    return data


# rimozione eventuali simboli a inizio e fine parola e split delle parole per spazio
def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()


# creazione dizionario da glove pretrainato
def create_dict(file):
    embeddings_dict = {}
    with open(file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector


def glove_embedding(lines):
    corpus = Corpus()
    corpus.fit(lines, window=100)
    glove = Glove(no_components=5, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')


# rimozione stopwords
remove_stop_words = lambda line: [word for word in line if word not in stop_words]


# lemmatizzazione
lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]


plots = read_file('data/mpst_full_data.csv', 2, 2, n=10)

questions = read_file('data/questions.txt', 0, n=5)

plots = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), plots))
questions = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), questions))

print(plots)
print(questions)

exit()

# salvatggio dei dati
dataset = []
with open("dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)
