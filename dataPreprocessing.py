import numpy as np
import progressbar
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import spatial


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


glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove.6B.300d.txt")
stop_words = list(stopwords.words('english'))

wordnet_lemmatizer = WordNetLemmatizer()

remove_stop_words = lambda line: [word for word in line if word not in stop_words]

lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]

gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else 1e-5*np.ones(300) for word in x]


def main():
    in_questions_train = []
    in_phrases_train = []
    out_answer_train = []


def elaborate(plot, question, answer, padding=100):
    plot = plot.replace("\n", '')
    phrases_tmp = plot.split('.')
    phrases = [p for p in phrases_tmp if p is not '']

    phrases_words = list(map(lambda x: lemmatize_words(clean_split_string(x)), phrases)) #list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
    question_words = lemmatize_words(clean_split_string(question)) #lemmatize_words(remove_stop_words(clean_split_string(question)))
    answer_words = lemmatize_words(clean_split_string(answer)) #lemmatize_words(remove_stop_words(clean_split_string(answer)))

    question_vect = gloveize_phrase(question_words)
    answer_vect = gloveize_phrase(answer_words)
    phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))

    question_emb = np.mean(question_vect, axis=0)
    answer_emb = np.mean(answer_vect, axis=0)
    phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))

    dcos_all = []
    for i in range(len(phrases_emb)):
        dcos = spatial.distance.cosine(answer_emb, phrases_emb[i])
        dcos_all.append(dcos)

    imin = np.argmin(dcos_all)

    emb_size = phrases_emb[0].shape[0]
    plots = np.zeros((padding, emb_size))
    for i in range(len(phrases_emb)):
        plots[i] = phrases_emb[i]
    
    phrases_emb = plots

    #print(question)
    #print(answer)
    #print(phrases[imin])

    return [phrases_emb, question_emb, imin]


def load_dataset():
    train_phrases = np.load('n_train_phrases.npy')
    train_questions = np.load('n_train_questions.npy')
    train_answers = np.load('n_train_answers.npy')

    return [train_phrases, train_questions, train_answers]



if __name__ == '__main__':
    main()