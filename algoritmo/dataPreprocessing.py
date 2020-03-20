import progressbar
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import glob
import os
from scipy import spatial

#qa_json_path = '/home/mary/PycharmProjects/kebdi/data/qa/qa.json'
qa_json_path = '/home/mary/PycharmProjects/kebdi/data/qa/newqa.json'
plots_path = '/home/mary/PycharmProjects/kebdi/data/plot'


def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()


def create_dict(file):
    bar = progressbar.ProgressBar(max_value=400000, redirect_stdout=True) #14415
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


def read_plots(path_plot):
    data_dict = {}

    for file in glob.glob(path_plot + "/*"):
        with open(file) as f:
            key = os.path.basename(file).replace('.wiki', '')
            value = ' '.join(f.readlines())
            data_dict[key] = value

    return data_dict


def elaborate_data(file_json, path_plot):
    with open(file_json) as f:
        qa_dict = json.load(f)

    plot_dict = read_plots(path_plot)

    def prep_data(el):
        return [plot_dict[el['imdb_key']], el['question'], el['answers'][el['correct_index']]]

    data = [prep_data(el) for el in qa_dict if el['correct_index'] is not None and el['imdb_key'] in plot_dict.keys()]

    return data



glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove.6B.300d.txt")

#glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove_dataset_300d.txt")

stop_words = list(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

remove_stop_words = lambda line: [word for word in line if word not in stop_words]

lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]

gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else 1e-5*np.ones(300) for word in x]


def elaborate():
    data = elaborate_data(qa_json_path, plots_path)

    #np.seterr('raise')

    phrase_emb_list = []
    question_emb_list = []
    answer_emb_list = []
    dizionario_list = []
    for i in range(len(data)):
        nplot = data[i][0].replace("\n", '')
        phrases_tmp = nplot.split('.')
        phrasesa = [p for p in phrases_tmp if (len(p) is not 0)]
        phrases = [p for p in phrasesa if (len(p) is not 1)]

        phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
        question_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][1])))
        answer_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][2])))

        question_vect = gloveize_phrase(question_words)
        phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))
        answer_vect = gloveize_phrase(answer_words)

        question_emb = np.mean(question_vect, axis=0)
        phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))
        answer_emb = np.mean(answer_vect, axis=0)

        dcos_all = []
        for k in range(len(phrases_emb)):
            #print("DOMANDA: ", answer_emb)
            #print("RISPOSTA: ", phrases_emb[k])
            dcos = spatial.distance.cosine(answer_emb, phrases_emb[k])
            dcos_all.append(dcos)

        imin = np.argmin(dcos_all)
        phrasemin = phrases_emb[imin]

        dizionario = list()
        for value in phrases_emb:
            dizionario.append(value)
        for val in phrases:
            dizionario.append(val)

        phrase_emb_list.append(phrases_emb)
        question_emb_list.append(question_emb)
        dizionario_list.append(dizionario)
        answer_emb_list.append(phrasemin)

    return [phrase_emb_list, question_emb_list, answer_emb_list, dizionario_list]


if __name__ == '__main__':
    elaborate()
