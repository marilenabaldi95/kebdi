import numpy as np
import progressbar
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from glove import Corpus, Glove
from scipy import spatial
import os
import glob
import json

glove_path = "/home/mary/PycharmProjects/kebdi/data/glove/glove_dataset_300d.txt"
qa_json_path = '/home/mary/Scrivania/qa.json'
plots_path = '/home/mary/PycharmProjects/kebdi/data/plot'


wordnet_lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

remove_stop_words = lambda line: [word for word in line if word not in stop_words]

lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]


def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()

#genera dizionario a partire dal file di testo di glove (parola-vettore)
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


#train di glove sul corpus dato da tutte le trame del dataset -> scrive il file di testo parola-vettore
def create_glove():
    data = elaborate_data(qa_json_path, plots_path)

    all_plots = ''
    all_questions = ''
    for i in range(len(data)):
        all_plots = all_plots + '\n' + data[i][0]
        all_questions = all_questions + '\n' + data[i][1]

    all_plots = all_plots.replace("\n", '')
    all_questions = all_questions.replace('\n', '')
    phrases_tmp = all_plots.split('.')
    questions_tmp = all_questions.split('?')
    phrases = [p for p in phrases_tmp if p is not '']
    questions = [q for q in questions_tmp if q is not '']

    phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
    question_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), questions))
    all_words = phrases_words + question_words
    print(all_words)

    '''
    result_phrases = []
    duplicates_phrases = []
    for phrase in phrases_words:
        unique = []
        for word in phrase:
            if word not in duplicates_phrases:
                unique.append(word)
                duplicates_phrases.append(word)
        result_phrases.append(unique)
    unique_phrases_words = result_phrases

    result_questions = []
    duplicates_questions = []
    for phrase in question_words:
        unique = []
        for word in phrase:
            if word not in duplicates_questions:
                unique.append(word)
                duplicates_questions.append(word)
        result_questions.append(unique)
    unique_questions_words = result_questions

    unique_words = unique_phrases_words + unique_questions_words

    print(unique_words)
    exit()
    '''
    '''print(len(unique_phrases_words)) #numero frasi

    l = 0
    for i in range(len(unique_words)): #numero parole totali
        for el in unique_words[i]:
            l = l+1'''

    corpus = Corpus()
    corpus.fit(all_words, window=10)
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')
    #my_glove = glove.load('/home/mary/PycharmProjects/kebdi/glove.model')

    with open(glove_path, "w") as f:
        for word in glove.dictionary:
            f.write(word)
            f.write(" ")
            for i in range(0, 300):
                f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
                f.write(" ")
            f.write("\n")


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


#glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove.6B.300d.txt")
if not os.path.exists(glove_path):
    create_glove()

glove = create_dict(glove_path)

gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else 1e-5*np.ones(300) for word in x]


def elaborate(padding=100, custom_data=None):
    data = custom_data if custom_data is not None else elaborate_data(qa_json_path, plots_path)
    np.seterr('raise')

    train_phrases = []
    train_questions = []
    train_answers = []
    for i in range(len(data)):
        '''#plot = plot.replace("\n", '')
        #phrases_tmp = plot.split('.')
        phrases_tmp = data[i][0].split('\n')
        #phrases = [p for p in phrases_tmp if p is not '']
        phrases = [p for p in phrases_tmp if p is not ' ' and len(p) is not 0]
        #phrases = [p for p in phrases if len(p) is not 0]'''

        plot = data[i][0].replace("\n", '')
        phrases_tmp = plot.split('.')
        phrasesa = [p for p in phrases_tmp if (len(p) is not 0)]
        phrases = [p for p in phrasesa if (len(p) is not 1)]

        phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases)) #list(map(lambda x: lemmatize_words(clean_split_string(x)), phrases))
        question_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][1]))) #lemmatize_words(clean_split_string(question))
        answer_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][2]))) #lemmatize_words(clean_split_string(answer))

        question_vect = gloveize_phrase(question_words)
        answer_vect = gloveize_phrase(answer_words)
        phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))

        #if len(answer_vect) is not 0 and len(question_vect) is not 0 and len(el) is not 0:
        #MODIFICARE CON POWER MEAN
        question_emb = np.mean(question_vect, axis=0)
        answer_emb = np.mean(answer_vect, axis=0)
        phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))

        dcos_all = []
        for k in range(len(phrases_emb)):
            dcos = spatial.distance.cosine(answer_emb, phrases_emb[k])
            dcos_all.append(dcos)

        imin = np.argmin(dcos_all)

        emb_size = phrases_emb[0].shape[0]
        plots = np.zeros((padding, emb_size))
        for j in range(len(phrases_emb)):
            plots[j] = phrases_emb[j]

        phrases_emb = plots

        train_answers.append(imin)
        train_phrases.append(phrases_emb)
        train_questions.append(question_emb)

    train_phrases = np.asarray(train_phrases)
    train_questions = np.asarray(train_questions)
    train_answers = np.asarray(train_answers)

    if custom_data is not None:
        return [train_phrases, train_questions, train_answers]
    else:
        np.save('n_train_phrases.npy', train_phrases)
        np.save('n_train_questions.npy', train_questions)
        np.save('n_train_answers.npy', train_answers)



elaborate()


def load_dataset():
    train_phrases = np.load('n_train_phrases.npy')
    train_questions = np.load('n_train_questions.npy')
    train_answers = np.load('n_train_answers.npy')

    return [train_phrases, train_questions, train_answers]



