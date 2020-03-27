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

glove_path = 'data/glove/glove_dataset_300d.txt'
glove_model_path = 'models'
#qa_json_path = 'data/qa/qa.json'
qa_json_path = '/home/mary/PycharmProjects/kebdi/data/qa/newqa.json'
plots_path = 'data/plot'
data_path = 'data'


wordnet_lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

#funzione che si occupare di rimuovere le stop-words da una frase
remove_stop_words = lambda line: [word for word in line if word not in stop_words]

#funziona che si occuèa di lemmatizzare ogni parola di una frase
lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]


#funzione che si occupa di "ripulire", una per una, le frasi che le vengono passate: vengono rimossi tutti i simboli prima e dopo ogni parola; poi le frasi vengono splittate in parole
#in output viene restituita la frase come lista di parole
def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()


#funzione che si occupa di creare un dizionario python a partire dal file glove (quello con struttura parola-vettore ogni riga
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


#funzione che si occupa di addestrare il glove: tutto il corpus è riportato in forma di lista di frasi, che a loro volta sono liste di parole
#glove crea un modello a partire da questi dati e salva un file nella forma parola-vettore, in moodo da poterne salvare facilmente il dizionario con la funzione create_dict
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

    corpus = Corpus()
    corpus.fit(all_words, window=10)
    glove = Glove(no_components=300, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save(glove_model_path + '/glove.model')
    #my_glove = glove.load(glove_model_path + '/glove_model')

    with open(glove_path, "w") as f:
        for word in glove.dictionary:
            f.write(word)
            f.write(" ")
            for i in range(0, 300):
                f.write(str(glove.word_vectors[glove.dictionary[word]][i]))
                f.write(" ")
            f.write("\n")

    #print("Dimensione corpus glove: ", len(glove.dictionary))


#funzione che legge i plot dai file. Viene passato il path dei plot e viene restituito un dizionario nella forma id_trama, trama
def read_plots(path_plot):
    data_dict = {}

    for file in glob.glob(path_plot + "/*"):
        with open(file) as f:
            key = os.path.basename(file).replace('.wiki', '')
            value = ' '.join(f.readlines())
            data_dict[key] = value

    return data_dict


#funziona che, a partire dal json, facendo uso della funzione read_plots, crea un dizionario nella forma trama, domanda, risposta corretta
def elaborate_data(file_json, path_plot):
    with open(file_json) as f:
        qa_dict = json.load(f)

    plot_dict = read_plots(path_plot)

    def prep_data(el):
        return [plot_dict[el['imdb_key']], el['question'], el['answers'][el['correct_index']]]

    data = [prep_data(el) for el in qa_dict if el['correct_index'] is not None and el['imdb_key'] in plot_dict.keys()]

    return data


#creazione dizionario a partire da glove o pretrainato (primo) o trainato da noi (secondo)
glove = create_dict("/home/mary/PycharmProjects/kebdi/data/glove/glove.6B.300d.txt")
'''if not os.path.exists(glove_path + '/glove.model'):
    create_glove()

glove = create_dict(glove_path)'''


#funzione che si occupa di associare ad ogni parola della frase, il corrispondente vettore di glove
gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else 1e-5*np.ones(300) for word in x]


#funzione che si occupa di effettuare le operazioni di pulizia, rimozione stopwords e lemmatizzazione su tutte le trame, domande, risposte
#effettua anche la media dei vettori parole per rappresentare la frase
#fa anche l'associazione tra la risposta di qamovie e la frase della trama che contiene la risposta
#trova l'indice della risposta corretta all'interno della trama
#fa anche il padding delle frasi della trama
#salva i vettori risultanti (trame, domande, risposte) in file .npy
def elaborate(padding=100, custom_data=None):
    data = custom_data if custom_data is not None else elaborate_data(qa_json_path, plots_path)
    np.seterr('raise')

    train_phrases = []
    train_questions = []
    train_answers = []
    for i in range(len(data)):
        plot = data[i][0].replace("\n", '')
        phrases_tmp = plot.split('.')
        phrasesa = [p for p in phrases_tmp if (len(p) is not 0)]
        phrases = [p for p in phrasesa if (len(p) is not 1)]

        phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
        question_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][1])))
        answer_words = lemmatize_words(remove_stop_words(clean_split_string(data[i][2])))

        question_vect = gloveize_phrase(question_words)
        answer_vect = gloveize_phrase(answer_words)
        phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))

        question_emb = np.mean(question_vect, axis=0)
        answer_emb = np.mean(answer_vect, axis=0)
        phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))

        '''q_weights = np.array([freq_dict[word] for word in question_words])
        a_weights = np.array([freq_dict[word] for word in answer_words])
        #p_weights =

        question_emb = np.average(np.array(question_vect), axis=0, weights=q_weights)
        answer_emb = np.average(np.array(answer_vect), axis=0, weights=a_weights)


        print(question_emb)
        exit()'''

        dcos_all = []
        for k in range(len(phrases_emb)):
            dcos = spatial.distance.cosine(answer_emb, phrases_emb[k])
            dcos_all.append(dcos)

        imin = np.argmin(dcos_all)
        #phrasemin = phrases_emb[imin]

        emb_size = phrases_emb[0].shape[0]
        plots = np.zeros((padding, emb_size))
        for j in range(len(phrases_emb)):
            plots[j] = phrases_emb[j]

        phrases_emb = plots

        train_answers.append(imin)
        #train_answers.append(phrasemin)
        train_phrases.append(phrases_emb)
        train_questions.append(question_emb)

    train_phrases = np.asarray(train_phrases)
    train_questions = np.asarray(train_questions)
    train_answers = np.asarray(train_answers)

    if custom_data is not None:
        return [train_phrases, train_questions, train_answers]
    else:
        np.save(data_path + '/glove_phrases.npy', train_phrases)
        np.save(data_path + '/glove_questions.npy', train_questions)
        np.save(data_path + '/glove_answers.npy', train_answers)


#funzione che carica i file numpy, in modo da non dover sempre riprocessare tutti i dati
def load_dataset():
    train_phrases = np.load(data_path + '/glove_phrases.npy')
    train_questions = np.load(data_path + '/glove_questions.npy')
    train_answers = np.load(data_path + '/glove_answers.npy')

    return [train_phrases, train_questions, train_answers]


if __name__ == '__main__':
    elaborate()
