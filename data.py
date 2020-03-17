import json
import glob
import os
import numpy as np
from dataPreprocessing import elaborate

qa_json_path = '/home/mary/Scrivania/qa.json'
plots_path = '/home/mary/PycharmProjects/kebdi/data/plot'


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


def main():
    data = elaborate_data(qa_json_path, plots_path)

    train_phrases = []
    train_questions = []
    train_answers = []
    np.seterr('raise')
    for i in range(len(data)):
        phrases_emb, question_emb, imin = elaborate(data[i][0], data[i][1], data[i][2])

        train_answers.append(imin)
        train_phrases.append(phrases_emb)
        train_questions.append(question_emb)

    train_phrases = np.asarray(train_phrases)
    train_questions = np.asarray(train_questions)
    train_answers = np.asarray(train_answers)

    np.save('n_train_phrases.npy', train_phrases)
    np.save('n_train_questions.npy', train_questions)
    np.save('n_train_answers.npy', train_answers)

    ''' 
        m = 0
        for d in data:
            m = d[0].count('.') if d[0].count('.') > m else m
    
        print(m) #nella trama più lunga ci sono 94 punti => almeno 94 frasi => dimensione embedding fissata a 100 - padding a 100
    '''


def load_dataset():
    train_phrases = np.load('n_train_phrases.npy')
    train_questions = np.load('n_train_questions.npy')
    train_answers = np.load('n_train_answers.npy')

    return [train_phrases, train_questions, train_answers]


if __name__ == '__main__':
    main()