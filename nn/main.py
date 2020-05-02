from nn.model import get_model
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'models/model'
data_path = 'data'

batch_size = 16
epochs = 250
k = 100

TRAIN = False #cambiare in False per testare


#funziona che si occupa di effettuare il testing della rete (sul test set)
def test(path_model, test_phrases, test_questions, test_answers):
    model = load_model(path_model)

    prediction = model.predict([test_phrases, test_questions])

    scores = model.evaluate([test_phrases, test_questions], test_answers)
    print('TEST output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    '''for i in range(len(test_answers)):
        print('RESULT {} PREDICT {}'.format(test_answers[i], np.argmax(prediction[i])))'''


#funziona che si occupa di effettuare il training della rete (sul training set)
def train(train_phrases, train_questions, train_answers):
    model = get_model(300, 100, k=k)
    model.summary()

    model.fit([train_phrases, train_questions], train_answers, batch_size=batch_size, epochs=epochs)

    scores = model.evaluate([train_phrases, train_questions], train_answers)
    print('TRAINING output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    model.save(model_path)


#funziona che si occupa di caricare training e test set per richiamare le funzioni o di train o di test
def main():
    if TRAIN is True:
        train_phrases, train_questions, train_answers = load_trainset()
        train_questions = np.reshape(train_questions, (train_questions.shape[0], 1, train_questions.shape[1]))
        train_phrases = np.reshape(train_phrases, (train_phrases.shape[0], train_phrases.shape[1], train_phrases.shape[2]))

        train(train_phrases, train_questions, train_answers)
    else:
        test_phrases, test_questions, test_answers = load_testset()
        test_questions = np.reshape(test_questions, (test_questions.shape[0], 1, test_questions.shape[1]))
        test_phrases = np.reshape(test_phrases, (test_phrases.shape[0], test_phrases.shape[1], test_phrases.shape[2]))

        test(model_path, test_phrases, test_questions, test_answers)


#funzione che si occupa di caricare i file numpy del training set generati dal dataPreprocessing
def load_trainset():
    train_phrases = np.load(data_path + '/train_phrases.npy')
    train_questions = np.load(data_path + '/train_questions.npy')
    train_answers = np.load(data_path + '/train_answers.npy')

    return [train_phrases, train_questions, train_answers]


#funzione che si occupa di caricare i file numpy del test set generati dal dataPreprocessing
def load_testset():
    train_phrases = np.load(data_path + '/test_phrases.npy')
    train_questions = np.load(data_path + '/test_questions.npy')
    train_answers = np.load(data_path + '/test_answers.npy')

    return [train_phrases, train_questions, train_answers]


if __name__ == '__main__':
    main()