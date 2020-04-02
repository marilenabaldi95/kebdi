from model import get_model
from dataPreprocessing import load_trainset, load_testset
import numpy as np
from tensorflow.keras.models import load_model

model_path = '/home/mary/PycharmProjects/kebdi/models/model'

batch_size = 32
epochs = 30


#funzione che fa il test della rete. Si passano il modello della rete, le trame, domande, indici_risposte del test set
def test(model_path, test_phrases, test_questions, test_answers):
    model = load_model(model_path)

    prediction = model.predict([test_phrases, test_questions]) #funzione per fare il test

    scores = model.evaluate([test_phrases, test_questions], test_answers) #funzione per ottenere i valori di loss e accuracy a fine test
    print('TEST output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    for i in range(len(test_answers)):
        print('RESULT {} PREDICT {}'.format(test_answers[i], np.argmax(prediction[i])))


#funzione che fa il training della rete. Si passano trame, domande e indici risposte del training set
def train(train_phrases, train_questions, train_answers):
    model = get_model(300, 100, m=100, k=100)
    model.summary()

    model.fit([train_phrases, train_questions], train_answers, batch_size=batch_size, epochs=epochs) #funzione per fare il training (quindi diamo anche l'output)

    scores = model.evaluate([train_phrases, train_questions], train_answers) #funzione per ottenere i valori di loss e accuracy a fine training
    print('TRAINING output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    model.save(model_path) #salvataggio del modello, in modo da non doverlo rielaborare per il test


def main():
    train_phrases, train_questions, train_answers = load_trainset()
    train_questions = np.reshape(train_questions, (train_questions.shape[0], 1, train_questions.shape[1]))
    train_phrases = np.reshape(train_phrases, (train_phrases.shape[0], train_phrases.shape[1], train_phrases.shape[2]))

    test_phrases, test_questions, test_answers = load_testset()
    test_questions = np.reshape(test_questions, (test_questions.shape[0], 1, test_questions.shape[1]))
    test_phrases = np.reshape(test_phrases, (test_phrases.shape[0], test_phrases.shape[1], test_phrases.shape[2]))

    train(train_phrases, train_questions, train_answers)
    test(model_path, test_phrases, test_questions, test_answers)


if __name__ == '__main__':
    main()