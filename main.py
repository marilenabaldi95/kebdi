from model import get_model
from dataPreprocessing import load_dataset
import numpy as np
from tensorflow.keras.models import load_model

model_path = '/home/mary/PycharmProjects/kebdi/models/model'

batch_size = 32
epochs = 30


def test(model_path, test_phrases, test_questions, test_answers):
    model = load_model(model_path)

    prediction = model.predict([test_phrases, test_questions])

    scores = model.evaluate([test_phrases, test_questions], test_answers)
    print('TEST output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    for i in range(len(test_answers)):
        print('RESULT {} PREDICT {}'.format(test_answers[i], np.argmax(prediction[i])))


def train(train_phrases, train_questions, train_answers):
    model = get_model(300, 100, m=100, k=100)
    model.summary()

    model.fit([train_phrases, train_questions], train_answers, batch_size=batch_size, epochs=epochs)

    scores = model.evaluate([train_phrases, train_questions], train_answers)
    print('TRAINING output_loss: {} - output_acc: {}'.format(scores[0], scores[1]))

    model.save(model_path)


def main():
    phrases, questions, answers = load_dataset()
    questions = np.reshape(questions, (questions.shape[0], 1, questions.shape[1]))
    phrases = np.reshape(phrases, (phrases.shape[0], phrases.shape[1], phrases.shape[2]))

    d_slice = int((phrases.shape[0]) * 3 / 4)
    #print(d_slice)
    #print("TOT ", (phrases.shape[0]))

    train_phrases = phrases[:d_slice]
    train_questions = questions[:d_slice]
    train_answers = answers[:d_slice]

    #test_phrases = phrases[d_slice:]
    test_phrases = phrases[d_slice-80:d_slice+190]
    #test_questions = questions[d_slice:]
    test_questions = questions[d_slice-80:d_slice+190]
    #test_answers = answers[d_slice:]
    test_answers = answers[d_slice-80:d_slice+190]

    train(train_phrases, train_questions, train_answers)
    test(model_path, test_phrases, test_questions, test_answers)


if __name__ == '__main__':
    main()