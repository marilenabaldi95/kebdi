from model import get_model
from data import load_dataset
import numpy as np

batch_size = 32
epochs = 25

train_phrases, train_questions, train_answers = load_dataset()

model = get_model(300, 100)
model.summary()

print(train_phrases.shape)
print(train_questions.shape)
print(train_answers.shape)

train_questions = np.reshape(train_questions, (3784, 1, 300))


print(train_phrases.shape)
print(train_questions.shape)
print(train_answers.shape)

model.fit([train_phrases, train_questions], train_answers, batch_size=batch_size, epochs=epochs)
model.save('/home/mary/PycharmProjects/kebdi/models/model')

model.summary()

scores = model.evaluate([train_phrases, train_questions], train_answers)
