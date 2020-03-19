from keras.models import Model
from keras.layers import Input, Flatten, Dense, Multiply, Concatenate, Subtract


def get_model(embedding_size, phrases_size, m=100, k=100):
    context_input = Input([phrases_size, embedding_size], name='context_input')
    query_input = Input([1, embedding_size], name='query_input')

    context_flatten = Flatten(name='context_flatten')(context_input)
    query_flatten = Flatten(name='query_flatten')(query_input)

    context_embedding = Dense(m, name='context_embedding')(context_flatten)
    query_embedding = Dense(m, name='query_embedding')(query_flatten)

    flatten = Multiply()([context_embedding, query_embedding])

    embedding = Dense(k, name='embedding')(flatten)

    output = Dense(phrases_size, activation='softmax', name='probability_distribution')(embedding)

    model = Model([context_input, query_input], output)

    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
