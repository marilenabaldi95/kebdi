from keras.models import Model
from keras.layers import Input, Flatten, Dense, Multiply, Concatenate, Subtract


def get_model(embedding_size, phrases_size, m=100, k=100):
    context_input = Input([phrases_size, embedding_size], name='context_input') #input trame [100x300]
    query_input = Input([1, embedding_size], name='query_input') #input domanda [1x300]

    context_flatten = Flatten(name='context_flatten')(context_input) #ridimensionamento trame
    query_flatten = Flatten(name='query_flatten')(query_input) #ridimensionamento domande

    context_embedding = Dense(m, name='context_embedding')(context_flatten) #fully connected trame
    query_embedding = Dense(m, name='query_embedding')(query_flatten) #fully connected domande

    flatten = Multiply()([context_embedding, query_embedding]) #moltiplicazione, al fine di correlare le informazioni e "enfatizzare" il vettore della frase corretta

    embedding = Dense(k, name='embedding')(flatten) #fully connected

    output = Dense(phrases_size, activation='softmax', name='probability_distribution')(embedding) #livello per l'uscita, col vettore softmax

    model = Model([context_input, query_input], output) #definizione livelli input e output per il modello

    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #definizione parametri come algoritmo di ottimizzazione, funzione di costo e metrica per la valutazione del modello

    return model
