from keras.models import Model
from keras.layers import Input, Flatten, Dense, Multiply, Dot, Lambda


#funzione che si occupa della creazione del modello della rete
def get_model(embedding_size, phrases_size, k=100):

    #livelli in input della rete - il primo è l'ingresso delle trame -> ogni trama è [100,300] - ogni domande è [1,300]
    context_input = Input([phrases_size, embedding_size], name='context_input')
    query_input = Input([1, embedding_size], name='query_input')

    #mul = Multiply()([context_input, query_input])

    #livello di merge per gli input di trame e domande, che vengono uniti col prodotto scalare lungo la dimensione 2 (la dimensione dell'embedding dei dati, 300)
    mul = Dot(axes=[2, 2], normalize=True)([context_input, query_input])

    #livello per il ridimensionamento dei dati (vengono "schiacciati" su una sola dimensione), affinché si possa utilizzare il fully connected successivo
    flatten = Flatten()(mul)

    #livello di "embedding" fully connected
    embedding = Dense(k, name='embedding')(flatten)

    #livello di output fully connected con funzione di attivazione softmax
    output = Dense(phrases_size, activation='softmax', name='output')(embedding)

    #definizione del modello, quindi indicazione dei livelli di input e output
    model = Model([context_input, query_input], output)

    #selezione di metrica, funzione di costo e ottimizzatore per il modello realizzato
    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
