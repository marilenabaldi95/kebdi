import numpy as np
from numpy.linalg import norm
import pickle
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt


#funzione che restituisce la norma euclidea
def euclidean_distance(a, b):
	return norm(a-b)


#funziona che restituisce i k vicini data la lista di frasi e la domanda da classificare
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


#funzione che si occupa di caricare le trame, domande, risposte elaborate nel dataPreprocessingKnn.py
def load_data():
	with open("data.txt", "rb") as fp:
		return pickle.load(fp)


#funzione che realizza l'esecuzione dell'knn e la valutazione dell'accuratezza
def main():
	[phrases_emb_list, question_emb_list, answer_emb_list, dizionario_list] = load_data()

	risposta_list = []
	num_corrette = 0
	num_sbagliate = 0
	num_totali = 0
	for i in range(len(phrases_emb_list)):

		risposta_list.append(get_neighbors(phrases_emb_list[i], question_emb_list[i], 1))

		salta = int(len(dizionario_list[i])/2)
		for j in range(int(len(dizionario_list[i]))):
			if np.array_equal(risposta_list[i][0], dizionario_list[i][j]):
				#frase = dizionario_list[i][j+salta]
				#print(frase)

				if np.array_equal(dizionario_list[i][j], answer_emb_list[i]):
					num_corrette = num_corrette + 1
					num_totali = num_totali + 1
				else:
					num_sbagliate = num_sbagliate + 1
					num_totali = num_totali + 1

	print("CORRETTE: ", num_corrette)
	print("SBAGLIATE: ", num_sbagliate)
	print("TOTALI: ", num_totali)
	print("accuracy: ", num_corrette/num_totali)


if __name__ == "__main__":
	main()


#di seguito è commentato il codice utilizzato per rappresentare nello spazio un esempio di esecuzione dell'knn su una trama (come si vede nelle slide)

'''
vicini = get_neighbors(phrases_emb_list[0][0], question_emb_list[0], len(phrases_emb_list[0]) - 1)
pca = PCA(n_components=3)
i=0
matrice = np.zeros((len(phrases_emb_list[0])+2, len(phrases_emb_list[0][0])))
for vet in vicini:
	matrice[i] = vet
	i = i+1

matrice[len(phrases_emb_list[0])] = phrases_emb_list[0]
matrice[len(phrases_emb_list[0])+1] = phrases_emb_list[0][0]
#print(matrice)
reduced = pca.fit_transform(matrice)

#print(reduced)
t = reduced.transpose()
#print(t)
#print(t)
#print(t[0], t[1])

fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')
ax.scatter3D(t[0][:len(phrases_emb)], t[1][:len(phrases_emb)], t[2][:len(phrases_emb)], c='cyan')
ax.scatter3D(t[0][len(phrases_emb)], t[1][len(phrases_emb)], t[2][len(phrases_emb)], c='k')
ax.scatter3D(t[0][len(phrases_emb)+1], t[1][len(phrases_emb)+1], t[2][len(phrases_emb)+1], c='k')

plt.show()

salta = int(len(dizionario)/2)

for i in range(int(len(dizionario))):
	if np.array_equal(risposta[0], dizionario[i]):
		print("entrato")
		frase = dizionario[i+salta]
		print(frase)

'''