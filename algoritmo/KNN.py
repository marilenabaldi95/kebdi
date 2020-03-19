# Example of making predictions
#from math import sqrt
import gensim as gensim
import numpy as np
from dataPreprocessing import elaborate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


from mpl_toolkits import mplot3d
#%matplotlib inline


from numpy.linalg import norm
# calculate the Euclidean distance between two vectors
def euclidean_distance(a, b):
	'''distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])^2'''

	#return sqrt(distance)
	return norm(a-b)

# Locate the most similar neighbors
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


# Test distance function
plot = '''Gary Hook, a new recruit to the British Army, takes leave of his much younger brother Darren. Hook's squad of British soldiers is sent to Belfast in 1971 in the early years of The Troubles. Under the leadership of the inexperienced Lieutenant Armitage, his squad goes to a volatile area of Belfast where Catholic Nationalists and Protestant Loyalists live side by side. The unit provides support for the Royal Ulster Constabulary as it inspects homes for firearms, shocking Hook with their rough treatment of women and children. The Catholic neighbourhood has been alerted to the activity and a crowd gathers to protest and provoke the British troops who, though heavily armed, can only respond by trying to hold the crowd back.
One soldier leaves his gun on the ground in the confusion and a young boy runs off through the mob with it; Hook and another pursue him. As the crowd's protest escalates into rock-throwing, the soldiers and police pull out, leaving the two soldiers behind. Hook and the other soldier are briefly rescued by a sympathetic woman who fails to hold back a small crowd who are beating them. Hook sees the other soldier shot dead at point blank range by the young Nationalist Paul Haggerty and then, with the crowd physically engaging Haggerty, flees through streets and back alleys, finally eluding his pursuers and hiding until dark.
A Protestant youngster brings Hook to a local pub that serves as a front for the Loyalists, where he glimpses a Loyalist group in a back room constructing a bomb under the guidance of the Military Reaction Force (MRF), the covert counter-insurgency unit of the British Army. Hook steps outside the pub just before an enormous explosion destroys the building. Hook flees once more into the dark streets. Unaware that the Loyalists have blown themselves up, each of the two IRA factions charges the other with responsibility for the bombing.
Two Catholics, Eamon and his daughter Brigid, discover Hook as he lies unconscious in the street. They take him to their apartment and, even though they discover he is a British soldier, Eamon stitches his wounds. Eamon contacts Boyle, a senior IRA official, for help, expecting a more humane solution than the younger IRA faction would allow. Boyle, less radical and violent than younger IRA members, has a working relationship with the MRF. He tells MRF Captain Browning, leader of the British MRF section, of Hook's whereabouts and asks in return that Browning assassinate James Quinn, a key leader of the younger IRA faction.
Quinn and his IRA squad have been tailing Boyle since the pub explosion and saw him visit Eamon's apartment without knowing why he was there. Sensing danger, Hook flees the apartment, taking with him a sheathed blade. Moving painfully through the apartment complex halls and stairways, he eludes the IRA men who have separated to search for him. Finally, unable to get away from Haggerty, who is about to come around a corner and discover him, Hook stabs him. As the wounded man lies dying, Hook reaches down and grasps his shoulder, sharing strength and sympathy as they hold each other's gaze and the IRA man dies.
Hook is captured by Quinn's group and taken to a hideout. Quinn orders Sean, a young teen whom Quinn has recruited, to execute Hook. When Sean hesitates, Quinn prepares to execute Hook. Browning's group arrives, and Lewis, to Hook's horror, shoots Sean. Lewis attempts to strangle Hook to prevent him from informing others of the bomb. As Lieutenant Armitage and his men enter in support of Browning, Armitage sees Lewis' attempt to kill Hook. Sean raises himself and fires at Lewis before being shot by Armitage. Browning finds Quinn and, rather than arrest him, tells him Boyle wants him dead. He promises to contact him soon, telling him he expects him to prove to be reasonable.
Hook returns to his barracks. Later, despite a formal complaint by Armitage, the officer in charge dismisses the incident between Hook, Lewis, and Sean as a confused situation that merits no further inquiry. Hook returns to England and reunites with Darren.'''

question = "Why does Hook leave Eamon's apartment?"

[phrases_emb, question_emb, dizionario] = elaborate(plot, question)
vicini = get_neighbors(phrases_emb, question_emb, len(phrases_emb)-1)
risposta = get_neighbors(phrases_emb, question_emb, 1)


salta = int(len(dizionario)/2)
for i in range(int(len(dizionario))):
	if np.array_equal(risposta[0], dizionario[i]):
		print("entrato")
		frase = dizionario[i+salta]
		print(frase)

pca = PCA(n_components=3)
i=0
matrice = np.zeros((len(phrases_emb)+2, len(phrases_emb[0])))
for vet in vicini:
	matrice[i] = vet
	i = i+1

matrice[len(phrases_emb)] = question_emb
matrice[len(phrases_emb)+1] = risposta[0]
#print(matrice)
reduced = pca.fit_transform(matrice)

#print(reduced)
t = reduced.transpose()
#print(t)
#print(t)
#print(t[0], t[1])
'''plt.scatter3D(t[0], t[1], t[2])
plt.scatter3D(t[0][len(phrases_emb)], t[1][len(phrases_emb)], t[2][len(phrases_emb)], edgecolors='g')
plt.scatter3D(t[0][len(phrases_emb)+1], t[1][len(phrases_emb)+1], t[2][len(phrases_emb)+1], edgecolors='r')
plt.show()
'''

fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')
ax.scatter3D(t[0][:len(phrases_emb)], t[1][:len(phrases_emb)], t[2][:len(phrases_emb)], c='cyan')
ax.scatter3D(t[0][len(phrases_emb)], t[1][len(phrases_emb)], t[2][len(phrases_emb)], c='k')
ax.scatter3D(t[0][len(phrases_emb)+1], t[1][len(phrases_emb)+1], t[2][len(phrases_emb)+1], c='k')

plt.show()
'''
salta = int(len(dizionario)/2)

for i in range(int(len(dizionario))):
	if np.array_equal(risposta[0], dizionario[i]):
		print("entrato")
		frase = dizionario[i+salta]
		print(frase)

'''