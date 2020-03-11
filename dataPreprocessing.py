import numpy as np
import progressbar
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import spatial

plot = '''Gary Hook, a new recruit to the British Army, takes leave of his much younger brother Darren. Hook's squad of British soldiers is sent to Belfast in 1971 in the early years of The Troubles. Under the leadership of the inexperienced Lieutenant Armitage, his squad goes to a volatile area of Belfast where Catholic Nationalists and Protestant Loyalists live side by side. The unit provides support for the Royal Ulster Constabulary as it inspects homes for firearms, shocking Hook with their rough treatment of women and children. The Catholic neighbourhood has been alerted to the activity and a crowd gathers to protest and provoke the British troops who, though heavily armed, can only respond by trying to hold the crowd back.
One soldier leaves his gun on the ground in the confusion and a young boy runs off through the mob with it; Hook and another pursue him. As the crowd's protest escalates into rock-throwing, the soldiers and police pull out, leaving the two soldiers behind. Hook and the other soldier are briefly rescued by a sympathetic woman who fails to hold back a small crowd who are beating them. Hook sees the other soldier shot dead at point blank range by the young Nationalist Paul Haggerty and then, with the crowd physically engaging Haggerty, flees through streets and back alleys, finally eluding his pursuers and hiding until dark.
A Protestant youngster brings Hook to a local pub that serves as a front for the Loyalists, where he glimpses a Loyalist group in a back room constructing a bomb under the guidance of the Military Reaction Force (MRF), the covert counter-insurgency unit of the British Army. Hook steps outside the pub just before an enormous explosion destroys the building. Hook flees once more into the dark streets. Unaware that the Loyalists have blown themselves up, each of the two IRA factions charges the other with responsibility for the bombing.
Two Catholics, Eamon and his daughter Brigid, discover Hook as he lies unconscious in the street. They take him to their apartment and, even though they discover he is a British soldier, Eamon stitches his wounds. Eamon contacts Boyle, a senior IRA official, for help, expecting a more humane solution than the younger IRA faction would allow. Boyle, less radical and violent than younger IRA members, has a working relationship with the MRF. He tells MRF Captain Browning, leader of the British MRF section, of Hook's whereabouts and asks in return that Browning assassinate James Quinn, a key leader of the younger IRA faction.
Quinn and his IRA squad have been tailing Boyle since the pub explosion and saw him visit Eamon's apartment without knowing why he was there. Sensing danger, Hook flees the apartment, taking with him a sheathed blade. Moving painfully through the apartment complex halls and stairways, he eludes the IRA men who have separated to search for him. Finally, unable to get away from Haggerty, who is about to come around a corner and discover him, Hook stabs him. As the wounded man lies dying, Hook reaches down and grasps his shoulder, sharing strength and sympathy as they hold each other's gaze and the IRA man dies.
Hook is captured by Quinn's group and taken to a hideout. Quinn orders Sean, a young teen whom Quinn has recruited, to execute Hook. When Sean hesitates, Quinn prepares to execute Hook. Browning's group arrives, and Lewis, to Hook's horror, shoots Sean. Lewis attempts to strangle Hook to prevent him from informing others of the bomb. As Lieutenant Armitage and his men enter in support of Browning, Armitage sees Lewis' attempt to kill Hook. Sean raises himself and fires at Lewis before being shot by Armitage. Browning finds Quinn and, rather than arrest him, tells him Boyle wants him dead. He promises to contact him soon, telling him he expects him to prove to be reasonable.
Hook returns to his barracks. Later, despite a formal complaint by Armitage, the officer in charge dismisses the incident between Hook, Lewis, and Sean as a confused situation that merits no further inquiry. Hook returns to England and reunites with Darren.'''

question = "Why does Hook leave Eamon's apartment?"

answer = "Because he senses danger."


# "Because he is warned it's not safe.",
# "Because he wants to get some fresh air.",
# "Because he is called away.",
# "Because he has to get some stuff.",
# "Because he senses danger."

def clean_split_string(phrase):
    stop_chars = '|\\'.join(list(string.punctuation)) + ''
    m_regex = r'(' + stop_chars + ')*(\w+)(' + stop_chars + ')*'
    sub_regex = r'\2'
    result = re.sub(m_regex, sub_regex, phrase)
    return result.split()


def create_dict(file):
    bar = progressbar.ProgressBar(max_value=400000, redirect_stdout=True)
    i = 0

    embeddings_dict = {}
    with open(file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float)
            embeddings_dict[word] = vector
            bar.update(i)
            i += 1

    bar.finish()
    return embeddings_dict

glove = create_dict("/home/mary/PycharmProjects/kebdi/glove.6B.300d.txt")
stop_words = list(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

remove_stop_words = lambda line: [word for word in line if word not in stop_words]

lemmatize_words = lambda line: [wordnet_lemmatizer.lemmatize(word) for word in line]

gloveize_phrase = lambda x: [glove[word] if word in glove.keys() else np.zeros(300) for word in x]


def main():
    in_questions_train = []
    in_phrases_train = []
    out_answer_train = []



def elaborate(plot, question, answer, padding=100):
    phrases = plot.split('.')

    phrases_words = list(map(lambda x: lemmatize_words(remove_stop_words(clean_split_string(x))), phrases))
    question_words = lemmatize_words(remove_stop_words(clean_split_string(question)))
    answer_words = lemmatize_words(remove_stop_words(clean_split_string(answer)))

    question_vect = gloveize_phrase(question_words)
    answer_vect = gloveize_phrase(answer_words)
    phrases_vect = list(map(lambda x: gloveize_phrase(x), phrases_words))

    question_emb = np.mean(question_vect, axis=0)
    answer_emb = np.mean(answer_vect, axis=0)
    phrases_emb = list(map(lambda x: np.mean(x, axis=0), phrases_vect))

    dcos_all = []
    for i in range(len(phrases_emb) - 1):
        dcos = spatial.distance.cosine(answer_emb, phrases_emb[i])
        dcos_all.append(dcos)

    imin = np.argmin(dcos_all)

    emb_size = phrases_emb[0].shape[0]
    plots = np.zeros((padding, emb_size))
    for i in range(len(phrases_emb)):
        plots[i] = phrases_emb[i]

    phrases_emb = plots

    #print(question)
    #print(answer)
    #print(phrases[imin])

    return [phrases_emb, question_emb, imin]


if __name__ == '__main__':
    main()