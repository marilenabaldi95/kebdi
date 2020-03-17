from model import get_model
from dataPreprocessing1 import load_dataset, elaborate
import numpy as np
from tensorflow.keras.models import load_model

model_path = '/home/mary/PycharmProjects/kebdi/models/modelp'

batch_size = 32
epochs = 30


def test(model_path):
    model = load_model(model_path)

    plot = '''Gary Hook, a new recruit to the British Army, takes leave of his much younger brother Darren. Hook's squad of British soldiers is sent to Belfast in 1971 in the early years of The Troubles. Under the leadership of the inexperienced Lieutenant Armitage, his squad goes to a volatile area of Belfast where Catholic Nationalists and Protestant Loyalists live side by side. The unit provides support for the Royal Ulster Constabulary as it inspects homes for firearms, shocking Hook with their rough treatment of women and children. The Catholic neighbourhood has been alerted to the activity and a crowd gathers to protest and provoke the British troops who, though heavily armed, can only respond by trying to hold the crowd back.
    One soldier leaves his gun on the ground in the confusion and a young boy runs off through the mob with it; Hook and another pursue him. As the crowd's protest escalates into rock-throwing, the soldiers and police pull out, leaving the two soldiers behind. Hook and the other soldier are briefly rescued by a sympathetic woman who fails to hold back a small crowd who are beating them. Hook sees the other soldier shot dead at point blank range by the young Nationalist Paul Haggerty and then, with the crowd physically engaging Haggerty, flees through streets and back alleys, finally eluding his pursuers and hiding until dark.
    A Protestant youngster brings Hook to a local pub that serves as a front for the Loyalists, where he glimpses a Loyalist group in a back room constructing a bomb under the guidance of the Military Reaction Force (MRF), the covert counter-insurgency unit of the British Army. Hook steps outside the pub just before an enormous explosion destroys the building. Hook flees once more into the dark streets. Unaware that the Loyalists have blown themselves up, each of the two IRA factions charges the other with responsibility for the bombing.
    Two Catholics, Eamon and his daughter Brigid, discover Hook as he lies unconscious in the street. They take him to their apartment and, even though they discover he is a British soldier, Eamon stitches his wounds. Eamon contacts Boyle, a senior IRA official, for help, expecting a more humane solution than the younger IRA faction would allow. Boyle, less radical and violent than younger IRA members, has a working relationship with the MRF. He tells MRF Captain Browning, leader of the British MRF section, of Hook's whereabouts and asks in return that Browning assassinate James Quinn, a key leader of the younger IRA faction.
    Quinn and his IRA squad have been tailing Boyle since the pub explosion and saw him visit Eamon's apartment without knowing why he was there. Sensing danger, Hook flees the apartment, taking with him a sheathed blade. Moving painfully through the apartment complex halls and stairways, he eludes the IRA men who have separated to search for him. Finally, unable to get away from Haggerty, who is about to come around a corner and discover him, Hook stabs him. As the wounded man lies dying, Hook reaches down and grasps his shoulder, sharing strength and sympathy as they hold each other's gaze and the IRA man dies.
    Hook is captured by Quinn's group and taken to a hideout. Quinn orders Sean, a young teen whom Quinn has recruited, to execute Hook. When Sean hesitates, Quinn prepares to execute Hook. Browning's group arrives, and Lewis, to Hook's horror, shoots Sean. Lewis attempts to strangle Hook to prevent him from informing others of the bomb. As Lieutenant Armitage and his men enter in support of Browning, Armitage sees Lewis' attempt to kill Hook. Sean raises himself and fires at Lewis before being shot by Armitage. Browning finds Quinn and, rather than arrest him, tells him Boyle wants him dead. He promises to contact him soon, telling him he expects him to prove to be reasonable.
    Hook returns to his barracks. Later, despite a formal complaint by Armitage, the officer in charge dismisses the incident between Hook, Lewis, and Sean as a confused situation that merits no further inquiry. Hook returns to England and reunites with Darren.'''

    question = "Sensing danger, Hook flees the apartment?"

    answer = "Sensing danger, Hook flees the apartment, taking with him a sheathed blade."

    data = [[plot, question, answer]]
    phrase, question, label = elaborate(custom_data=data)
    #phrase, question, label = elaborate(plot, question, answer)

    phrase = np.reshape(phrase, (1, 100, 300))
    question = np.reshape(question, (1, 1, 300))
    label = np.reshape(label, (1, 1))

    prediction = model.predict([phrase, question])

    #print(prediction)
    pred = np.argmax(prediction[0])
    #print(prediction[0])
    print('result: {} predict: {}'.format(label, pred)) #la frase corretta è la label+1esima
    #print(phrase[0, pred, :])
    #print("QUESTION: ", question)
    #print("PREDICTION: ", prediction)


def train():
    train_phrases, train_questions, train_answers = load_dataset()

    model = get_model(300, 100, m=100, k=100)
    model.summary()

    train_questions = np.reshape(train_questions, (train_questions.shape[0], 1, train_questions.shape[1]))

    model.fit([train_phrases, train_questions], train_answers, batch_size=batch_size, epochs=epochs)

    model.save(model_path)

def main():
    train()
    test(model_path)


if __name__ == '__main__':
    main()