import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf 
from keras import Sequential 
from keras.layers.core import Dense,Dropout

# Audio
import pyaudio
import speech_recognition as sr
import sys
from gtts import gTTS
from gtts.tokenizer.pre_processors import abbreviations, end_of_line
from pygame import mixer
import time
from mutagen.mp3 import MP3


def speak():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        textval = r.recognize_google(audio, language="fr-FR")
        print("J'ai entendu :", textval)
    except sr.UnknownValueError:
        print("J'ai rien compris")
    except sr.RequestError as e:
        print("L'API Speech de Google est hors service" + format(e))
    if textval == 'oui':
        print("Le fantôme arrive et est prêt à réponse au questions voulues")

    return textval

def robot(sentence):

    tts = gTTS(sentence, lang="fr", slow=False, pre_processor_funcs = [abbreviations, end_of_line]) # Save the audio in a mp3 file
    tts.save('buffer.mp3')# Play the audio
    audio = MP3("buffer.mp3")
    mixer.init()
    mixer.music.load("buffer.mp3")
    mixer.music.play()# Wait for the audio to be played
    time.sleep(audio.info.length)

nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()

dictionnaireDintentions = {"joueurs_equipe_france" : [
    {"tag" : "ATTAQUANTS",
     "patterns" : ["attaquants","quels sont les joueurs en attaque"],
     "reponses" : ["Marcus Thuram", "Wissam Ben Yedder", "Randal Kolo Muani", "Karim Benzema", "Ousmane Dembélé","Kylian Mbappé", "Olivier Giroud"]},
     {"tag" : "MILIEUX",
     "patterns" : ["milieu","quels sont les joueurs du milieu"],
     "reponses" : ["Eduardo Camavinga", "Paul Pogba","Antoine Griezmann","Aurélien Tchouameni","Kingsley Coman","Thomas Lemar","Mattéo Guendouzi","Adrien Rabiot","Florian Thauvin","Jordan Veretout"
     ,"Youssouf Fofana"]},
     {"tag" : "DÉFENSEURS",
     "patterns" : ["défense","quels sont les joueurs en défense"],
     "reponses" : ["Axel Disasi", "Benjamin Pavard", "William Saliba","Ibrahima Konaté","Raphaël Varane","Jules Koundé","Dayot Upamecano","Lucas Digne","Léo Dubois","Clément Lenglet"
     ,"Lucas Hernández","Samuel Umtiti","Theo Hernández"]},
     {"tag" : "GARDIENS",
     "patterns" : ["but","gardien"],
     "reponses" : ["Hugo Lloris", "Steve Mandanda", "Mike Maignan","Alphonse Aréola"]},
     {"tag" : "DATE",
     "patterns" : ["date","coupe du monde","mois"],
     "reponses" : ["La coupe du monde démarre le dimanche 20 novembre"]},
     {"tag" : "VOIR",
     "patterns" : ["voir le match","ou","chez"],
     "reponses" : ["Tu peux voir le match de la coupe du monde chez Jeremy"]}]
}

#print(dictionnaireDintentions["intentions"][0]["tag"])

tagsList = []
doc_X = []
doc_y = []
motsDistinctPattern = []

print("----------------------------------------")
for i in dictionnaireDintentions["joueurs_equipe_france"]:
    tagsList.append(i["tag"])


for i in dictionnaireDintentions["joueurs_equipe_france"]:
    for pattern in i["patterns"]:
        doc_X.append(pattern)

for i in dictionnaireDintentions["joueurs_equipe_france"]:
    tag = i["tag"] 
    for pattern in i["patterns"]:
        doc_y.append(tag)

# mot distincs pattern
for i in dictionnaireDintentions["joueurs_equipe_france"]:
    for pattern in i["patterns"]:
        for w in word_tokenize(pattern):
            if w not in motsDistinctPattern:
                motsDistinctPattern.append(w)

# liste pour les données d'entraînement
training = []
out_empty = [0] * len(tagsList)
# création du modèle d'ensemble de mots
for idx, doc in enumerate(doc_X):
    bow = []
    text = wnl.lemmatize(doc.lower())
    for word in motsDistinctPattern:
        bow.append(1) if word in text else bow.append(0)
    # marque l'index de la classe à laquelle le pattern atguel est associé à
    output_row = list(out_empty)
    output_row[tagsList.index(doc_y[idx])] = 1
    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])
# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)
# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

model.fit(x=train_X, y=train_y, epochs=15)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [wnl.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
        if word == w: 
            bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json): 
    tag = intents_list[0]
    list_of_intents = intents_json["joueurs_equipe_france"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["reponses"])
            break
    return result

robot("Bienvenue au Qatar prêt à dépenser ton fric petite merde, allez j'ai pas que ça a faire ?")
while True:
    print("---------------------")
    question = "Pose ici ta question"
    robot(question)
    #message = input()
    message = speak()
    if message.lower() == "quitter":
        robot("Au plaisir de parler avec toi beau goss!")
        break
    else:
        intentions = pred_class(message, motsDistinctPattern, tagsList)
        reponse = get_response(intentions, dictionnaireDintentions)
        print(reponse)
        robot(reponse)
