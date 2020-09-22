import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, redirect, url_for, render_template, request, send_file, session
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os
import playsound
import string 
from gtts import gTTS
import psycopg2
from sqlalchemy import create_engine

TRAIN = False #Flag that controls training the model

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if TRAIN == False:
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    try:
        s_words = nltk.word_tokenize(s)
    except:
        nltk.download('punkt')
        s_words = nltk.word_tokenize(s)

    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat(s):

    results = model.predict([bag_of_words(s.lower(), words)])
    results_index = numpy.argmax(results)

    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return random.choice(responses)

def speak(s):
    tts = gTTS(text=s, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def savereply(s):
    tts = gTTS(text=s, lang="en")
    filename = ''.join(random.choices(string.ascii_lowercase, k = 5)) 
    filenamefull = str("static/audio/" + filename + ".mp3")
    tts.save(filenamefull)  
    return str(filename)

#CODE STARTS HERE
app = Flask(__name__, template_folder='static', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'key'

DBNAME = "dbt7kev3gaguro" #Database name
HOSTNAME = "ec2-52-17-53-249.eu-west-1.compute.amazonaws.com" #Host name
USER = "ydwfpipdtiadxy" #User's name
PASS = "7e9a920eba2619f02ccf651a5ed6af726f651e3df7e90f55d166355dc8723451" #Password

#Connect to database
conn = psycopg2.connect(dbname=DBNAME, user=USER, password=PASS, host=HOSTNAME)
engine = create_engine('postgres://ydwfpipdtiadxy:7e9a920eba2619f02ccf651a5ed6af726f651e3df7e90f55d166355dc8723451@ec2-52-17-53-249.eu-west-1.compute.amazonaws.com:5432/dbt7kev3gaguro', paramstyle='format') #postgresql://postgres:latlong123@localhost:5432/inventorymanager
cur = conn.cursor() 

@app.route("/", methods=["POST", "GET"])
def home():

    message = ""
    path = ""
    mydir = "static/audio/"

    filelist = [ f for f in os.listdir(mydir) if f.endswith(".mp3") ]
    for f in filelist:
        os.remove(os.path.join(mydir, f))

    if request.method == "POST":
        message = request.form["msg"]

        print(message)

        if message != "":
            output = chat(message)
            print(output)
            path = savereply(output)
            with open("currmessage.pickle", "wb") as f:
                pickle.dump((message, output), f)

    return render_template("index.html", soundpath = str("audio/" + path + ".mp3"))

@app.route("/report")
def report():
    with open("currmessage.pickle", "rb") as f:
         message, output = pickle.load(f)

    if message != None and output != None and message != "" and output != "":

        output = output.replace("'",'')
        message = message.replace("'",'')

        appendstring = "INSERT INTO public.reports (userinput, robbieoutput) VALUES ('{}', '{}');".format(message, output)

        cur.execute(appendstring)
        conn.commit()
        conn.close()  

    with open("currmessage.pickle", "wb") as f:
        pickle.dump(("", ""), f)

    return redirect(url_for("home"))   

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, use_reloader=True)

'''
SELECT * FROM public.reports
ORDER BY report_id ASC 
'''