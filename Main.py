from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer, util #loading sbert sentence transformet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Bidirectional
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer

main = Tk()
main.title("Hashtag-Based Tweet Expansion for Improved Topic Modeling")
main.geometry("1300x1200")

global filename, dataset, X, Y, vectorizer, scaler, precision, fscore, tweets

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    tf1.insert(END,str(filename))
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))

def preprocessDataset():
    global dataset, tweets
    tweets = []
    text.delete('1.0', END)
    dataset = dataset.values
    cnt = Counter()
    for i in range(len(dataset)):
        tweets.append(dataset[i,0])
        words = dataset[i,0].split(" ")
        words = [w for w in words if not w in stop_words]
        for j in range(len(words)):
            cnt[words[j]] += 1
    word = []
    count = []
    top_words = sorted(cnt.items(), key=itemgetter(1), reverse=True)
    for w,c in top_words:
        word.append(w)
        count.append(c)
        if len(word) > 10:
            break
    height = count
    bars = word
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Top K Words")
    plt.ylabel("Count")
    plt.show()
    

def runBILSTM():
    text.delete('1.0', END)
    global dataset, vectorizer, X, Y, scaler, precision, fscore
    precision = []
    fscore = []
    scaler = MinMaxScaler(feature_range = (0, 1))
    if os.path.exists("model/X.npy"):
        with open('model/vector.txt', 'rb') as file:
            vectorizer = pickle.load(file)
        file.close()
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        textdata = []
        labels = []
        for i in range(len(dataset)):
            msg = dataset[i,0]
            msg = msg.strip().lower()
            clean = cleanPost(msg)
            for j in range(0,5):
                textdata.append(clean)
                labels.append(dataset[i,1])
        vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
        X = vectorizer.fit_transform(textdata).toarray()
        np.save("model/X", X)
        np.save("model/Y", labels)
        with open('model/vector.txt', 'wb') as file:
            pickle.dump(vectorizer, file)
        file.close()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = scaler.fit_transform(X)
    Y = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], 30, 10))
    print(XX.shape)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    bilstm = Sequential()#defining deep learning sequential object
    #adding bi-directional LSTM layer with 32 filters to filter given input X train data to select relevant features
    bilstm.add(Bidirectional(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)))
    #adding dropout layer to remove irrelevant features
    bilstm.add(Dropout(0.2))
    #adding another layer
    bilstm.add(Bidirectional(LSTM(32)))
    bilstm.add(Dropout(0.2))
    #defining output layer for prediction
    bilstm.add(Dense(y_train.shape[1], activation='softmax'))
    #compile BI-LSTM model
    bilstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if os.path.exists("model/bilstm.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/bilstm.hdf5', verbose = 1, save_best_only = True)
        hist = bilstm.fit(X_train, y_train, batch_size = 16, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/bilstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        bilstm = load_model("model/bilstm.hdf5")
    predict = bilstm.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    fscore.append(f)
    text.insert(END,"Bi-LSTM Hashtag Expansion Precision = "+str(p)+"\n")
    text.insert(END,"Bi-LSTM Hashtag Expansion FSCORE    = "+str(f)+"\n\n")

def runBert():
    global dataset
    if os.path.exists('model/bert_X.npy'):
        bert_X = np.load('model/bert_X.npy')#save embedding for future use    
    else:
        bert = SentenceTransformer('nli-distilroberta-base-v2')
        embeddings = bert.encode(textdata, convert_to_tensor=True)#now using bert to encode sepsis data to create embeded vector=========
        bert_X = embeddings.numpy()
        np.save('model/bert_X',bert_X)#save embedding for future use
    Y1 = np.load("model/Y.npy")        
    indices = np.arange(bert_X.shape[0])
    np.random.shuffle(indices) #shuffle the dataset
    bert_X = bert_X[indices]
    Y1 = Y1[indices]
    Y1 = Y1.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(bert_X, Y1, test_size=0.8)
    rf = GaussianNB()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    precision.append(p)
    fscore.append(f)
    text.insert(END,"BERT Hashtag Expansion Precision = "+str(p)+"\n")
    text.insert(END,"BERT Hashtag Expansion FSCORE    = "+str(f)+"\n\n")

def graph():
    precision, fscore
    df = pd.DataFrame([['Bi-LSTM','Precision',precision[0]],['Bi-LSTM','F1 Score',fscore[0]],
                       ['BERT','Precision',precision[1]],['BERT','F1 Score',fscore[1]],                                           
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()

def getTopic(tweetdata):
    data = []
    for i in range(0,10):
        data.append(tweetdata)
    number_of_Words = 10
    topicList = []
    weightList = []
    textList = []
    num_comp = 50
    iteration = 5
    method = 'online'
    offset = 50.
    state = 42
    minValue = 1
    maxValue = 100
    totalFeatures = 100
    vector = CountVectorizer(min_df=minValue, stop_words='english', max_df=maxValue, max_features=totalFeatures)
    tfIDF = vector.fit_transform(data)
    lda_allocation = LatentDirichletAllocation(max_iter=iteration,n_components=num_comp, learning_offset=offset, learning_method=method, random_state=state)
    lda_allocation.fit(tfIDF)
    features = vector.get_feature_names()
    for index, name in enumerate(lda_allocation.components_):
        topic="Topic No %d: " % index
        topic+=" ".join([features[j] for j in name.argsort()[:-number_of_Words - 1:-1]])
        textList.append(topic)

    for m,topic in enumerate(lda_allocation.components_):
        topicList.append([vector.get_feature_names()[i] for i in topic.argsort()[-1:]][0])
    topics = ""
    top = []
    for i in range(len(topicList)):
        if len(top) < 3:
            if len(topicList[i]) > 5:
                if topicList[i] not in top:
                    topics += topicList[i]+" "
                    top.append(topicList[i])
    text.insert(END,"\n\nDetected Topic  : "+str(topics)+"\n")
    
def predict():
    global vectorizer, X, Y, scaler
    text.delete('1.0', END)
    with open('model/vector.txt', 'rb') as file:
        vectorizer = pickle.load(file)
    file.close()
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
    hashtag = simpledialog.askstring("Enter Hashtag to expand Tweet", "Enter Hashtag to expand Tweet",parent=main)
    tag = hashtag
    hashtag = hashtag.strip().lower()
    index = -1
    for i in range(len(tweets)):
        arr = tweets[i].split()
        if hashtag in arr:
            index = i
            break
    if index != -1:
        text.insert(END,"Expanded Tweet from given Hashtag : "+tag+"\n\n")
        text.insert(END,tweets[index])
        getTopic(tweets[index])
    else:
        text.insert(END, "Unable to find topic")

def close():
    main.destroy()


font = ('times', 15, 'bold')
title = Label(main, text='Hashtag-Based Tweet Expansion for Improved Topic Modeling')
title.config(bg='mint cream', fg='royal blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=40)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = Button(main, text="Upload Tweets Dataset", command=uploadDataset)
uploadButton.place(x=50,y=150)
uploadButton.config(font=ff)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=350,y=150)
preprocessButton.config(font=ff)

lstmButton = Button(main, text="Run Bi-LSTM Algorithm", command=runBILSTM)
lstmButton.place(x=50,y=200)
lstmButton.config(font=ff)

bertButton = Button(main, text="Run Bert Algorithm", command=runBert)
bertButton.place(x=350,y=200)
bertButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=ff)

predictButton = Button(main, text="Tweet Expansion & Modelling using Hashtag", command=predict)
predictButton.place(x=350,y=250)
predictButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=650,y=250)
exitButton.config(font=ff)

font1 = ('times', 13, 'bold')
text=Text(main,height=18,width=125)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='salmon')
main.mainloop()
