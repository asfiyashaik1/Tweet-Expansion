import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util #loading sbert sentence transformet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import os
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Bidirectional
from sklearn.naive_bayes import GaussianNB

'''
dataset = pd.read_csv("Dataset/tweets.csv",usecols = ["tweet"])
dataset = dataset.values

X = []
for i in range(len(dataset)):
    tweet = dataset[i,0]
    hashtag = set(part[1:] for part in tweet.split() if part.startswith('#'))
    hashtag = list(hashtag)
    for j in range(1, len(hashtag)):
        tweet = tweet.replace(hashtag[j], "")
    tweet = tweet.replace("#","")    
    tweet_arr = tweet.split()
    if len(tweet_arr) > 10 and len(hashtag) > 0:
        X.append([tweet, "#"+hashtag[0]])

data = pd.DataFrame(X, columns=['tweet','tag'])
data.to_csv("dataset.csv", index = False)
print(data.shape)
'''
'''
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

dataset = pd.read_csv("dataset.csv")
le = LabelEncoder()
dataset['tag'] = pd.Series(le.fit_transform(dataset['tag'].astype(str)))


textdata = []
labels = []

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

dataset = dataset.values
for i in range(len(dataset)):
    msg = dataset[i,0]
    msg = msg.strip().lower()
    clean = cleanPost(msg)
    for j in range(0,5):
        textdata.append(clean)
        labels.append(dataset[i,1])
'''
#now convert dataset into bert embedding vector
if os.path.exists('model/bert_X.npy'):
    bert_X = np.load('model/bert_X.npy')#save embedding for future use    
else:
    bert = SentenceTransformer('nli-distilroberta-base-v2')
    embeddings = bert.encode(textdata, convert_to_tensor=True)#now using bert to encode sepsis data to create embeded vector=========
    bert_X = embeddings.numpy()
    np.save('model/bert_X',bert_X)#save embedding for future use
Y = np.load("model/Y.npy")        
indices = np.arange(bert_X.shape[0])
np.random.shuffle(indices) #shuffle the dataset
bert_X = bert_X[indices]
Y = Y[indices]
Y = Y.astype(int)
print(Y)
print(bert_X)

X_train, X_test, y_train, y_test = train_test_split(bert_X, Y, test_size=0.8)

rf = GaussianNB()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)
        
'''
vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
X = vectorizer.fit_transform(textdata).toarray()
np.save("model/X", X)
np.save("model/Y", labels)
with open('model/vector.txt', 'wb') as file:
    pickle.dump(vectorizer, file)
file.close()

'''

scaler = MinMaxScaler(feature_range = (0, 1))

with open('model/vector.txt', 'rb') as file:
    vectorizer = pickle.load(file)
file.close()
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X = scaler.fit_transform(X)
Y = to_categorical(Y)

XX = np.reshape(X, (X.shape[0], 30, 10))
print(XX.shape)
X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)


print(X.shape)
print(Y)


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
acc = accuracy_score(y_test, predict)
print(acc)




