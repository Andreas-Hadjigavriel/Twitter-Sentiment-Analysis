# --------------------------------------------- MAIN -------------------------------------------------- #
from prediction import predict 
from load_models import load_model
from tweet_scraping import scrape
import pandas as pd
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model as loadModel

model = loadModel('best_model.h5')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Return the predicted sentiment
    return sentiment_classes[yt[0]]

if __name__=="__main__": 
    # Enter Hashtag and initial date
    words = input("Enter Twitter HashTag to search for: ")
    date_since = input("Enter Date since The Tweets are required in yyyy-mm--dd: ")

    # number of tweets you want to extract in one run
    numtweet = 100
    scrape(words, date_since, numtweet)

    vectoriserFilename ="Vectoriser"
    lrFilename = "LogisticRegression"
    svcFilename = "LinearSVC"
    nbFilename = "MultinomialNB"
    xgbFilename = "XGBClassifier"
    dtcFilename = "DTClassifier"
    lstmFilename = 'LSTM'

    # Loading the models
    vectoriser, xgb_clf = load_model(vectoriserFilename, xgbFilename)
    vectoriser, nb_clf = load_model(vectoriserFilename, nbFilename)
    vectoriser, SVCmodel = load_model(vectoriserFilename, svcFilename)
    vectoriser, LRmodel = load_model(vectoriserFilename, lrFilename)
    vectoriser, dtc_clf = load_model(vectoriserFilename, dtcFilename)

    df = pd.read_csv('scraped_tweets.csv') 

    text = df['text'].to_list()
    
    # Prediction based on each model
    print(f"-------------------------- XGBoost ---------------------------")
    pr1 = predict(vectoriser, xgb_clf, text)
    print(pr1.head())   

    print(f"\n------------------ Multinomial Naive Bayes -------------------")
    pr2 = predict(vectoriser, nb_clf, text)
    print(pr2.head())

    print(f"\n------------------------- Linear SVC -------------------------")
    pr3 = predict(vectoriser, SVCmodel, text)
    print(pr3.head())
    
    print(f"\n---------------------- Linear Regression ---------------------")
    pr4 = predict(vectoriser, LRmodel, text)
    print(pr4.head())

    print(f"\n----------------------- Decision Tree ------------------------")
    pr5 = predict(vectoriser, dtc_clf, text)
    print(pr5.head())

    print(f"\n---------------------------- LSTM ----------------------------")
    temp = []
    for i in text:
        temp.append((i, predict_class(i)))
    pr6 = pd.DataFrame(temp, columns = ['text','sentiment'])
    print(pr6.head())
