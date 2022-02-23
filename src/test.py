from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
from tweet_scraping import scrape
import pandas as pd
import csv

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model('LSTM.h5')

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    
    sentiment_classes = ['Negative', 'Positive']
    max_len=50
    
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    # print('The predicted sentiment is', sentiment_classes[yt[0]])

    return sentiment_classes[yt[0]]

if __name__=="__main__": 
    
    # Enter Hashtag and initial date
    words = input("Enter Twitter HashTag to search for: ")
    date_since = input("Enter Date since The Tweets are required in yyyy-mm--dd: ")

    # number of tweets you want to extract in one run
    numtweet = 100
    scrape(words, date_since, numtweet)
    print('Scraping has completed!')
    print(f'\n Tweets Sentiment Prediction')

    df = pd.read_csv('scraped_tweets.csv')

    text = [['Mitsotakis is a very bad bad boy'], ['Mitsotakis is the best prime minister we have ever had in this beautiful country!!!!!'], ['Papandreou']]
    #df['text'].to_list()
    
    result = []
    for i in text:
        result.append(predict_class(i))

    for i in range(len(text)):
        print(text[i])
        print(result[i])
