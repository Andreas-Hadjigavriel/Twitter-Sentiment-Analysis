# Machine Learning Algorithms and Deep Learning Models we used in this Project:
    # Logistic Regression
    # Ταξινοµητή XGBoost 
    # Naive Bayes 
    # Deep Learning Model: LSTM

### Βήματα Κώδικα:
    # Προεπεξεργασία Tweets
    # Μετατροπή τους σε Διανύσματα
    # Εκπαίδευση με βάσει το συναίσθημα χρησιμοποιώντας μοντέλα ταξινόμησης και αλγόριθμους βαθιάς μάθησης (LogReg, XGBoost, NaiveBayes, LSTM)

# ---------------------------------------------------------------------------------------------------------- #

# time
import time

# utilities
import re
import pickle
import numpy as np  
import pandas as pd

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltkma
import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# For evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------------------------------------- #

# Importing the dataset
ENCODING_DATA ="ISO-8859-1"
COLUMN_NAMES = ["sentiment", "ids", "date", "flag", "user", "text"]
dataset = pd.read_csv(r"Dataset_2.csv", encoding=ENCODING_DATA, names=COLUMN_NAMES)

# Choose only the columns we want to use
dataset = dataset[['sentiment','text']]
# Replacing the values to ease understanding
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

# Plotting the distribution for dataset (Negative: Sentiment = 0, Positive: Sentiment = 4)
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)
# Storing data in lists.
text, sentiment = list(dataset['text']), list(dataset['sentiment'])

# ---------------------------------------------------------------------------------------------------------- #

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# Defining List containing all stopwords
# Stopwords are the English words which does not add much meaning to a sentence. 
# They can safely be ignored without sacrificing the meaning of the sentence. (eg: "the", "he", "have")
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

# ---------------------------------------------------------------------------------------------------------- #

# Preprocessing Function (Clean the Data)
def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    # Lemmatization is the process of converting a word to its base form. (e.g: “Great” to “Good”)
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:

        # Each text is converted to lowercase
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText

# Process Time
t = time.time()
# Processed Dataset
processedtext = preprocess(text)
print(f"Text Preprocessing complete.")
print(f"Time Taken: {round(time.time()-t)} seconds")

# Ploting Word Clouds for Positive and Negative tweets from our dataset and see which words occur the most.
data_neg = processedtext[:800000]
plt.figure(figsize = (20,20))
wcNeg = WordCloud(max_words = 1000 , width = 1600 , height = 800, collocations=False).generate(" ".join(data_neg))
plt.imshow(wcNeg)

data_pos = processedtext[800000:]
wcPos = WordCloud(max_words = 1000 , width = 1600 , height = 800, collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (20,20))
plt.imshow(wcPos)

# ---------------------------------------------------------------------------------------------------------- #

# Splitting Data to Training and Testing Data
# Training Data: The dataset upon which the model would be trained on. Contains 95% data.
# Test Data: The dataset upon which the model would be tested against. Contains 5% data.
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size = 0.05, random_state = 0)
print(f'Data Split Complete.')

# Data Vectorize - Tokenization
# TF-IDF indicates what the importance of the word is in order to understand the document or dataset. 
# Let us understand with an example:
# Suppose you have a dataset where students write an essay on the topic, 
# My House. In this dataset, the word a appears many times; 
# it’s a high frequency word compared to other words in the dataset. 
# The dataset contains other words like home, house, rooms and so on that appear less often, 
# so their frequency are lower and they carry more information compared to the word. 

# TF-IDF Vectoriser converts a collection of raw documents to a matrix of TF-IDF features. 
# The Vectoriser is usually trained on only the X_train dataset.

# ngram_range: is the range of number of words in a sequence. 
# [e.g "very expensive" is a 2-gram that is considered as an extra feature separately from "very" and "expensive" when you have a n-gram range of (1,2)]

# max_features specifies the number of features to consider. 
# [Ordered by feature frequency across the corpus].
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

# Transforming the X_train and X_test dataset into matrix of TF-IDF Features by using the TF-IDF Vectoriser. 
# This datasets will be used to train the model and test against it.
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformation Complete. \n')

# ---------------------------------------------------------------------------------------------------------- #

def model_Evaluate(model):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted Values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual Values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

    print('Model Accuracy :',accuracy_score(y_test,y_pred))


# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
xgb_clf.fit(X_train,y_train)
model_Evaluate(xgb_clf)

# Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train,y_train)
model_Evaluate(nb_clf)

# Logistic Regrassion
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)

# ---------------------------------------------------------------------------------------------------------- #

# Saving the Models for later use
file = open('Vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectoriser, file)
file.close()

file = open('Sentiment-XGB.pickle','wb')
pickle.dump(xgb_clf, file)
file.close()

file = open('Sentiment-NB.pickle','wb')
pickle.dump(nb_clf, file)
file.close()

file = open('Sentiment-LR.pickle','wb')
pickle.dump(LRmodel, file)
file.close()

# ---------------------------------------------------------------------------------------------------------- #

def load_models():
    # Load the vectoriser.
    file = open('Vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the XGBoost Model.
    file = open('Sentiment-XGB.pickle', 'rb')
    xgb_clf = pickle.load(file)
    file.close()
    # Load the Naive Bayes Model.
    file = open('Sentiment-NB.pickle', 'rb')
    nb_clf = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, xgb_clf, nb_clf, LRmodel

# ---------------------------------------------------------------------------------------------------------- #

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

# ---------------------------------------------------------------------------------------------------------- #

if __name__=="__main__":
    # Loading the models.
    vectoriser, xgb_clf, nb_clf, LRmodel = load_models()

    # Text to classify should be in a list.
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    df1 = predict(vectoriser, xgb_clf, text)
    print(df1.head())

    df2 = predict(vectoriser, nb_clf, text)
    print(df2.head())

    df3 = predict(vectoriser, LRmodel, text)
    print(df3.head())


