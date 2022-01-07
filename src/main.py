# ----------------------------------- IMPORTING DEPENDENCIES ---------------------------------------- #

# project files
from emojis_stopwords_def import *
from preprocessing import preprocess
from save_models import save_model
from load_models import load_model
from prediction import predict
 
# time
import time

# utilities
import numpy as np  
import pandas as pd

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# For evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# -------------------------------------- DATA COLLECTION ---------------------------------------------- #

ENCODING_DATA ="ISO-8859-1"
COLUMN_NAMES = ["sentiment", "ids", "date", "flag", "user", "text"]
dataset = pd.read_csv(r"../Dataset_2.csv", encoding=ENCODING_DATA, names=COLUMN_NAMES)

# Choose only the columns we want to use
dataset = dataset[['sentiment','text']]
# Replacing the values to ease understanding
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

# Plotting the distribution for dataset (Negative: Sentiment = 0, Positive: Sentiment = 4)
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)
# Storing data in lists.
text, sentiment = list(dataset['text']), list(dataset['sentiment'])


# --------------------------------- DATA PREPROCESSING/PREPARATION ------------------------------------ #

# Process Time
t = time.time()
# Processed Dataset
processedtext = preprocess(text)   
print(f"Text Preprocessing complete.")
print(f"Time Taken: {round(time.time()-t)} seconds \n")


# ---------------------------- DATA SPLITTING TO TRAINING AND TESTING DATA ---------------------------- #

X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size = 0.05, random_state = 0)
print(f'Data Split Complete.')


# --------------------------------------- DATA VECTORIZATION ------------------------------------------ #

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('Number of feature words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformation to Vectors Completed. \n')


# ---------------------------------- MODEL ANALYSIS AND EVALUATION ------------------------------------ #

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

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '', xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted Values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual Values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)

    plt.show()

    print('Model Accuracy :',accuracy_score(y_test,y_pred))

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss',use_label_encoder=False)
xgb_clf.fit(X_train,y_train)
model_Evaluate(xgb_clf)

# Naive Bayes Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train,y_train)
model_Evaluate(nb_clf)

# SVM Classifier
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)

# Logistic Regrassion
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)


# ---------------------------------------- SAVING THE MODELS ------------------------------------------ #
vectoriserFilename ="Vectoriser"
lrFilename = "LogisticRegression"
svcFilename = "LinearSVC"
nbFilename = "MultinomialNB"
xgbFilename = "XGBClassifier"

save_model(vectoriser, vectoriserFilename)
save_model(LRmodel, lrFilename)
save_model(SVCmodel, svcFilename)
save_model(nb_clf, nbFilename)
save_model(xgb_clf, xgbFilename)


# --------------------------------------------- MAIN -------------------------------------------------- #

if __name__=="__main__": 
    # Loading the models
    vectoriser, xgb_clf = load_model(vectoriserFilename, xgbFilename)
    vectoriser, nb_clf = load_model(vectoriserFilename, nbFilename)
    vectoriser, SVCmodel = load_model(vectoriserFilename, svcFilename)
    vectoriser, LRmodel = load_model(vectoriserFilename, lrFilename)
    
    print(f'\n Tweets Sentiment Prediction')

    # Text to classify
    text = ["The weather is good today",
            "I don't like the weather today"]

    # Prediction 
    df1 = predict(vectoriser, xgb_clf, text)
    print(df1.head())   

    df2 = predict(vectoriser, nb_clf, text)
    print(df2.head())

    df3 = predict(vectoriser, SVCmodel, text)
    print(df3.head())
    
    df4 = predict(vectoriser, LRmodel, text)
    print(df4.head())
