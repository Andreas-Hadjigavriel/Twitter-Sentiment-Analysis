
# --------------------------------------------- MAIN -------------------------------------------------- #

from prediction import predict 
from load_models import load_model

if __name__=="__main__": 

    vectoriserFilename ="Vectoriser"
    lrFilename = "LogisticRegression"
    svcFilename = "LinearSVC"
    nbFilename = "MultinomialNB"
    xgbFilename = "XGBClassifier"

    # Loading the models
    vectoriser, xgb_clf = load_model(vectoriserFilename, xgbFilename)
    vectoriser, nb_clf = load_model(vectoriserFilename, nbFilename)
    vectoriser, SVCmodel = load_model(vectoriserFilename, svcFilename)
    vectoriser, LRmodel = load_model(vectoriserFilename, lrFilename)
    
    print(f'\n Tweets Sentiment Prediction')

    # Tweets to classify
    text = ["The weather is good today",
            "I don't like the weather today",
            "Mitsotaki and Kerameos fuck yourself together yesterday",
            "Emma you are beautiful",
            "Bad Mitsotakis and beautiful Kerameos fuck you."]
    
    # Prediction based on each model
    pr1 = predict(vectoriser, xgb_clf, text)
    print(pr1.head())   

    pr2 = predict(vectoriser, nb_clf, text)
    print(pr2.head())

    pr3 = predict(vectoriser, SVCmodel, text)
    print(pr3.head())
    
    pr4 = predict(vectoriser, LRmodel, text)
    print(pr4.head())
