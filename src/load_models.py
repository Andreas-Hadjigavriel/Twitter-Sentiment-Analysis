# ----------------------------------------- MODEL LOADING --------------------------------------------------- #

import pickle

def load_model(vect, model):
    # Load the vectoriser.
    file = open(f'{vect}.pickle', 'rb')
    vect = pickle.load(file)
    file.close()
    # Load the Model.
    file = open(f'{model}.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    
    return vect, model
