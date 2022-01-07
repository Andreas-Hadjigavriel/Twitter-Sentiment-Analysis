# ----------------------------------------- MODEL LOADING --------------------------------------------------- #

import pickle

def load_model(vectFilename, modelFilename):
    # Load the vectoriser.
    file = open(f'{vectFilename}.pickle', 'rb')
    vect = pickle.load(file)
    file.close()
    # Load the Model.
    file = open(f'{modelFilename}.pickle', 'rb')
    model = pickle.load(file)
    file.close()
    
    return vect, model
