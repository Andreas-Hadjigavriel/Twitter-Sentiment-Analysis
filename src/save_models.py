# ------------------------------------- MODELS SAVING  ------------------------------------------- #

import pickle

def save_model(model):
    file = open(f'{model}.pickle','wb')
    pickle.dump(model, file)
    file.close()