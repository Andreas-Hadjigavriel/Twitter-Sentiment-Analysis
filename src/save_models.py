# ------------------------------------- MODELS SAVING  ------------------------------------------- #

import pickle

def save_model(model, modelfilename):
    file = open(f'{modelfilename}.pickle','wb')
    pickle.dump(model, file)
    file.close()