import pickle
DATASET_PATH = '../data/kaggle-insults/raw_dummy.pickle'
dataset = pickle.load(open(DATASET_PATH, 'rb'))
print('Dummy')