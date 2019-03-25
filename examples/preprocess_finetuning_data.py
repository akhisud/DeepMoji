import pickle
from collections import defaultdict
import os
import random

# DATA = '/home/ubuntu/data/amazon'
DATA = '/home/ubuntu/data/yelp'
POS_TRAIN_FILE_PATH = os.path.join(DATA, 'sentiment.train.1')
NEG_TRAIN_FILE_PATH = os.path.join(DATA,'sentiment.train.0')
POS_VAL_FILE_PATH = os.path.join(DATA, 'sentiment.dev.1')
NEG_VAL_FILE_PATH = os.path.join(DATA,'sentiment.dev.0')
POS_TEST_FILE_PATH = os.path.join(DATA, 'sentiment.test.1')
NEG_TEST_FILE_PATH = os.path.join(DATA,'sentiment.test.0')
data = defaultdict(list)

pos_data = open(POS_TRAIN_FILE_PATH, 'r').readlines()
neg_data = open(NEG_TRAIN_FILE_PATH, 'r').readlines()
pos_val_data = open(POS_VAL_FILE_PATH, 'r').readlines()
neg_val_data = open(NEG_VAL_FILE_PATH, 'r').readlines()
pos_test_data = open(POS_TEST_FILE_PATH, 'r').readlines()
neg_test_data = open(NEG_TEST_FILE_PATH, 'r').readlines()

pos = [(i.strip(), {'label':1}) for i in pos_data]
neg = [(i.strip(), {'label':0}) for i in neg_data]
pos_val = [(i.strip(), {'label':1}) for i in pos_val_data]
neg_val = [(i.strip(), {'label':0}) for i in neg_val_data]
pos_test= [(i.strip(), {'label':1}) for i in pos_test_data]
neg_test = [(i.strip(), {'label':0}) for i in neg_test_data]
pos_neg = pos+neg
pos_neg_val = pos_val+neg_val
pos_neg_test = pos_test+neg_test
random.shuffle(pos_neg)
random.shuffle(pos_neg_val)
random.shuffle(pos_neg_test)

data[str('texts')] = [i[0] for i in pos_neg]+[i[0] for i in pos_neg_val]+[i[0] for i in pos_neg_test]
data[str('info')] = [i[1] for i in pos_neg]+[i[1] for i in pos_neg_val]+[i[1] for i in pos_neg_test]
data[str('train_ind')] = list(range(len(pos_neg)))
data[str('val_ind')] = list(range(len(pos_neg),len(pos_neg)+len(pos_neg_val)))
data[str('test_ind')] = list(range(len(pos_neg)+len(pos_neg_val),len(pos_neg)+len(pos_neg_val)+len(pos_neg_test)))

pickle.dump(dict(data), open('/home/ubuntu/raw.pickle', 'wb'), protocol=2)