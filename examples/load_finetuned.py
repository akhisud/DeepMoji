"""Finetuning example.

Trains the DeepMoji model on the kaggle insults dataset, using the 'chain-thaw'
finetuning method and the accuracy metric. See the blog post at
https://medium.com/@bjarkefelbo/what-can-we-learn-from-emojis-6beb165a5ea0
for more information. Note that results may differ a bit due to slight
changes in preprocessing and train/val/test split.

The 'chain-thaw' method does the following:
0) Load all weights except for the softmax layer. Extend the embedding layer if
   necessary, initialising the new weights with random values.
1) Freeze every layer except the last (softmax) layer and train it.
2) Freeze every layer except the first layer and train it.
3) Freeze every layer except the second etc., until the second last layer.
4) Unfreeze all layers and train entire model.
"""

from __future__ import print_function
import example_helper
import json
from deepmoji.model_def import deepmoji_transfer
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from deepmoji.finetuning import (
    load_benchmark,
    finetune)
import pickle
import time
from keras.models import load_model, Model
from keras.utils import CustomObjectScope
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.attlayer import AttentionWeightedAverage
import codecs
import sys
import numpy as np
TEST_SENTENCES = codecs.open(sys.argv[1], 'r', encoding='utf-8').readlines()

# TEST_SENTENCES = ['i had trouble sleeping on days when i took it',
#                   'the audio is pretty mundane with japanese audio tracks to cover the horrible english voice actresses..',
#                   'here on amazon ,  this was too expensive for this',
#                   'i m surprised no one has sued the company for manufacturing this awkward ,  uncomfortable ,  dangerous product',
#                   'they make much more reliable mice and i never had a problem with them',
#                   'this was a bad purchase ,  after using a month for water ,  something grew on the stainless inside wall',
#                   'i cleaned it thinking that maybe that would help ,  but to no avail']
# TEST_SENTENCES = [unicode(s) for s in TEST_SENTENCES]

print('Loading model.')
with CustomObjectScope({'AttentionWeightedAverage': AttentionWeightedAverage}):
    # model = load_model("/home/ubuntu/akhilesh_data/DeepMoji/model/model/amazon_chain_thaw/m.hdf5")
    model = load_model("/home/ubuntu/akhilesh_data/DeepMoji/model/model/yelp_chain_thaw/m.hdf5")
model.summary()

with open('../model/vocabulary.json', 'r') as f:
    vocab = json.load(f)

# Load dataset. Extend the existing vocabulary with up to 10000 tokens from
# the training dataset.
DATASET_PATH = '../data/yelp/raw.pickle'
# data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)
st = SentenceTokenizer(vocab, 20)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Reverse tokenizing.'.format(PRETRAINED_PATH), file=sys.stdout)
st_id2tok = [None]*len(st.vocabulary)
for w in st.vocabulary:
    st_id2tok[st.vocabulary[w]] = w
tokenized_sents = [[st_id2tok[i] for i in row] for row in tokenized]

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('attlayer').output)

print('Running predictions.')
prob = model.predict(tokenized)
_, att_weights = intermediate_layer_model.predict(tokenized)
fo = codecs.open(sys.argv[2], 'w', encoding='utf-8')
# fo = codecs.open('/tmp/waste.txt', 'w', encoding='utf-8')
for ind, (atts, prob) in enumerate(zip(att_weights, prob)):
    label = '+ve' if prob[0] > 0.5 else '-ve'
    fo.write("{}\nLABEL: {}\n".format(TEST_SENTENCES[ind], label))
    idxs = np.argsort(atts)[::-1]
    for i in idxs:
        if tokenized_sents[ind][i]==st_id2tok[0]:
            continue
        fo.write("{} {}\n".format(tokenized_sents[ind][i], atts[i]))
    fo.write('-----------\n')
fo.close()
print(ind)

