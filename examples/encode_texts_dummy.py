# -*- coding: utf-8 -*-

""" Use DeepMoji to encode texts into emotional feature vectors.
"""
from __future__ import print_function, division
import codecs
import example_helper
import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import VOCAB_PATH
from keras.models import load_model
import emoji
import sys

# Emoji map in emoji_overview.png
EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

TEST_SENTENCES = ['i had trouble sleeping on days when i took it',
                  'the audio is pretty mundane with japanese audio tracks to cover the horrible english voice actresses..',
                  'here on amazon ,  this was too expensive for this',
                  'i m surprised no one has sued the company for manufacturing this awkward ,  uncomfortable ,  dangerous product',
                  'they make much more reliable mice and i never had a problem with them',
                  'this was a bad purchase ,  after using a month for water ,  something grew on the stainless inside wall',
                  'i cleaned it thinking that maybe that would help ,  but to no avail']
TEST_SENTENCES = [unicode(s) for s in TEST_SENTENCES]
# TEST_SENTENCES = codecs.open(sys.argv[1], 'r', encoding='utf-8').readlines()

maxlen = 50
batch_size = 64
print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

# def batchify(tokenized):
#
#     return tokenized_batches
MODEL_PATH = "/home/ubuntu/akhilesh_data/akhilesh/DeepMoji/model/model/kaggle_insults_dummy_model/m.hdf5"
print('Loading encoding model from {}.'.format(MODEL_PATH))
st_id2tok = [None]*len(st.vocabulary)
for w in st.vocabulary:
    st_id2tok[st.vocabulary[w]]=w
tokenized_sents = [[st_id2tok[i] for i in row] for row in tokenized]
# model = torchmoji_feature_encoding(PRETRAINED_PATH, return_attention=True)
# print(model)
#
# print('Encoding texts..')
# encoding, att_weights = model(tokenized)
# att_weights = att_weights.cpu().data.numpy()

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

print('Loading emoji pred model from {}.'.format(MODEL_PATH))
# model = deepmoji_emojis(maxlen, PRETRAINED_PATH, return_attention=True)
model = load_model(MODEL_PATH)
model.summary()
print('Running predictions.')
prob = model.predict(tokenized)
pass

# emojis = []
# for prob in [prob]:
#     # Find top emojis for each sentence. Emoji ids (0-63)
#     # correspond to the mapping in emoji_overview.png
#     # at the root of the torchMoji repo.
#     for i, t in enumerate(TEST_SENTENCES):
#         t_tokens = tokenized[i]
#         t_score = [t]
#         t_prob = prob[i]
#         ind_top = top_elements(t_prob, 5)
#         tmp = map(lambda x: EMOJIS[x], ind_top)
#         emojis.append(tmp)
#
# for ind, (atts, ems) in enumerate(zip(att_weights, emojis)):
#     fo.write(emoji.emojize("{} {}\n".format(TEST_SENTENCES[ind], ' '.join(ems)), use_aliases=True))
#     idxs = np.argsort(atts)[::-1]
#     for i in idxs:
#         if tokenized_sents[ind][i]==st_id2tok[0]:
#             continue
#         fo.write("{} {}\n".format(tokenized_sents[ind][i], atts[i]))
#     fo.write('-----------\n')
# fo.close()
# print(ind, file=sys.stdout)