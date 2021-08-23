import pandas as pd
import numpy as np
import math
import string
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from collections import Counter
from tqdm import tqdm
import warnings
import os
import pathlib
warnings.filterwarnings("ignore")

embeddings_dict = {}
fast_embs = '../fast-embeddings/cc.ka.300.vec'
with open(fast_embs) as f:
    for line in f:
        w, em = line.split(maxsplit=1)
        em = np.fromstring(em, 'f', sep=" ")
        embeddings_dict[w] = em

data = pd.read_csv('../comments.csv')
data = data[['comment', 'label']]
data.drop_duplicates(inplace=True)
data['label'] = data['label'].astype(np.int8)
data_not_toxic = data[data['label']==0].iloc[:5361]
data_toxic = data[data['label']==1]
data = pd.concat([data_not_toxic, data_toxic])
data.reset_index(drop=True, inplace=True)

class CommentPreparator(object):
    
    def __init__(self, mode = 'LEM'):
        self.mode = mode
        self.remove_punct = str.maketrans('', '', string.punctuation)
        self.remove_numbers = str.maketrans('', '', string.digits)
        self.remove_latin = str.maketrans('', '', string.ascii_letters)
        self.stop_words = ['და', 'თუ', 'მაგრამ', 'თორემ', 'ხოლო', 'ან', 'რომ',
              'თუ არა', 'რადგან', 'რათა', 'როგორ', 'რაკი', 'ვიდრე',
              'ვინც', 'რაც', 'სადაც', 'საიდანაც', 'საითკენაც', 'როდესაც'
              'როცა', 'ხომ', 'კი', 'დიახ', 'აბა რა', 'ე.ი.', 'რადგანაც',
              'რისთვისაც', 'როგორაც', 'როგორც', 'ოღონდ', 'ანუ', 'აშ', 'ეი', 'მეთქი',
              'თქო']    
    
    def __call__(self, comment):
        
        words = comment.split()
        words = [w.translate(self.remove_punct) for w in words]
        words = [w.translate(self.remove_numbers) for w in words]
        words = [w.translate(self.remove_latin) for w in words]
        words = [w for w in words if w not in self.stop_words]
        words = [w for w in words if len(w) > 0]
        return ' '.join(words)

comment_prep = CommentPreparator()
data['comment'] = data['comment'].apply(comment_prep)

vectorizer = TextVectorization(max_tokens=40000, output_sequence_length=100)
text_ds = tf.data.Dataset.from_tensor_slices(data['comment']).batch(128)
vectorizer.adapt(text_ds)

vocabulary = vectorizer.get_vocabulary()
size = len(vocabulary)
word_dictionary = dict(zip(vocabulary, range(size)))

vocab_size = size + 2
emb_dim = 300
n_unfound = 0

matrix = np.zeros((vocab_size, emb_dim))
for w, ix in word_dictionary.items():
    emb = embeddings_dict.get(w)
    if emb is not None:
        matrix[ix] = emb
    else:
        n_unfound += 1