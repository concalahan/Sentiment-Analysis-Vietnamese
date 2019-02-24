# -*- coding: utf-8 -*-
import sys
import codecs
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import multiprocessing
import pandas as pd
import numpy as np
# np.set_printoptions(threshold=np.inf)
import tensorflow as tf
from sklearn import utils
from sklearn.preprocessing import scale
from keras.layers import Embedding,Conv1D,CuDNNLSTM,Input,GlobalMaxPooling1D,MaxPooling1D,Activation,LSTM,Bidirectional,TimeDistributed,BatchNormalization
from keras import optimizers,regularizers
from keras.models import Model,model_from_json
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
seed = 7
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

# create concatenated vectors with unigram DBOW of 200 dimensions for each word in the vocabularies
def get_w2v_ugdbowdmm(comment, size, model_ug_dbow, model_ug_dmm):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in comment.split():
        try:
            vec += np.append(model_ug_dbow[word],model_ug_dmm[word]).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def main():
    # config
    optimizer=optimizers.Adam(lr=0.0005)
    MAX_SEQUENCE_LENGTH=64
    loss= "categorical_crossentropy"

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("LSTMCNN_ALL_best_weights.hdf5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

        sequences_test = tokenizer.texts_to_sequences(["toi yeu samsung"])
        x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

        yhat_cnn = loaded_model.predict(x_test_seq)

        print(yhat_cnn)


if __name__ == "__main__":
    main()