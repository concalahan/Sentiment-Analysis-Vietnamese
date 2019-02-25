# -*- coding: utf-8 -*-
import sys
import os
import json

from keras import optimizers,regularizers
from keras.models import Model,model_from_json
from keras.preprocessing.sequence import pad_sequences
import pickle

from flask import Flask
from flask import request
from flask import abort, redirect, url_for
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # config
    optimizer=optimizers.Adam(lr=0.0005)
    MAX_SEQUENCE_LENGTH=64
    loss= "categorical_crossentropy"

    # load json and create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/LSTMCNN_ALL_best_weights.hdf5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

    # loading
    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

        sequences_test = tokenizer.texts_to_sequences(["toi yeu samsung"])
        x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

        yhat_cnn = loaded_model.predict(x_test_seq)

        print(yhat_cnn)

        return jsonify(yhat_cnn.tolist()) 

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