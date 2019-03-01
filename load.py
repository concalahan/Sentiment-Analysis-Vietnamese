# -*- coding: utf-8 -*-
import sys
import os
import json

from keras import optimizers,regularizers
from keras.models import Model,model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
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
    return 'Index Page'

@app.route('/analyze-text', methods=['GET', 'POST'])
def analyzeIndex():
    K.clear_session()

    # { "q": "a" }

    if(request.method == 'POST'):
        # get params
        q = request.get_json().get("q")

        split_sentence_to_array(q)

        # config
        optimizer=optimizers.Adam(lr=0.0005)
        MAX_SEQUENCE_LENGTH=128
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

            sequences_test = tokenizer.texts_to_sequences([q])
            x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

            yhat_cnn = loaded_model.predict(x_test_seq)

            return jsonify(yhat_cnn.tolist())
    else:
        abort(400)
        return 'ONLY ACCEPT POST REQUEST'

def split_sentence_to_array(sentence):
    results = []

    with open('attributes.json') as f:
        attributes = json.load(f)

        for key, values in attributes.items():
            temps = []

            for value in values:
                if(value in sentence):
                    # print(key + "/" + value)
                    temps.append(value)
            
            if(len(temps) != 0):
                results.append({key: temps})

    print(results)
    return results