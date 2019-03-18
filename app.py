# -*- coding: utf-8 -*-
import sys
sys.path.append('./crawling-article/')

import os
import json
import pickle
import numpy as np

from keras import optimizers,regularizers
from keras.models import Model,model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

from flask import Flask
from flask import request
from flask import abort, redirect, url_for
from flask import jsonify
from flask_cors import CORS

from review_site import * 
from utils import *

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
        
        attributes = split_sentence_to_array(q)

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
            data = []

            for x in simpleAnalyzeOntology(q):
                sequences_test = tokenizer.texts_to_sequences([x[2]])
                x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

                yhat_cnn = loaded_model.predict(x_test_seq)

                sentimentResult = get_sentiment(yhat_cnn)

                temp = {}
                temp['attribute'] = x[0]
                temp['keywords'] = x[1]
                temp['sentence'] = x[2]
                temp['sentiment'] = sentimentResult[0]
                temp['score'] = str(sentimentResult[1])

                data.append(temp)

            return jsonify(data = data)
    else:
        abort(400)
        return 'ONLY ACCEPT POST REQUEST'

@app.route('/get-article', methods=['GET', 'POST'])
def getContentReviewSite():
    if(request.method == 'POST'):
        # Call object
        tinhte = Tinhte()
        vnreview = Vnreview()

        # Get json 
        url = request.get_json().get("q")
        content = ""
        
        if "tinhte" in url:
            content = tinhte.getArticle(url)
        elif "vnreview" in url:
            content = vnreview.getArticle(url)
            
        return content
    else:
        abort(400)
        return 'ONLY ACCEPT POST REQUEST'

def split_sentence_to_array(sentence):
    results = []

    with open('attributes.json', encoding="utf-8") as f:
        attributes = json.load(f)

        for key, values in attributes.items():
            temps = []

            for value in values:
                if(value in sentence):
                    temps.append(value)
            
            if(len(temps) != 0):
                results.append({key: temps})
    
    return results

def get_sentiment(modelResult):
    # modelResult is numpy narray

    # Get the max float in the narray -> what AI classify the sentiment
    maxElement = np.amax(modelResult)

    # Get the indices of maximum element in numpy array
    maxIndex = np.where(modelResult[0] == np.amax(modelResult[0]))

    sentiment = "positive"

    if(maxIndex == 0):
        # negative
        sentiment = "negative"
    elif(maxIndex == 1):
        # neutral
        sentiment = "neutral"

    if(maxElement > 0.6):
        # if AI result is greater than 0.6, then take the result
        return (sentiment, maxElement)
    else:
        # else make it as NEUTRAL
        # print("maxElement " + str(maxElement))
        return ("neutral", 0.5)
    
def separatingParagraph(paragraph):
    entities = readJson('entities.json')
    # extend all the entities in json file 
    valuesList = list(item for valueList in entities.values() for item in valueList)
    # Result is a list of each sentence separated by "." and ","
    result = list()
    # Split comment with character "."
    splitComment = paragraph.split(".")
    for comment in splitComment:
        # countKey var to check if there are more than 2 values of entities or not
        countKey = 0
        for value in valuesList:
            if value in comment:
                countKey += 1
                if countKey == 2: break
        
        # If there is only one entities in the comment then there is no need to separate the comment
        if countKey == 1:
            result += [comment]
        # else if there is more than 2 entities then we should separate the comment with ","
        elif countKey == 2:
            result += [x for x in comment.split(",")]

    return result

def mergeEntity(comment):
    result = set()
    entities = readJson('entities.json')
    
    # Get all the sentence from the paragraph
    sentenceList = separatingParagraph(comment)

    for sentence in sentenceList:
        # FlagExist for the case the we set only the sentence with one entity
        # flagExist = False
        for key in entities:
            for value in entities[key]:
                if value in sentence.lower():
                    # print(key + " - " + value + " - " + sentence)
                    result.add((key,sentence))
                    # flagExist = True
                    break
            # if flagExist is True: break
    
    # Return a tuple: 
    # first is key ("PIN", "MANHINH" , ...)
    # second is the comment sentence

    return result

def mergeAttribute(mergeEntityResult):
    result = list()
    attributes = readJson('attributes.json')
    
    for x in mergeEntityResult:\
        # x[0] is key ("PIN", "MANHINH" , ...)
        entity = x[0]
        attrList = list()
        for attr in attributes[entity]:
            if attr in x[1].lower():
                attrList.append(attr)
        if attrList != []:
            result.append((entity,attrList,x[1]))
                
    return result

def simpleAnalyzeOntology(paragraph):
    mergeEntityResult = mergeEntity(paragraph)
    result = mergeAttribute(mergeEntityResult)

    return result