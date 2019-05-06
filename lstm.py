# -*- coding: utf-8 -*-
import sys
import codecs
import io
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from gensim.models import Doc2Vec
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import utils
from sklearn.preprocessing import scale
from keras.layers import Embedding,Conv1D,CuDNNLSTM,Input,GlobalMaxPooling1D,MaxPooling1D,Activation,LSTM,Bidirectional,TimeDistributed,BatchNormalization
from keras import optimizers,regularizers
from keras.models import Model,model_from_json
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import pickle
seed = 7

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

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def main():
    cores = multiprocessing.cpu_count()

    cols = ['text','label']
    my_df = pd.read_csv("./clean_comments.csv",header=0, names=cols, encoding='utf8').dropna()

    x = my_df['text']
    y = my_df['label']

    SEED = 2000
    
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=.02, random_state=SEED)

    y_train=tf.keras.utils.to_categorical(np.asarray(y_train), num_classes= 3)
    y_validation=tf.keras.utils.to_categorical(np.asarray(y_validation), num_classes= 3)

    print("Loading the model...")
    
    model_ug_dbow = Doc2Vec.load('model/d2v_model_ug_dbow.doc2vec')
    model_bg_dmm = Doc2Vec.load('model/d2v_model_bg_dmm.doc2vec')
    
    # CBOW: w2v_model_ug_cbow.word2vec
    # SKIP GRAM: w2v_model_ug_sg.word2vec

    # DBOW Unigram: d2v_model_ug_dbow.doc2vec 0.7285714285714285
    # DBOW Bigram: d2v_model_bg_dbow.doc2vec 0.7267857142857143
    # DMM Bigram: d2v_model_bg_dmm.doc2vec 0.7232142857142857
    
    train_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_bg_dmm, x_train, 200)
    validation_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_bg_dmm, x_validation, 200)

    clf = LogisticRegression()
    clf.fit(train_vecs_ugdbow_tgdmm, y_train)

    print(clf.score(train_vecs_ugdbow_tgdmm, y_train))
    print(clf.score(validation_vecs_ugdbow_tgdmm, y_validation))

    embedding_size=200
    fully_connected_layers= [1000,1000]
    conv_layers=[500,3,1]
    dropout_p= 0.1
    optimizer=optimizers.Adam(lr=0.0005)
    loss= "categorical_crossentropy"
    input_size= MAX_SEQUENCE_LENGTH
    num_of_classes=3

    inputs = Input(shape=(input_size,), name='sent_input', dtype='int64')

    x = Embedding(20000, embedding_size, weights=[train_vecs_ugdbow_tgdmm], input_length=input_size, trainable=True)(inputs)

    # Convolution Layers
    x = Conv1D(conv_layers[0], [conv_layers[1]], padding='same', strides=conv_layers[2])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # LSTM Layer
    x = Bidirectional(LSTM(250, return_sequences=True))(x)
    x = Activation('relu')(x)

    # Fully connected layers
    for fl in fully_connected_layers:
        x = Dense(fl)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_p)(x)

    # Maxpool and Flaten Layers        
    x = MaxPooling1D(5,strides=2)(x)
    x = Flatten()(x)

    # Output layer
    predictions = Dense(num_of_classes, activation='softmax')(x)

    # Build and compile model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

    filepath="LSTMCNN_ALL_best_weights.hdf5"

    earlystopper = EarlyStopping(patience=30, verbose=1)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    reduce_lr = ReduceLROnPlateau( monitor='val_acc',factor=0.1,
                                patience=1, min_lr=5e-06, verbose=1, mode='auto')

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.fit(x_train_seq, y_train, batch_size=16, epochs=5,
                        validation_data=(x_val_seq, y_validation_and_test), callbacks = [reduce_lr,checkpoint,earlystopper])

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])

    # load weights into new model
    loaded_model.load_weights("LSTMCNN_ALL_best_weights.hdf5")
    print("Loaded model from disk")
    
    sequences_test = tokenizer.texts_to_sequences(x_validation_and_test)
    x_test_seq = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    yhat_cnn = loaded_model.predict(x_test_seq)

    scores = loaded_model.evaluate(x_test_seq, y_validation_and_test, verbose=1)

    print("----------final score----------")
    print(scores)


if __name__ == "__main__":
    main()