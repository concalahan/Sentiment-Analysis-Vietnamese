# -*- coding: utf-8 -*-
import sys
import codecs
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
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

def main():
    cores = multiprocessing.cpu_count()

    cols = ['text','label']
    my_df = pd.read_csv("./clean_comments_2.csv",header=0, names=cols, encoding='utf8').dropna()

    x = my_df['text']
    y = my_df['label']

    SEED = 2000
    
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)

    y_train=tf.keras.utils.to_categorical(np.asarray(y_train), num_classes= 3)
    y_validation_and_test=tf.keras.utils.to_categorical(np.asarray(y_validation_and_test))

    print("Loading the model...")
    model_ug_cbow = KeyedVectors.load('model/w2v_model_ug_cbow.word2vec')
    model_ug_sg = KeyedVectors.load('model/w2v_model_ug_sg.word2vec')
    
    print("Tokenizing the model...")

    # a number of vocabularies you want to use
    tokenizer = Tokenizer(num_words=20000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890',
                                   lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(x_train)

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(x_train[:5])

    sequences = tokenizer.texts_to_sequences(x_train)

    embeddings_index = {}
    for w in model_ug_cbow.wv.vocab.keys():
        embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
    print('Found %s word vectors.' % len(embeddings_index))

    
    # first five entries of the original train data
    # for x in x_train[:5]:
    #     print(x)

    # Actual number presentation
    # print(sequences[:5])

    length = []
    for x in x_train:
        length.append(len(x.split()))

    Sumlength=0
    for x in sequences:
        length.append(len(x))
        Sumlength=Sumlength+len(x)

    AvarageLength=round(Sumlength/len(x_train))

    print("-----------------------")
    print("REAL MAX_SEQUENCE_LENGTH {0}".format(max(length)))
    print(Sumlength)
    print(len(x_train))
    print(AvarageLength)
    print("-----------------------")

    # base on AvarageLength to define MAX_SEQUENCE_LENGTH
    MAX_SEQUENCE_LENGTH=64
    if((max(length)>128)):
        MAX_SEQUENCE_LENGTH=128
    if((max(length)<64)):
        MAX_SEQUENCE_LENGTH=64

    print("Padding the model with max length is {0}...".format(MAX_SEQUENCE_LENGTH))

    x_train_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    sequences_val = tokenizer.texts_to_sequences(x_validation_and_test)
    x_val_seq = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)

    # only care about 20000 most frequent words in the training set
    num_words = 20000
    embedding_matrix = np.zeros((num_words, 200))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # word at 2221 is 'core'
    print("Checking the word at matrix 2192 is core")
    print(np.array_equal(embedding_matrix[2192] ,embeddings_index.get('core')))

    embedding_size=200
    fully_connected_layers= [1000,1000]
    conv_layers=[500,3,1]
    dropout_p= 0.1
    optimizer=optimizers.Adam(lr=0.0005)
    loss= "categorical_crossentropy"
    input_size= MAX_SEQUENCE_LENGTH
    num_of_classes=3

    inputs = Input(shape=(input_size,), name='sent_input', dtype='int64')

    x = Embedding(num_words, embedding_size, weights=[embedding_matrix], input_length=input_size, trainable=True)(inputs)

    # Convolution Layers
    x = Conv1D(conv_layers[0], [conv_layers[1]], padding='same', strides=conv_layers[2])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # LSTM Layer
    x = Bidirectional(LSTM(250, return_sequences=True))(x)
    x = Activation('relu')(x)

    # Maxpool and Flaten Layers        
    x = MaxPooling1D(5,strides=2)(x)
    x = Flatten()(x)

    # Fully connected layers
    for fl in fully_connected_layers:
        x = Dense(fl)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_p)(x)

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
    model.fit(x_train_seq, y_train, batch_size=16, epochs=1,
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