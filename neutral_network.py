# -*- coding: utf-8 -*-
import sys
import codecs
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.preprocessing import scale

seed = 7

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping

if sys.stdout.encoding != 'utf8':
  sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf8':
  sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer, 'strict') 

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
    my_df = pd.read_csv("./clean_comments.csv",header=None, names=cols, encoding='utf8')
    my_df.dropna(inplace=True)
    my_df.reset_index(drop=True,inplace=True)
    my_df.info()
    new_header = my_df.iloc[0] #grab the first row for the header
    my_df = my_df[1:] #take the data less the header row
    my_df.columns = new_header #set the header row as the df header

    x = my_df['text']
    y = my_df['label']

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    model_ug_dbow = Doc2Vec.load('d2v_model_ug_dbow.doc2vec')
    model_ug_dmm = Doc2Vec.load('d2v_model_ug_dmm.doc2vec')
    model_tg_dmm = Doc2Vec.load('d2v_model_tg_dmm.doc2vec')
    model_ug_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_ug_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model_tg_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # Not use the concatenated vectors of different n-grams, since they will not consist of the same vocabularies
    train_vecs_w2v_dbowdmm_sum = np.concatenate([get_w2v_ugdbowdmm(z, 200,model_ug_dbow,model_ug_dmm) for z in x_train])
    validation_vecs_w2v_dbowdmm_sum = np.concatenate([get_w2v_ugdbowdmm(z, 200,model_ug_dbow,model_ug_dmm) for z in x_validation])

    train_vecs_w2v_dbowdmm_sum_s = scale(train_vecs_w2v_dbowdmm_sum)
    validation_vecs_w2v_dbowdmm_sum_s = scale(validation_vecs_w2v_dbowdmm_sum)

    # train_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_tg_dmm, x_train, 200)
    # validation_vecs_ugdbow_tgdmm = get_concat_vectors(model_ug_dbow,model_tg_dmm, x_validation, 200)

    clf = LogisticRegression()
    clf.fit(train_vecs_w2v_dbowdmm_sum_s, y_train)
    clf.score(validation_vecs_w2v_dbowdmm_sum_s, y_validation)

    # print(clf.score(train_vecs_ugdbow_tgdmm, y_train))
    print(clf.score(validation_vecs_w2v_dbowdmm_sum_s, y_validation))

    # Model example 1: 1 hidden layer with 64 hidden nodes
    # filepath="model/d2v_09_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max') 
    # callbacks_list = [checkpoint, early_stop]
    # np.random.seed(seed)
    # model_d2v_09_es = Sequential()
    # model_d2v_09_es.add(Dense(256, activation='relu', input_dim=200))
    # model_d2v_09_es.add(Dense(256, activation='relu'))
    # model_d2v_09_es.add(Dense(256, activation='relu'))
    # model_d2v_09_es.add(Dense(1, activation='sigmoid'))
    # model_d2v_09_es.compile(optimizer='adam',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])

    # model_d2v_09_es.fit(train_vecs_w2v_dbowdmm, y_train,
    #                     validation_data=(validation_vecs_w2v_dbowdmm, y_validation), 
    #                     epochs=100, batch_size=32, verbose=2, callbacks=callbacks_list)

    # model_d2v_09_es.evaluate(x=validation_vecs_w2v_dbowdmm, y=y_validation)

if __name__ == "__main__":
    main()