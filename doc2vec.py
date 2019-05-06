# -*- coding: utf-8 -*-
import sys
import codecs
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import utils

if sys.stdout.encoding != 'utf8':
  sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf8':
  sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer, 'strict') 

# doc2vec training is completely unsupervised and thus there is no need to hold out any data, as it is unlabelled
def labelize_comments_ug(comments,label):
    result = []
    prefix = label
    for i, t in zip(comments.index, comments):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result

def labelize_comments_bg(comments,label,bigram):
    result = []
    prefix = label
    for i, t in zip(comments.index, comments):
        result.append(LabeledSentence(bigram[t.split()], [prefix + '_%s' % i]))
    return result

def labelize_tweets_tg(tweets,label,bigram,trigram):
    result = []
    prefix = label
    for i, t in zip(tweets.index, tweets):
        result.append(LabeledSentence(trigram[bigram[t.split()]], [prefix + '_%s' % i]))
    return result

def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs

def get_concat_vectors(model1,model2, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = np.append(model1.docvecs[prefix],model2.docvecs[prefix])
        n += 1
    return vecs

def qdsub(s):
    return float(re.sub('\,', '.', str(s)[2:-1]))

def main():
    cores = multiprocessing.cpu_count()

    cols = ['comment','label']
    my_df = pd.read_csv("./clean_comments.csv",header=None, names=cols, encoding='utf8')
    my_df.dropna(inplace=True)
    my_df.reset_index(drop=True,inplace=True)
    my_df.info()

    x = my_df['comment']
    y = my_df['label']

    SEED = 2000
    x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

    # detect the frequently used phrase and connect them together with underbar in the middle.
    tokenized_train = [t.split() for t in x_train]
    phrases = Phrases(tokenized_train)
    bigram = Phraser(phrases)

    tg_phrases = Phrases(bigram[tokenized_train])
    trigram = Phraser(tg_phrases)

    all_x = pd.concat([x_train,x_validation,x_test])
    all_x_w2v = labelize_comments_ug(all_x, 'all')
    all_x_w2v_bg = labelize_comments_bg(all_x, 'all', bigram)
    all_x_w2v_tg = labelize_tweets_tg(all_x, 'all',bigram,trigram)

    # model_ft = FastText.load_fasttext_format("wiki.vi/cc.vi.300.bin")
    # model_ft.build_vocab([x.words for x in tqdm(all_x_w2v)])

    # for epoch in range(30):
    #     model_ft.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    #     model_ft.alpha -= 0.002
    #     model_ft.min_alpha = model_ft.alpha

    # model_ft.save('w2v_model_fasttext_cbow.word2vec')
    
    # --------------------- Skip gram word2vec ---------------------
    # --------------------------------------------------------------

    # model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

    # for epoch in range(30):
    #     model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    #     model_ug_sg.alpha -= 0.002
    #     model_ug_sg.min_alpha = model_ug_sg.alpha

    # model_ug_sg.save('model/w2v_model_ug_sg.word2vec')

    # --------------------- CBOW word2vec ---------------------
    # ---------------------------------------------------------

    model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

    for epoch in range(30):
        model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
        model_ug_cbow.alpha -= 0.002
        model_ug_cbow.min_alpha = model_ug_cbow.alpha

    model_ug_cbow.save('model/w2v_model_ug_cbow.word2vec')

    # --------------------- DBOW doc2vec ---------------------
    # --------------------------------------------------------

    # model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

    # for epoch in range(30):
    #     model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    #     model_ug_dbow.alpha -= 0.002
    #     model_ug_dbow.min_alpha = model_ug_dbow.alpha
    
    # train_vecs_dbow = get_vectors(model_ug_dbow, x_train, 100)
    # validation_vecs_dbow = get_vectors(model_ug_dbow, x_validation, 100)

    # clf = LogisticRegression()
    # clf.fit(train_vecs_dbow, y_train)

    # print(clf.score(validation_vecs_dbow, y_validation))

    # model_ug_dbow.save('model/d2v_model_ug_dbow.doc2vec')

    # --------------------- DBOW Bigram ---------------------
    # --------------- Distributed Bag Of Words --------------
    # -------------------------------------------------------

    # model_bg_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_bg_dbow.build_vocab([x for x in tqdm(all_x_w2v_bg)])

    # for epoch in range(30):
    #     model_bg_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    #     model_bg_dbow.alpha -= 0.002
    #     model_bg_dbow.min_alpha = model_bg_dbow.alpha
 
    # train_vecs_dbow_bg = get_vectors(model_bg_dbow, x_train, 100)
    # validation_vecs_dbow_bg = get_vectors(model_bg_dbow, x_validation, 100)

    # clf = LogisticRegression()
    # clf.fit(train_vecs_dbow_bg, y_train)

    # print("Model BG DBOW")
    # print(clf.score(validation_vecs_dbow_bg, y_validation))

    # model_bg_dbow.save('model/d2v_model_bg_dbow.doc2vec')

    # --------------------- DMM Bigram ---------------------
    # ---------------(Distributed Memory Mean)--------------
    # ------------------------------------------------------

    # model_bg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_bg_dmm.build_vocab([x for x in tqdm(all_x_w2v_bg)])

    # for epoch in range(30):
    #     model_bg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_bg)]), total_examples=len(all_x_w2v_bg), epochs=1)
    #     model_bg_dmm.alpha -= 0.002
    #     model_bg_dmm.min_alpha = model_bg_dmm.alpha

    # train_vecs_dmm_bg = get_vectors(model_bg_dmm, x_train, 100)
    # validation_vecs_dmm_bg = get_vectors(model_bg_dmm, x_validation, 100)

    # clf_dmm_bg = LogisticRegression()
    # clf_dmm_bg.fit(train_vecs_dmm_bg, y_train)

    # print("Model BG DMM")
    # print(clf_dmm_bg.score(validation_vecs_dmm_bg, y_validation))

    # model_bg_dmm.save('model/d2v_model_bg_dmm.doc2vec')

    # for epoch in range(30):
    #     model_ug_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    #     model_ug_dmm.alpha -= 0.002
    #     model_ug_dmm.min_alpha = model_ug_dmm.alpha
        
    # train_vecs_dmm = get_vectors(model_ug_dmm, x_train, 100)
    # validation_vecs_dmm = get_vectors(model_ug_dmm, x_validation, 100)

    # model_tg_dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_tg_dmm.build_vocab([x for x in tqdm(all_x_w2v_tg)])



    # model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
    # model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

    # for epoch in range(30):
    #     model_tg_dmm.train(utils.shuffle([x for x in tqdm(all_x_w2v_tg)]), total_examples=len(all_x_w2v_tg), epochs=1)
    #     model_tg_dmm.alpha -= 0.002
    #     model_tg_dmm.min_alpha = model_tg_dmm.alpha

    # train_vecs_dmm_tg = get_vectors(model_tg_dmm, x_train, 100)
    # validation_vecs_dmm_tg = get_vectors(model_tg_dmm, x_validation, 100)

    # for epoch in range(30):
    #     model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    #     model_ug_dbow.alpha -= 0.002
    #     model_ug_dbow.min_alpha = model_ug_dbow.alpha
        
    # train_vecs_dbow = get_vectors(model_ug_dbow, x_train, 100)
    # validation_vecs_dbow = get_vectors(model_ug_dbow, x_validation, 100)

    # clf = LogisticRegression()
    # clf.fit(train_vecs_dbow, y_train)
    # clf.score(validation_vecs_dbow, y_validation)

    # clf.fit(train_vecs_dbow_bg, y_train)
    # clf.score(validation_vecs_dbow_bg, y_validation)

    # clf.fit(train_vecs_dmm, y_train)
    # clf.score(validation_vecs_dmm, y_validation)

    # print(clf.score(validation_vecs_dmm, y_validation))

    # for key, value in model_ug_dbow.similar_by_word("vui", topn=10):
    #     print(key)
    #     print(value)

    # model_ug_dbow.save('d2v_model_ug_dbow.doc2vec')

if __name__ == "__main__":
    main()