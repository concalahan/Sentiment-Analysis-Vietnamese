# -*- coding: utf-8 -*-
import sys
import codecs
import re
import pandas as pd  
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

if sys.stdout.encoding != 'utf8':
  sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf8':
  sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer, 'strict') 

# for data cleaning
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'

short_words_dict = {
    "ko":"không",
    "k":"không",
    "1":"một",
    "2":"hai",
    "3":"ba",
    "4":"bốn",
    "5":"năm",
    "6":"sáu",
    "7":"bảy",
    "8":"tám",
    "9":"chín",
    "10":"mười",
    "đc":"được",
    "dc":"được"
}

short_words_pattern = re.compile(r'\b(' + '|'.join(short_words_dict.keys()) + r')\b')

short_words_dict_2 = {}

with open("./short_word.csv", mode='r', encoding='utf8') as input_file:
    rows = []
    for row in input_file:
        rows.append(row.rstrip('\n').split(","))

    for values in rows[1:]:
        short_words_dict_2[values[0].lower()] = values[1].lower()

short_words_pattern_2 = re.compile(r'\b(' + '|'.join(short_words_dict_2.keys()) + r')\b')

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def main():
    cols = ['comment','label']
    df = pd.read_csv("./nckh.comments.csv",header=None, names=cols, encoding='utf8')

    # delete column header
    df.drop(0, inplace=True)

    clean_comment_texts = []

    # for check duplicate
    temp_comments = []

    for comment in df.comment:
        temp = comment_cleaner(comment)

        if(not temp.isspace() and temp not in temp_comments):
            temp_comments.append(temp)
            clean_comment_texts.append(temp)
    
    # add pre_clean_len column
    df['pre_clean_len'] = [len(t) for t in df.comment]

    clean_df = pd.DataFrame(clean_comment_texts, columns=['comment'])
    clean_df['label'] = df.label
    
    clean_df.to_csv('clean_comments.csv',encoding='utf-8')
    csv = 'clean_comments.csv'
    my_df = pd.read_csv(csv,index_col=0)
    my_df.head()

def delete_char(text):
    # https://stackoverflow.com/questions/3411771/multiple-character-replace-with-python
    # a) 1000000 loops, best of 3: 1.47 μs per loop

    filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\'' #1234567890
    
    for c in filters:
        text = text.replace(c, r' ')
    
    return text

def comment_cleaner(text):
    # HTML decoding
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()

    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped

    #  ‘@’mention and hashtag / numbers
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)

    # .,!?
    char_stripped = delete_char(stripped)

    lower_case = char_stripped.lower()

    short_words_handled = short_words_pattern.sub(lambda x: short_words_dict[x.group()], lower_case)

    short_words_handled_2 = short_words_pattern_2.sub(lambda x: short_words_dict_2[x.group()], short_words_handled)

    emoji_handled = emoji_pattern.sub(r'', short_words_handled_2)

    return emoji_handled

if __name__ == "__main__":
    main()