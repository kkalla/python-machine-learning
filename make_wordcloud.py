# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:12:16 2017

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os

## Reading datasets
pwd = %pwd
datadir = os.path.join(pwd,'text_data')
text_data = []
for files in os.listdir(datadir):
    text = open(os.path.join(datadir,files)).read()
    text_data.append(text)

## Preprocessing kor ver
text_kor = []
for i in range(0,len(text_data)):
    text = text_data[i]
    text2 = re.sub('<[^>]*>','',text)
    text3 = re.sub('[a-zA-Z]','',text2)
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    text4 = hangul.sub('',text3)
    text5 = re.sub('[:punct:]','',text4)
    text6 = re.sub('  ',' ',text5)
    text_kor.append(text6)

print(text_kor[0])

## Preprocessing En ver
text_en = []
for i in range(0,len(text_data)):
    text = text_data[i]
    text2 = re.sub('<[^>]*>','',text)
    text3 = re.sub('[\(\)-=.#/?:$}*\'\&_]','',text2)
    hangul = re.compile('[ㄱ-ㅣ가-힣]+')
    text4 = hangul.sub('',text3)
    text5 = re.sub('[0-9]','',text4)
    text6 = re.sub('[`]|[{]','',text5)
    text7 = re.sub('[\W]+',' ',text6)
    text_en.append(text7.lower())

print(text_en[0])

## Making wordcloud
from konlpy.tag import Hannanum
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import itertools
#Korean
t = Hannanum()
tokens_ko = []
for i in range(0,len(text_kor)):
    text = text_kor[i]
    tokens_ko.append(t.nouns(text))
tokens_ko2 = list(itertools.chain(*tokens_ko))
tokens_ko2 = [w for w in tokens_ko2 if len(w)>1]
stopwords_ko = ['악보','마디','도입부','일월화수목금토','사용','등등','브랜치',
             '이용','방법','부분']
tokens_ko2 = [w for w in tokens_ko2 if w not in stopwords_ko]
count = Counter(tokens_ko2).most_common(100)
print(count)

corpus = dict(count)


wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/NanumGothic.ttf',
                      background_color='white',colormap='RdYlBu')\
.generate_from_frequencies(corpus)
wordcloud.words_

plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


#nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokens_en = []
for i in range(0,len(text_en)):
    text = text_en[i]
    tokens_en.append([w for w in text.split() if w not in stop])
    
tokens_en2 = list(itertools.chain(*tokens_en))
print(tokens_en2)
print(Counter(tokens_en2).most_common(100))
wordcloud2 = WordCloud(font_path='/usr/share/fonts/truetype/NanumGothic.ttf')\
.generate_from_frequencies(dict(Counter(tokens_en2).most_common(100)))
wordcloud2.words_

plt.figure(figsize=(12,12))
plt.imshow(wordcloud2,interpolation='bilinear')
plt.axis('off')
plt.show()

## Image masking
#from PIL import Image
#image = np.array(Image.open('images.jpg'))
#wordcloud2 = WordCloud(background_color='white',mask=image)\
#.generate_from_frequencies(dict(Counter(tokens_en2).most_common(1000)))
#wordcloud2.words_
#
#plt.figure(figsize=(12,12))
#plt.imshow(wordcloud2,interpolation='bilinear')
#plt.axis('off')
#plt.show()
#
#
#plt.figure(figsize=(12,12))
#plt.imshow(image,cmap=plt.cm.gray,interpolation='bilinear')
#plt.axis('off')
#plt.show()