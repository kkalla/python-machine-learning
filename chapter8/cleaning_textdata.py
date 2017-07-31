# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:58:15 2017

@author: user
"""

## Cleaning text data
import re
def preprocessor(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(\D\P)',text)
    text = re.sub('[\W]+',' ',text.lower()).join(emoticons).replace('-','')
    return text
