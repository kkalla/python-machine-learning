# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:30:46 2017

@author: Ajou
"""

import pandas as pd
ratings = pd.read_table('data/movielens/ratings.dat',sep='::',header=None)
users = pd.read_table('data/movielens/users.dat',sep='::',header=None)
movies = pd.read_table('data/movielens/movies.dat',sep='::',header=None)

ratings.columns = ['userId','movieId','rating','timestamp']
users.columns = ['userId','gender','age','occupation','zip_code']
movies.columns = ['movieId','title','genres']

data = pd.merge(pd.merge(ratings,users),movies)
mean_ratings = data.pivot_table('rating',index='title',
                                columns='gender',aggfunc='mean').dropna()

rating_by_title = data.groupby('title').size()

active_titles = rating_by_title[rating_by_title >= 250]

mean_ratings2=mean_ratings.iloc[active_titles]

top_female_ratings = mean_ratings2.sort_index(by='F',ascending=None)
mean_ratings2['diff'] = mean_ratings2['M']-mean_ratings2['F']
sorted_by_diff = mean_ratings2.sort_index(by='diff',ascending=False)
sorted_by_diff[:5]
aaa = sorted_by_diff[::-1][:15] ## [::-1] - 행 순서 뒤집기

###
rating_std = data.groupby('title')['rating']
rating_std.std()
