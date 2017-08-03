# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:31:45 2017

@author: Ajou
"""

## Setting up a SQLite database for data storage
## splite3 : https://docs.python.org/3.4/library/sqlite3.html

import sqlite3
import os
## create a connection
conn = sqlite3.connect('review.sqlite')
c = conn.cursor()
## Create table with three cols
c.execute('CREATE TABLE review_db(review TEXT, sentiment INTEGER, date TEXT)')
example1 = 'I love this movie'
c.execute(
        'INSERT INTO review_db(review, sentiment, date) VALUES(?,?,DATETIME("now"))',
        (example1,1))
example2 = 'I disliked this movie'
c.execute(
        'INSERT INTO review_db(review, sentiment, date) VALUES(?,?,DATETIME("now"))',
        (example2,0))
conn.commit()
conn.close()

## Check database table
conn = sqlite3.connect('review.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date BETWEEN '2015-01-01 00:00:00'"+
          " AND DATETIME('now')")
results = c.fetchall()
conn.close()
print(results)

