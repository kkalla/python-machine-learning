# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:53:05 2017

@author: Ajou
"""

from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('first_app.html')
if __name__ == '__main__':
    app.run()