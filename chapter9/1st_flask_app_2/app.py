# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:07:13 2017

@author: Ajou
"""

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

class HelloForm(Form):
    sayhello = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = HelloForm(request.form)
    return render_template('first_app.html',form=form)

@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['sayhello']
        return render_template('hello.html',name=name)
    return render_template('first_app.html',form=form)

if __name__ =='__main__':
    app.run()
    