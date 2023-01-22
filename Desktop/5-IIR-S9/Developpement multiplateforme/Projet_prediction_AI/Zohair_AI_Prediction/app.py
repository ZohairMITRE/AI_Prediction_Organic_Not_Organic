from __future__ import division, print_function
# coding=utf-8
import numpy as np 
import os


import keras
from keras.models import load_model
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, redirect, url_for, request, render_template 
from werkzeug.utils import secure_filename 

app = Flask(__name__) 

MODEL_PATH = 'models/bestmodel.hdf5' 

model = load_model(MODEL_PATH) 

# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    img1 = keras.utils.load_img(img_path, target_size=(155, 155)) 
    img1  = keras.utils.img_to_array(img1) 
    img1 = np.expand_dims(img1, axis=0) 
    result = model.predict(img1/255) 

    return result 


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'TEST', secure_filename( f.filename)) 
        f.save(file_path) 

        preds = model_predict(file_path, model)
        acc_recyclable =  preds[0][1]*100 
        acc_organic =  preds[0][0]*100 

        if   preds[0][1] < 0.5:  
            return f"Le contenu est predicter comme Organic avec une precision de {acc_organic:.2f}%" 
        else:
            return f"Le contenu est predicter comme recyclable avec une precision de {acc_recyclable:.2f}%"
   
    return None


if __name__ == '__main__':
    app.run(debug=True)

