# import the necessary package
import os
import sys
import tensorflow as tf
from keras.models import load_model
from flask import *
import numpy as np
import cv2

from PIL import Image
from keras.preprocessing.image import load_img
from tensorflow.python.keras.backend import set_session

# create an instance of Flask
app = Flask(__name__)  

global sess
global graph

# saved model path
MODEL_PATH = 'models/model.h5'

sess = tf.Session() 
set_session(sess)

graph = tf.get_default_graph()

# load model and compile
model=tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

# predict the output of the image
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.array(img)/255.0
    img = img.reshape(-1, 224, 224,3)
    
    with graph.as_default():
        set_session(sess)
        prediction = model.predict(img)
        return prediction


@app.route('/')  
def upload():  
    return render_template("index.html")  
 
@app.route('/', methods = ['POST'])  
def success():
    classes = ['covid','normal']

    # upload file from browser 
    if request.method == 'POST':  
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', f.filename)
        f.save(file_path)
        
        # make prediction with the model
        prediction = model_predict(file_path, model)
        
        label=prediction.argmax(axis=-1)

        val = abs(float(prediction[0][0])-float(prediction[0][1]))

        # predicted label
        predicted_class = classes[label[0]]

        print("prediction is",predicted_class,file=sys.stderr)

        if(predicted_class == 'covid'):
            return render_template("positive.html")
        else:
            return render_template("negative.html")
            
    return None  
  
if __name__ == '__main__':  
    app.run(debug = False)  
