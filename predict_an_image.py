# import the necessary packages
import tensorflow as tf
import numpy as np
from keras.models import *
import cv2
from sys import argv

#load model and print the summary of the network
model=tf.keras.models.load_model("models/model.h5")
model.summary()

# define output labels
class_labels = ['covid', 'normal']
im = cv2.imread(argv[1])

# image pre-processing
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
im=cv2.resize(im,(224,224))
im=np.array(im)/255.0
im=im.reshape(-1,224,224,3)

# predict
out=model.predict(im)
lab=out.argmax(axis=-1)

print(out)
print(class_labels[lab[0]])
