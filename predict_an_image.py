# import the necessary packages
import tensorflow as tf
import numpy as np
from keras.models import *
import cv2

#load model and print the summary of the network
model=tf.keras.models.load_model("model.h5")
model.summary()

# define output labels
class_labels = ['covid', 'normal']
im = cv2.imread(r"D:\College\Projects\COR_X_Analyser\repo-keras-covid-19\repository\dataset\covid\lancet-case2a.jpg")

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
