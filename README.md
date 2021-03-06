# COR-X-Analyser
Web application to predict COVID-19 result using X-rays. Built using Keras framework and Flask API.

COR-X Analyser is an easy to use web application which helps in detecting and predicting COVID19 with X-ray images. This tool suffices the need of doctors in detecting corona cases.

This model was built using transfer learning with pre-trained VGG16 network in Keras framework. For the ease of use, it was deployed as a web application using Flask API.

### <strong>Steps to reproduce this project:</strong>
1. Clone this repository.
2. Create a python/conda enironment.
3. Install the necessary packages in the environment with **requirements.txt**.
4. Run **train_covid19.py** to train the network and save the model.
5. The prediction results for a single image can be tested with **predict_an_image.py.**
6. Create a folder named **uploads** in flask_app folder, where the uploaded images from the browser will be saved and used for prediction.
7. Run **cor_x.py**.
8. Go to ```http://127.0.0.1:5000/``` in a browser.
9. Upload an image and click on **Predict!**.
10. If needed, host the web application on a cloud server.

### Screenshot of web app
![Screenshot (8)](https://user-images.githubusercontent.com/37014747/93715182-5d22b800-fb85-11ea-8d64-67bb5a1263ec.png)


To have a demo of COR-X Analyser, watch this [video.](https://www.youtube.com/watch?v=u-bw8SZFfLs "COR X Analyser")
