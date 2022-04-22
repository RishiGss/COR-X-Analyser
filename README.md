# COR-X-Analyser

COR-X Analyser is an easy to use web application which helps in detecting and predicting COVID19 with X-ray images. This tool suffices the need of doctors in detecting corona cases.

This model was built using transfer learning with pre-trained VGG16 network in Keras framework. For the ease of use, it was deployed as a web application using Flask API.

### <strong>Steps to reproduce this project:</strong>

1. Clone this repository.
2. Create a python/conda enironment.
3. Install the necessary packages in the environment with **`requirements.txt`**.
4. Run **`train_covid19.py`** to train the network and save the model.
5. The prediction results for a single image can be tested with **`predict_an_image.py`**.
6. For web application, run **`cor_x.py`**.
7. Go to `http://127.0.0.1:5000/` in a browser.
8. Upload an image. The uploaded images from the browser will be saved in `uploads` folder and will be used for prediction.
9. Click on **Predict!** to check the predicted output.
10. If needed, host the web application on a cloud server.

### Screenshot of web app
![Screenshot (1129)](https://user-images.githubusercontent.com/37014747/164711825-ccd1dd5d-9109-4ca0-8b40-579c8451ee22.png)


To have a demo of COR-X Analyser, watch this [video.](https://www.youtube.com/watch?v=u-bw8SZFfLs "COR X Analyser")
