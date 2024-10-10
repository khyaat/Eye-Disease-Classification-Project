
import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load the trained model
model = load_model("my_model.h5")

# Resize the image to the expected size
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/result', methods=["POST"])
def result():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction)
        classes = {0: 'Bulging_Eyes', 1: 'Cataracts', 2: 'Crossed_Eyes', 3: 'Normal', 4: 'Uveitis'}
        result = classes[prediction_class]

        return render_template('result.html', pred=result)

if __name__ == "__main__":
    app.run(debug=True)