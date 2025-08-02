from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Dictionary for predictions with flower information and Wikipedia URLs
dic = {
    0: {
        'name': 'daisy',
        'info': 'Daisies symbolize innocence and purity. They are commonly found in meadows.',
        'image': 'daisy.jpg',
        'wiki': 'https://en.wikipedia.org/wiki/Bellis_perennis'
    },
    1: {
        'name': 'dandelion',
        'info': 'Dandelions are known for their yellow flowers and have many medicinal uses.',
        'image': 'dandelion.jpg',
        'wiki': 'https://en.wikipedia.org/wiki/Taraxacum'
    },
    2: {
        'name': 'rose',
        'info': 'Roses are a symbol of love and are widely used in perfumes and beauty products.',
        'image': 'rose.jpg',
        'wiki': 'https://en.wikipedia.org/wiki/Rose'
    },
    3: {
        'name': 'sunflower',
        'info': 'Sunflowers are known for turning their heads towards the sun and symbolize loyalty.',
        'image': 'sunflower.jpg',
        'wiki': 'https://en.wikipedia.org/wiki/Helianthus'
    },
    4: {
        'name': 'tulip',
        'info': 'Tulips are spring-blooming perennials and come in many vibrant colors.',
        'image': 'tulip.jpg',
        'wiki': 'https://en.wikipedia.org/wiki/Tulip'
    }
}

# Load the model
model = load_model('flower.h5')
model.make_predict_function()

# Function to predict the label of the uploaded image
def predict_label(img_path):
    i = image.load_img(img_path, target_size=(64, 64))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)  # Add batch dimension
    p = model.predict(i)
    class_idx = np.argmax(p, axis=1)[0]
    return dic[class_idx]

# Routes
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("tindex.html")

@app.route("/submit", methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        # Save the uploaded image
        img = request.files['my_image']
        img_path = os.path.join('static', img.filename)
        img.save(img_path)

        # Get prediction
        prediction_info = predict_label(img_path)

        # Pass the prediction, flower info, image, and wiki link to the template
        return render_template(
            "tindex.html",
            prediction=prediction_info['name'],
            img_path=img_path,
            flower_info=prediction_info['info'],
            prediction_image=prediction_info['image'],
            wiki_url=prediction_info['wiki']
        )

if __name__ == '__main__':
    # Make sure 'static' folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    app.run(debug=True)
