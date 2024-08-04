from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.vgg16 as keras_applications
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the pre-trained model and index
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Load precomputed features and labels
train_features_flat = np.load('data/train_features.npy')
train_labels_flat = np.load('data/train_labels.npy')

# Cargar las rutas de las imágenes
image_paths = np.load('data/image_paths.npy', allow_pickle=True)


# Fit the NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_features_flat)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def extract_features(image):
    features = model.predict(image)
    return features.flatten().reshape(1, -1)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Función para guardar la imagen subida
def save_image(file):
    if file and file.filename:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return filename
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(request.url)

    # Process the uploaded image
    image = Image.open(image_file)
    image = preprocess_image(image)
    query_features = extract_features(image)

    # Find the nearest neighbors
    distances, indices = nn_model.kneighbors(query_features)
    indices = indices[0]
    print("Indices of nearest neighbors:", indices)
    print("Distances to nearest neighbors:", distances)
    
     # Save the uploaded image and get its path
    uploaded_image_filename = save_image(image_file)
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_filename)


    # Obtener las rutas de las imágenes de los vecinos más cercanos
    neighbor_images = [os.path.relpath(image_paths[idx], 'static').replace("\\", "/") for idx in indices]
        
    print(neighbor_images)    
        
    return render_template('results.html', uploaded_image_path=uploaded_image_path, neighbor_images=neighbor_images)

if __name__ == '__main__':
    app.run(debug=True)
