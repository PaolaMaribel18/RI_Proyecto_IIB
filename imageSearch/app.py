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
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
app = Flask(__name__)

# Load the pre-trained model and index
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
# Configurar el modelo InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Load precomputed features and labels
train_features_flat = np.load('data/train_features.npy')
train_labels_flat = np.load('data/train_labels.npy')

# Imprimir la forma de los arrays
print("Forma de train_features:", train_features_flat.shape)
print("Forma de train_labels:", train_labels_flat.shape)

# Cargar las rutas de las imágenes
image_paths = np.load('data/image_paths.npy', allow_pickle=True)


# Fit the NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_features_flat)

def preprocess_image(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def extract_features(image):
    features = model.predict(image)
    return features.flatten().reshape(1, -1)

'''
def get_image_path_from_index(index, base_dir):
    """Construct image path from index."""
    all_image_paths = []
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if os.path.isdir(class_path):  # Check if it's a directory
            for image_file in os.listdir(class_path):
                all_image_paths.append(os.path.join(class_path, image_file))
    
    if index < len(all_image_paths):
        return all_image_paths[index]
    else:
        return None  # Handle the case where index is out of bounds
'''
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = set({'png', 'jpg', 'jpeg'})
# Función para guardar la imagen subida

from werkzeug.utils import secure_filename


def allowed_file(file):
    file=file.split(".")
    if file[1].lower() in ALLOWED_EXTENSIONS:
        return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    print(image_file,image_file.filename)
    filename=secure_filename(image_file.filename)
    print(filename)
    if image_file and allowed_file(filename):
        print("Archivo permitido")
        image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        


    uploaded_image_p = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
    uploaded_image_path = uploaded_image_p.replace("static/", "")
    print( "path",uploaded_image_path)


    # Process the uploaded image
    image = Image.open(image_file)
    image = preprocess_image(image)
    query_features = extract_features(image)

    # Find the nearest neighbors
    distances, indices = nn_model.kneighbors(query_features)
    indices = indices[0]
    print("Indices of nearest neighbors:", indices)
    print("Distances to nearest neighbors:", distances)
    
    # Obtener las rutas de las imágenes de los vecinos más cercanos
    neighbor_images = [os.path.relpath(image_paths[idx], 'static').replace("\\", "/") for idx in indices]
    
    print(neighbor_images)    
        
    return render_template('results.html', uploaded_image_path=uploaded_image_path, neighbor_images=neighbor_images)

if __name__ == '__main__':
    app.run(debug=True)