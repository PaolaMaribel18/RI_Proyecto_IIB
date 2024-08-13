from flask import Flask, render_template, request, redirect
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageFilter
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model and index
print("Loading model...")
start_time = time.time()

# Configure InceptionV3 model
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
#model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Load precomputed features, labels, and images
train_features_flat = np.load('data/train_features.npy')
train_labels_flat = np.load('data/train_labels.npy')
train_img_flat = np.load('data/train_img.npy')
name_classes = np.load('data/name_classes.npy', allow_pickle=True)

# Load image paths
#image_paths = np.load('data/image_paths.npy', allow_pickle=True)
#print("Primeras 2 rutas:")
#print(image_paths[:2])

# Fit the NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='cosine').fit(train_features_flat)

print(f"Model and data loading took {time.time() - start_time:.2f} seconds")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255.0  
    image = tf.expand_dims(image, axis=0)
    return image

def extract_features(image):
    features = model.predict(image)
    return features.flatten().reshape(1, -1)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    distances = distances[0] 
    indices = indices[0]
    print("Indices of nearest neighbors:", indices)
    print("Distances to nearest neighbors:", distances)
    
    # Retrieve nearest images and labels using indices
    nearest_images = [train_img_flat[i] for i in indices.flatten()]
    nearest_labels = [train_labels_flat[i] for i in indices.flatten()]

    # Save nearest images to a temporary folder
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])

    for i, img in enumerate(nearest_images):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))  # Convert back to uint8
        img_pil.save(os.path.join(app.config['RESULT_FOLDER'], f'neighbor_{i}.jpg'))

    # Generate paths for the nearest images
    neighbor_images_data = [{
        'url': f'results/neighbor_{i}.jpg',  # Solo la ruta relativa
        'label': name_classes[nearest_labels[i]],  # Nombre de la clase
        'distances': distances[i]  # Distancia al vecino más cercano
    } for i in range(len(nearest_images))]

    # Print the generated URLs and labels for validation
    print("Generated URLs and labels for neighbor images:")
    for image_data in neighbor_images_data:
        print(f"URL: {image_data['url']}, Label: {image_data['label']}")

    # Pass neighbor_images_data and name_classes to the template
    return render_template('results.html', uploaded_image_path=uploaded_image_path, neighbor_images_data=neighbor_images_data, name_classes=name_classes, distances=distances)

    return redirect(request.url)
if __name__ == '__main__':
    app.run(debug=True)