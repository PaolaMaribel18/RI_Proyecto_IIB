# Example query: Retrieve the 5 nearest neighbors for a test image
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


name_classes = np.load('data/name_classes.npy', allow_pickle=True)


# Cargar los datos
train_features_flat = np.load('data/train_features.npy')
train_labels_flat = np.load('data/train_labels.npy')
train_img_flat = np.load('data/train_img.npy')
test_img_flat = np.load('data/test_img.npy') 

# Fit the NearestNeighbors model
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(train_features_flat)

imagen_consulta =test_img_flat[2]
imagen_consulta = tf.expand_dims(imagen_consulta, axis=0)
plt.imshow(imagen_consulta[0])

# Load the VGG16 model with pretrained weights from ImageNet, without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a new model that outputs the feature maps
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)


query_features = model.predict([imagen_consulta]).flatten().reshape(1, -1)
# Find the nearest neighbors
distances, indices = nn_model.kneighbors(query_features)
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)


nearest_images =[]
nearest_labels = []
# Recuperar las imágenes de los vecinos más cercanos usando los índices
nearest_images = [train_img_flat[i] for i in indices.flatten()]
nearest_labels = [train_labels_flat[i] for i in indices.flatten()]
plt.figure(figsize=(15, 3))  # Ajustar el tamaño de la figura
for i, image in enumerate(nearest_images):
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title("Clase: " + name_classes[nearest_labels[i]] + "\nDistancia: " + str(round(distances[0][i], 2))+ "\nIndice: " + str(indices[0][i]))
    plt.axis('off')
plt.show()

query_label = train_labels_flat[2]  # Asumiendo que estás usando la imagen en test_img_flat[2] y tiene la etiqueta en train_labels_flat[2]
correct_predictions = sum([1 for label in nearest_labels if label == query_label])
precision = correct_predictions / len(nearest_labels)
print(f"Precisión en los 5 primeros resultados: {precision:.2f}")
