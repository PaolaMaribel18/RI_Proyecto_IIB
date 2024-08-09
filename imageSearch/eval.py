from sklearn.metrics import precision_score, recall_score, average_precision_score
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

train_features_flat=np.load('data/train_features.npy')
train_labels_flat=np.load('data/train_labels.npy')
test_img_flat=np.load('data/test_img.npy')
test_labels = np.load('data/test_labels.npy')

nn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine').fit(train_features_flat)

# Load the VGG16 model with pretrained weights from ImageNet, without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Definir el número de vecinos más cercanos para la evaluación
k = 5

# Evaluar el sistema sobre un conjunto de pruebas
precisions = []
recalls = []
average_distances = []
retrieval_times = []

for i in range(len(test_img_flat)):
    imagen_consulta = test_img_flat[i]
    imagen_consulta = tf.expand_dims(imagen_consulta, axis=0)
    
    # Extraer características de la imagen de consulta
    query_features = model.predict([imagen_consulta]).flatten().reshape(1, -1)
    
    # Medir el tiempo de recuperación
    start_time = time.time()
    
    # Encontrar los vecinos más cercanos
    distances, indices = nn_model.kneighbors(query_features)
    
    # Medir el tiempo de recuperación
    retrieval_times.append(time.time() - start_time)
    
    # Obtener las etiquetas de los vecinos más cercanos
    nearest_labels = [train_labels_flat[idx] for idx in indices.flatten()]
    
    # Calcular la precisión y el recall para esta consulta
    true_label = test_labels[i]
    precision = precision_score([true_label]*k, nearest_labels, average='macro')
    recall = recall_score([true_label]*k, nearest_labels, average='macro')
    
    precisions.append(precision)
    recalls.append(recall)
    
    # Calcular la distancia promedio
    average_distance = np.mean(distances)
    average_distances.append(average_distance)

# Calcular la precisión promedio, el recall promedio, y la distancia promedio
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_average_distance = np.mean(average_distances)
mean_retrieval_time = np.mean(retrieval_times)

print(f'Precision promedio: {mean_precision}')
print(f'Recall promedio: {mean_recall}')
print(f'Distancia promedio: {mean_average_distance}')
print(f'Tiempo de recuperación promedio: {mean_retrieval_time} segundos')

# Calcular mAP (Mean Average Precision)
y_true = test_labels
y_scores = []
for i in range(len(test_img_flat)):
    imagen_consulta = test_img_flat[i]
    imagen_consulta = tf.expand_dims(imagen_consulta, axis=0)
    query_features = model.predict([imagen_consulta]).flatten().reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_features)
    nearest_labels = [train_labels_flat[idx] for idx in indices.flatten()]
    y_scores.append(nearest_labels)

mAP = average_precision_score(y_true, y_scores, average='macro')
print(f'Precision promedio: {mean_precision}')
print(f'Recall promedio: {mean_recall}')
print(f'Distancia promedio: {mean_average_distance}')
print(f'Tiempo de recuperación promedio: {mean_retrieval_time} segundos')
print(f'mAP (Mean Average Precision): {mAP}')
