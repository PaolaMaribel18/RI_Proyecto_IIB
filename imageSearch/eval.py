from sklearn.metrics import precision_score, recall_score, average_precision_score
import time
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Load the precomputed features, labels, and images
train_features_flat = np.load('data/train_features.npy')
train_labels_flat = np.load('data/train_labels.npy')
test_img_flat = np.load('data/test_img.npy')
test_labels = np.load('data/test_labels.npy')

# Configure the NearestNeighbors model
n_neighbors = 6
algorithm = 'kd_tree'  # 'ball_tree', 'kd_tree', o 'auto'
metric = 'cosine' if algorithm == 'brute' else None  # Solo se usa 'cosine' para brute-force

nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric).fit(train_features_flat) if metric else NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(train_features_flat)

# Load the VGG16 model with pretrained weights from ImageNet, without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Define the number of neighbors to retrieve
k = n_neighbors

# Evaluate the model using the test set
precisions = []
recalls = []
average_distances = []
retrieval_times = []
detailed_results = []

# Iterate over the test images
for i in range(len(test_img_flat)):
    # Preprocess the query image
    imagen_consulta = test_img_flat[i]
    imagen_consulta = tf.expand_dims(imagen_consulta, axis=0)
    
    # Extract features from the query image
    query_features = model.predict([imagen_consulta]).flatten().reshape(1, -1)
    
    # Calculate the time taken to retrieve the neighbors
    start_time = time.time()
    
    # Find the k nearest neighbors
    distances, indices = nn_model.kneighbors(query_features)
    
    # Get the time taken to retrieve the neighbors
    retrieval_time = time.time() - start_time
    retrieval_times.append(retrieval_time)
    
    # Get the labels of the k nearest neighbors
    nearest_labels = [train_labels_flat[idx] for idx in indices.flatten()]
    
    # Calculate precision and recall at k
    true_label = test_labels[i]
    precision = precision_score([true_label]*k, nearest_labels, average='macro', zero_division=0)
    recall = recall_score([true_label]*k, nearest_labels, average='macro', zero_division=0)
    
    precisions.append(precision)
    recalls.append(recall)
    
    # Calculate the average distance of the k nearest neighbors
    average_distance = np.mean(distances)
    average_distances.append(average_distance)
    
    # Store detailed results for each query
    detailed_results.append(
        f'Consulta {i+1}: Precision@{k}: {precision:.2f}, Recall@{k}: {recall:.2f}, Avg Distance: {average_distance:.2f}, Retrieval Time: {retrieval_time:.4f} segundos'
    )

# Calculate mean precision, recall, average distance, and retrieval time
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_average_distance = np.mean(average_distances)
mean_retrieval_time = np.mean(retrieval_times)

# Calculate mAP (Mean Average Precision)
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

# Print detailed results
for result in detailed_results:
    print(result)

# Print overall metrics
print(f'Precision promedio: {mean_precision}')
print(f'Recall promedio: {mean_recall}')
print(f'Distancia promedio: {mean_average_distance}')
print(f'Tiempo de recuperación promedio: {mean_retrieval_time} segundos')
print(f'mAP (Mean Average Precision): {mAP}')

output_dir = 'result'
output_file = 'resultados_evaluacion2.txt'
output_path = os.path.join(output_dir, output_file)

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Save the evaluation results to a text file
with open(output_path, 'w') as file:
    file.write('Modelo utilizado:\n')
    file.write('VGG16 (sin la capa superior de clasificación, preentrenado en ImageNet)\n')
    file.write('\nParámetros del modelo de KNN:\n')
    file.write(f'Algoritmo: {algorithm}\n')
    if metric:
        file.write(f'Metric: {metric}\n')
    file.write(f'n_neighbors: {n_neighbors}\n')
    file.write('\nResultados de la evaluación:\n')
    file.write(f'Precision promedio: {mean_precision}\n')
    file.write(f'Recall promedio: {mean_recall}\n')
    file.write(f'Distancia promedio: {mean_average_distance}\n')
    file.write(f'Tiempo de recuperación promedio: {mean_retrieval_time} segundos\n')
    file.write(f'mAP (Mean Average Precision): {mAP}\n')
    file.write('\nResultados detallados:\n')
    for result in detailed_results:
        file.write(result + '\n')

print(f'Resultados guardados en {output_path}')
