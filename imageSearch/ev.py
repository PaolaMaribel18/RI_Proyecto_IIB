import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

# Cargar los datos
train_features = np.load('data/train_features.npy')
train_labels = np.load('data/train_labels.npy')
train_img = np.load('data/train_img.npy')
name_classes = np.load('data/name_classes.npy')
test_features = np.load('data/test_features.npy')
test_labels = np.load('data/test_labels.npy')

# Configurar el modelo de Nearest Neighbors para la recuperación
n_neighbors = 5
nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine').fit(train_features)

# Función para realizar una consulta y obtener los vecinos más cercanos
def search_image(query_feature):
    distances, indices = nn_model.kneighbors([query_feature])
    return indices[0]

# Función para evaluar la precisión y recall en el sistema de recuperación
def evaluate_retrieval(test_features, test_labels, k=5):
    precisions = []
    recalls = []
    
    for i, query_feature in enumerate(test_features):
        true_label = test_labels[i]
        indices = search_image(query_feature)
        retrieved_labels = train_labels[indices]

        # Calcular precision@k y recall@k
        relevant = sum([1 for label in retrieved_labels if label == true_label])
        precision = relevant / k
        recall = relevant / sum(train_labels == true_label)
        
        precisions.append(precision)
        recalls.append(recall)
        
        print(f'Consulta {i+1}: Precision@{k}: {precision:.2f}, Recall@{k}: {recall:.2f}')

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    return avg_precision, avg_recall


### generar el archivo test_features.npy
### general el archivo test_labels.npy= train_labels.npy
# Evaluar el sistema con los datos de test
test_features_flat = np.array([feature.flatten() for feature in test_features])
test_labels_flat = np.array([label for label in test_labels])

avg_precision, avg_recall = evaluate_retrieval(test_features_flat, test_labels_flat, k=n_neighbors)
print(f'Precision@{n_neighbors}: {avg_precision:.2f}, Recall@{n_neighbors}: {avg_recall:.2f}')
