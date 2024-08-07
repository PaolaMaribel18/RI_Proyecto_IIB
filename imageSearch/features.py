import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import tensorflow_datasets as tfds

# Directorio para la descarga
data_dir = r'C:\Users\Paola\Downloads\caltech101'

# Descargar y cargar el dataset
(train_dataset, test_dataset), dataset_info = tfds.load(
    name='caltech101:3.0.2',
    split=['train[:80%]', 'test[:20%]'],
    with_info=True,
    as_supervised=True,
    data_dir=data_dir,
    download=False # Cambiar a True para descargar
)

name_classes = dataset_info.features['label'].names
print ("Número de clases:", len(name_classes))
print ("Clases:", name_classes)
np.save('name_classes.npy', name_classes)

# Preprocesar imágenes
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # InceptionV3 usa 299x299
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess_image).shuffle(1000).batch(32)
test_dataset = test_dataset.map(preprocess_image).batch(32)

# Configurar el modelo InceptionV3
#base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
#model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Función para extraer características e imagenes
def extract_features(dataset):
    features = []
    labels = []
    imgs = []
    for images, lbls in dataset:
        imgs.append(images)
        feature_maps = model.predict(images)
        features.append(feature_maps)
        labels.append(lbls.numpy())
    return features, labels,imgs

# Extraer características para train y test
train_features, train_labels, train_img = extract_features(train_dataset)
test_features, test_labels, test_img = extract_features(test_dataset)

#Indexación
# Aplanar las características para el índice
train_features_flat = np.array([feature.flatten() for batch in train_features for feature in batch])
train_labels_flat = np.array([label for batch in train_labels for label in batch])
train_img_flat = np.array([img for batch in train_img for img in batch])

# Guardar el archivo
np.save('data/train_features.npy', train_features_flat)
np.save('data/train_labels.npy', train_labels_flat)
np.save('data/train_img.npy', train_img_flat)

print("Características y etiquetas guardadas exitosamente.")
