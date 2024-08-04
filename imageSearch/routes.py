import os
import numpy as np


def list_images(base_path):
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file).replace("\\", "/"))
    return image_paths

base_image_path = 'static/images/caltech101/101_ObjectCategories'
image_paths = list_images(base_image_path)

# Guarda las rutas en un archivo
np.save('data/image_paths.npy', image_paths)

print("Rutas guardadas exitosamente.")
print("Primeras 5 rutas:")
print(image_paths[:5])
