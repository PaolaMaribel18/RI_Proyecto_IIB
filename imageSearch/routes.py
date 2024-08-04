import os
import numpy as np

def list_images(base_path):
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

base_image_path = 'static/images/caltech101/101_ObjectCategories'
image_paths = list_images(base_image_path)

# Guarda las rutas en un archivo
np.save('data/image_paths.npy', image_paths)
