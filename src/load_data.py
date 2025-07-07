import numpy as np
import os
import random
from PIL import Image
from tqdm.auto import tqdm

SEED = 42

# Cargar imagen y convertir a vector
def cargar_imagen(path, size=(64, 64)):
    img = Image.open(path).convert('L').resize(size)
    return np.array(img).flatten().astype(np.float32)

# Dividir datos en train y test, balanceados
def split_dataset(healthy_dir, parkinson_dir, test_ratio=0.2, size=(64, 64), seed=SEED): 
    random.seed(seed)

    # Obtener paths de imágenes
    healthy_imgs = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    parkinson_imgs = [os.path.join(parkinson_dir, f) for f in os.listdir(parkinson_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Mezclar
    random.shuffle(healthy_imgs)
    random.shuffle(parkinson_imgs)

    # Split
    split_h = int(len(healthy_imgs) * (1 - test_ratio))
    split_p = int(len(parkinson_imgs) * (1 - test_ratio))

    train_imgs = healthy_imgs[:split_h] + parkinson_imgs[:split_p]
    test_imgs  = healthy_imgs[split_h:] + parkinson_imgs[split_p:]

    y_train = [0]*split_h + [1]*split_p
    y_test  = [0]*(len(healthy_imgs) - split_h) + [1]*(len(parkinson_imgs) - split_p)

    # Cargar imágenes
    X_train = [cargar_imagen(p, size) for p in train_imgs]
    X_test  = [cargar_imagen(p, size) for p in test_imgs]

    # Mezclar
    train = list(zip(X_train, y_train))
    test  = list(zip(X_test, y_test))
    random.shuffle(train)
    random.shuffle(test)
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def normalizar_dataset(X):
    return X / 255.0