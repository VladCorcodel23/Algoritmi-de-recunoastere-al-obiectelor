import os
import cv2
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from PIL import Image

# Funcție pentru a încărca imaginile și a le redimensiona
def load_images(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))  # Redimensionează imaginile la o dimensiune specificată
        img_array = np.array(img)
        images.append(img_array.flatten())  # Converteste imaginea in vector si adauga la lista de imagini
        labels.append(label)
    return np.array(images), np.array(labels)

# Încarcă datele de antrenare
train_apple_images, train_apple_labels = load_images("dataset/apple", 0)
train_banana_images, train_banana_labels = load_images("dataset/banana", 1)

# Concatenează datele de antrenare
X_train = np.concatenate((train_apple_images, train_banana_images), axis=0)
y_train = np.concatenate((train_apple_labels, train_banana_labels), axis=0)

# Încarcă datele de validare
val_apple_images, val_apple_labels = load_images("validation/apple", 0)
val_banana_images, val_banana_labels = load_images("validation/banana", 1)

# Concatenează datele de validare
X_val = np.concatenate((val_apple_images, val_banana_images), axis=0)
y_val = np.concatenate((val_apple_labels, val_banana_labels), axis=0)

# Încarcă datele de testare
test_apple_images, test_apple_labels = load_images("test/apple", 0)
test_banana_images, test_banana_labels = load_images("test/banana", 1)

# Concatenează datele de testare
X_test = np.concatenate((test_apple_images, test_banana_images), axis=0)
y_test = np.concatenate((test_apple_labels, test_banana_labels), axis=0)

# Definirea hiperparametrilor pe care vrei să-i ajustezi
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Exemplu: hiperparametru alpha pentru clasificatorul MultinomialNB
}

# Inițierea clasificatorului Naive Bayes
clf = MultinomialNB()

# Inițierea căutării în grilă cu validare încrucișată
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


# Antrenează clasificatorul cu cei mai buni hiperparametri
best_clf = grid_search.best_estimator_

# Evaluare pe setul de validare
y_val_pred = best_clf.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Accuracy on validation set: {accuracy_val}")

# Evaluare pe setul de test
y_test_pred = best_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Accuracy on test set: {accuracy_test}")
