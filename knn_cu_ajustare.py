from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


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

# Crează un model K-NN cu scalare și căutare a hiperparametrilor
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier())

# Definirea grilei de hiperparametri
param_grid = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 7, 9]}

# Inițializarea GridSearchCV
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')

# Antrenarea modelului cu căutarea hiperparametrilor
grid_search.fit(X_train, y_train)

# Afișarea celor mai buni hiperparametri
print("Best Hyperparameters:", grid_search.best_params_)

accuracy_train_before_search = knn_model.fit(X_train, y_train).score(X_train, y_train)
print(f'Accuracy on training set: {accuracy_train_before_search}')

# Afișarea acurateții pe setul de validare cu cei mai buni hiperparametri
accuracy_val_with_best_params = grid_search.best_estimator_.score(X_val, y_val)
print(f'Accuracy on validation set with best hyperparameters: {accuracy_val_with_best_params}')

# Evaluează pe setul de testare cu cei mai buni hiperparametri
y_pred_test = grid_search.best_estimator_.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Accuracy on test set with best hyperparameters: {accuracy_test}')

# Acuratețea pe setul de antrenare înainte de căutarea hiperparametrilor
accuracy_train_before_search = knn_model.fit(X_train, y_train).score(X_train, y_train)

# Acuratețea pe setul de validare cu cei mai buni hiperparametri
accuracy_val_with_best_params = grid_search.best_estimator_.score(X_val, y_val)

# Acuratețea pe setul de testare cu cei mai buni hiperparametri
accuracy_test = accuracy_score(y_test, y_pred_test)

# Etichetele pentru seturi
labels = ['Training Before Search', 'Validation with Best Params', 'Test with Best Params']

# Valorile de acuratețe corespunzătoare
accuracies = [accuracy_train_before_search, accuracy_val_with_best_params, accuracy_test]

# Crearea unui grafic de bare
plt.bar(labels, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.ylim([0, 1])  # Limitarea axei y între 0 și 1 pentru acuratețile proporționale

# Adăugarea etichetelor
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.title('Accuracy on Different Datasets')
plt.show()
