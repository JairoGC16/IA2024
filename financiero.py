import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.random.rand(1500, 3)

y = np.zeros((1500, 3), dtype=int)

for i in range(1500):
    if X[i, 0] > 0.8 and X[i, 1] > 0.7 and X[i, 2] < 0.4:
        y[i] = [1, 0, 0]
    elif (0.5 <= X[i, 0] <= 0.8 and X[i, 2] < 0.6) or (X[i, 1] > 0.5 and X[i, 2] < 0.7):
        y[i] = [0, 1, 0]
    elif X[i, 0] < 0.5 or X[i, 2] > 0.6:
        y[i] = [0, 0, 1]

validos = np.any(y, axis=1)  # Verificar si al menos una etiqueta activa
X = X[validos]
y = y[validos]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(6, input_dim=3, activation='relu'),  # Capa oculta con 6 neuronas 
    Dense(3, activation='softmax')            # Capa de salida con 3 neuronas
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=150, batch_size=4, verbose=1, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}")

nuevo_cliente = np.array([[0.4, 0.4, 0.8]])  
prediccion = model.predict(nuevo_cliente)
print(f"Predicción para el cliente nuevo: {prediccion}")
print(f"Categoría asignada: {np.argmax(prediccion)}")

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss= history.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()