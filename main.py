import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Crear el modelo de la red neuronal
model = keras.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(None, 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entrenar el modelo con los datos de entrada
model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_val, y_val))
