from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
import os

#Directorios de entrenamiento y Pruebas
train_path = os.path.join('fuentes','Train')
test_path = os.path.join('fuentes','Test')

#cantidad de lementos de entrenmiento y prueba
count_train,count_test = 0,0
for d in os.listdir(train_path):
    count_train += len(os.listdir(os.path.join(train_path, d)))
    count_test += len(os.listdir(os.path.join(test_path, d)))

img_size = 32
batch_size = 128
iteraciones = 10

#generar data de entrenamiento
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_path,
                                                           shuffle=True,
                                                           target_size=(img_size,img_size),
                                                           color_mode="grayscale")#grayscale para que no sea rgb/ shuffle para mezclar los datos y no provoque errores

val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_path,
                                                           shuffle=True,
                                                           target_size=(img_size,img_size),
                                                           color_mode="grayscale")

#Generacion de capas de la red neuronal
cnn_model = Sequential()
cnn_model.add(Conv2D(50,(3,3), activation='relu',input_shape=(img_size,img_size,1), name='conv2d_layer'))
cnn_model.add(MaxPooling2D((2,2)))
cnn_model.add(Conv2D(100,(3,3), activation='relu', name='conv2d_layer_2'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Flatten())
cnn_model.add(Dense(1200,activation='relu',name='hidden_layer'))
cnn_model.add(Dense(200,activation='relu',name='hidden_layer_2'))
cnn_model.add(Dense(62,activation='softmax',name='output'))#62: numero de clases que puede tener el input

cnn_model.summary()

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = cnn_model.fit_generator(
    train_data_gen,
    epochs=iteraciones,
    validation_data=val_data_gen,
)

cnn_model.save("caracteres32")