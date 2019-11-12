from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import time

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

start = time.time()

## FOR TRAINING WITH AUGMENTATION
data_generator = ImageDataGenerator(rotation_range=10,zoom_range=0.1, width_shift_range=0.2, height_shift_range=0.2)
data_generator.fit(x_train)

model.fit_generator(data_generator.flow(x_train, y_train,batch_size=batch_size),epochs=epochs,validation_data=(x_test, y_test),verbose=1)
model.save_weights('modelaug_'+str(epochs)+'.h5')

# # FOR NORMAL TRAINING
# # model.fit(x_train, y_train,
# #           batch_size=batch_size,
# #           epochs=epochs,
# #           verbose=1,
# #           validation_data=(x_test, y_test))
# # model.save_weights('model_'+str(epochs)+'.h5')

time_taken = time.time()-start

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(f'Training {epochs} epochs took {time_taken/60} minutes')

## TO SEE MODEL PREDICTION OF AUGMENTED DATA 
# from matplotlib import pyplot
# for X_batch, y_batch in data_generator.flow(x_train, y_train, batch_size=9):
# 	# create a grid of 3x3 images
# 	for i in range(0, 9):
# 		pyplot.subplot(330 + 1 + i)
# 		pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
# 		print(model.predict(X_batch[i].reshape(1, 28, 28, 1)))
# 		print(model.predict_classes(X_batch[i].reshape(1, 28, 28, 1)))
# 	# show the plot
# 	pyplot.show()

# run recognizer_keras.py with .h5 saved