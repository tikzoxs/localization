import tensorflow as tf
import absl.logging as logging
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint
import generate_dataset
import os

H = 120
W = 160
train_H = 60
train_W = 80
batch_size = 16

workdir = "/home/tharindu/Desktop/black/codes/Black/loclization/ckpt"
tensorboard = "/home/tharindu/Desktop/black/codes/Black/loclization/tensorboard"


logging.info('workdir: %s', workdir)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(2))

# #6 layer model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(2))

#2 layer model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

#defining optimizers and loss funtction
model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.3), 
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_squared_error'])

#load weights if available
ckpt = workdir + "/fine_tune.h5"
if(os.path.isfile(ckpt)):
	tf.print("Loading existing wewights found inside: ", ckpt)
	model.load_weights(ckpt)
else:
	tf.print("No wights found. Training a balnk model")


#print model
model.summary()


#checkpoint saving callback
checkpointer = ModelCheckpoint(filepath=ckpt, verbose=1, save_best_only=True)

#learning rate callbacks to check plateau and schedule
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000000001)


#tensorboard callback
# tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard, histogram_freq=0, 
# 	batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
# 	embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

#fitting the model to the data from generator
history = model.fit_generator(
	generate_dataset.readImagesFromTrainFolder(), 
	steps_per_epoch=6500//batch_size, epochs=40,
	verbose=1, callbacks=[checkpointer,reduce_lr], 
	validation_data=generate_dataset.readImagesFromValidationFolder(), validation_steps=1500//batch_size, validation_freq=1, class_weight=None, 
	max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

print('\nhistory dict:', history.history)

model.save(workdir)

model.reset_metrics()
logging.info('Done !!!')

