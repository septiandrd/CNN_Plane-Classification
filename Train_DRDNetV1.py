import pandas
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from keras.applications import inception_v3
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, ReLU, add, concatenate, GlobalAveragePooling2D
from keras.models import Model
from datetime import datetime
import tensorflow as tf
import time
import pickle
import os

import telegram
bot = telegram.Bot(token="631334683:AAEKuP9g-WcJ_jJgIvFfaQ99uHs5C5S73nU")

def short_inception(num_filter,inputs) :
  t1 = Conv2D(filters=num_filter, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
  t2 = Conv2D(filters=num_filter, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
  x = concatenate([t1,t2],axis=3)
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  return x

def long_inception_residual(num_filter,inputs) :
  t1 = Conv2D(filters=num_filter, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
  t1 = BatchNormalization(axis=3)(t1)
  t1 = ReLU()(t1)
  t1 = Conv2D(filters=num_filter, kernel_size=(3,3), strides=(1,1), padding='same')(t1)

  t2 = Conv2D(filters=num_filter, kernel_size=(5,5), strides=(1,1), padding='same')(inputs)
  t2 = BatchNormalization(axis=3)(t2)
  t2 = ReLU()(t2)
  t2 = Conv2D(filters=num_filter, kernel_size=(5,5), strides=(1,1), padding='same')(t2)

  x = concatenate([t1,t2],axis=3)
  x = add([inputs,x])
  x = BatchNormalization(axis=3)(x)
  x = ReLU()(x)
  return x

def DRDNetV1(input_shape,num_class) :
  inputs = Input(input_shape)
  x = short_inception(32,inputs)
  x = long_inception_residual(32,x)
  x = short_inception(64,x)
  x = long_inception_residual(64,x)
  x = short_inception(128,x)
  x = long_inception_residual(128,x)
  x = short_inception(256,x)
  x = long_inception_residual(256,x)
  x = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
  x = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')(x)
  x = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(num_class, activation='softmax')(x)

  model = Model(input=inputs, output=predictions)
  return model

if __name__ == '__main__':

    start = time.time()

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zca_whitening=True,
        brightness_range=(0.0,1.0),
        channel_shift_range=5.0,
    )

    validation_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'Dataset/Training',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
    )

    validation_generator = validation_datagen.flow_from_directory(
        'Dataset/Validation',
        target_size=(200, 300),
        batch_size=25,
        class_mode='categorical',
    )

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    arch_name = "DRDNetV1"
    model_name = 'Model_DRDNetV1_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    weight_name = 'Weight_DRDNetV1_' + datetime.now().strftime('%d%m%y') + '.hdf5'
    history_name = 'History_DRDNetV1_' + datetime.now().strftime('%d%m%y')
    model_path = os.path.join(save_dir, model_name)
    weight_path = os.path.join(save_dir, weight_name)
    EPOCH = 20

    model = DRDNetV1(input_shape=(200,300,3), num_class=70)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['acc'])
    model.summary()

    tensorboard = TensorBoard()
    earlystop = EarlyStopping(patience=4)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            save_dir, 'Checkpoint_' +
            datetime.now().strftime('%d%m%y') +
            '_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'),
        save_best_only=True,
        verbose=1)

    with tf.Session() as sess :
      sess.run(tf.global_variables_initializer())
      training = model.fit_generator(
          train_generator,
          steps_per_epoch=1000,
          epochs=EPOCH,
          validation_data=validation_generator,
          callbacks=[checkpoint,tensorboard,earlystop],
      )

      score = model.evaluate_generator(
          validation_generator,
      )

    print('\nLoss \t\t:',score[0])
    print('Accuracy \t:',score[1]*100,'%')

    with open(os.path.join(save_dir,history_name), 'wb') as file:
        pickle.dump(training.history, file)

    end = time.time()

    menit = (end-start)/60

    print("\n"+model_name+
          "\n %i Epoch finished in %.2f minutes"%(EPOCH,menit))

    bot.send_message(chat_id='477030905', text="Training "+arch_name+" finished. "
                        "\nLoss : "+str(score[0])+" Accuracy : "+str(score[1]*100)+"%")